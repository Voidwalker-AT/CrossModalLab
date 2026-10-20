"""CrossModalLab — Phase 0C.1 Corrected Pilot Re-evaluation (V2).

Executes a rigorously audited and confound-free evaluation of SmolVLM-500M-Instruct
on the exact 15 cases from Phase 0C:
- Explicit token slicing: new_tokens = out_ids[0, prompt_len:]
- Pure completion decoding: decode ONLY new_tokens
- Parsing strictly isolated from prompt text
- First-token logit extraction: logit & probability tracking for Yes vs No vs Undetermined
- Choice-order sensitivity test: evaluates both canonical (Yes/No/Und) and counter-order (No/Yes/Und)
- Direct comparison with V1 baseline (artifacts/evaluations/initial_pilot_15_results.json)
- Preserves V1 artifacts untouched; outputs to artifacts/evaluations/pilot_15_corrected_v2.json
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.paths import (
    apply_default_cache_environment,
    get_hf_home,
    get_smolvlm2_snapshot_dir,
)

# Preserve legacy E: cache when present; honor env overrides; portable elsewhere.
apply_default_cache_environment()



def parse_completion(text: str) -> str:
    """Extracts Yes, No, or Undetermined strictly from newly generated completion."""
    cleaned = text.strip().lower()
    tokens = [t.strip(".,;:!? \n\t\"'") for t in cleaned.split()]
    if not tokens:
        return "Unparseable"
    
    first = tokens[0]
    if first in ["yes", "yeah", "correct", "true"]:
        return "Yes"
    if first in ["no", "nope", "incorrect", "false"]:
        return "No"
    if first in ["undetermined", "unknown", "uncertain", "unclear", "undecidable"]:
        return "Undetermined"
    
    # Substring search if first token was introductory
    if any(w in cleaned for w in ["undetermined", "cannot be determined", "cannot tell", "not enough information", "insufficient"]):
        return "Undetermined"
    if "yes" in tokens and "no" not in tokens:
        return "Yes"
    if "no" in tokens and "yes" not in tokens:
        return "No"
    
    return "Unparseable"

def get_candidate_token_ids(processor):
    tok = processor.tokenizer
    return {
        "yes_space": tok.encode(" Yes", add_special_tokens=False)[0],
        "yes_nospace": tok.encode("Yes", add_special_tokens=False)[0],
        "no_space": tok.encode(" No", add_special_tokens=False)[0],
        "no_nospace": tok.encode("No", add_special_tokens=False)[0],
        "und_space": tok.encode(" Und", add_special_tokens=False)[0],
        "und_nospace": tok.encode("Und", add_special_tokens=False)[0],
    }

def run_pilot_v2():
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
    revision = "a7da5b986cb59b408707209984f360a5f4ad7e47"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_id} (revision {revision}) on {device}...", flush=True)
    t0_load = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id, revision=revision, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",
        local_files_only=True
    ).to(device)
    model.eval()
    print(f"Model loaded in {time.perf_counter() - t0_load:.2f}s.\n", flush=True)

    cand_ids = get_candidate_token_ids(processor)

    # Load V1 baseline results for exact matching
    v1_results_path = Path("artifacts/evaluations/initial_pilot_15_results.json")
    with open(v1_results_path, "r", encoding="utf-8") as f:
        v1_data = json.load(f)
    v1_records_by_id = {r["sample_id"]: r for r in v1_data["records"]}

    # Load 180 controlled benchmark manifest
    manifest_path = Path("datasets/controlled/manifests/controlled_180_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    all_records = {r["sample_id"]: r for r in manifest["records"]}

    target_sample_ids = [r["sample_id"] for r in v1_data["records"]]
    print(f"Evaluating {len(target_sample_ids)} target cases with corrected pipeline.\n", flush=True)

    images_base_dir = Path("datasets/controlled/images")

    prompt_template_canonical = (
        "{text_premise}\n\n"
        "Question: {query}\n"
        "Carefully evaluate whether the question is entailed (Yes), contradicted (No), or underdetermined from the evidence (Undetermined).\n"
        "Answer directly with: Yes, No, or Undetermined."
    )

    prompt_template_permuted = (
        "{text_premise}\n\n"
        "Question: {query}\n"
        "Carefully evaluate whether the question is contradicted (No), entailed (Yes), or underdetermined from the evidence (Undetermined).\n"
        "Answer directly with: No, Yes, or Undetermined."
    )

    decoding_config = {
        "max_new_tokens": 25,
        "do_sample": False,
        "pad_token_id": processor.tokenizer.eos_token_id
    }

    results_canonical = []
    results_permuted = []

    correct_count_v2 = 0
    abstention_failures_v2 = 0
    deductive_errors_v2 = 0

    for idx, sid in enumerate(target_sample_ids, 1):
        sample = all_records[sid]
        top = sample["topology"]
        gt_decision = sample["ground_truth_decision"]
        epistemic_status = sample["epistemic_status"]
        img_fn = sample["image_file"]
        img_path = images_base_dir / img_fn
        text_premise = sample["text_premise"]
        query = sample["query"]

        img = Image.open(img_path).convert("RGB")

        # --- A: CANONICAL PROMPT ---
        user_text_canon = prompt_template_canonical.format(text_premise=text_premise, query=query)
        msg_canon = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text_canon}]}
        ]
        prompt_canon = processor.apply_chat_template(msg_canon, add_generation_prompt=True)
        inputs_canon = processor(text=prompt_canon, images=[img], return_tensors="pt").to(device)
        prompt_len_canon = inputs_canon["input_ids"].shape[1]

        torch.cuda.reset_peak_memory_stats(0)
        t0_inf = time.perf_counter()
        with torch.inference_mode():
            # 1. Forward pass for exact first-token logits
            outputs_canon = model(**inputs_canon)
            first_logits = outputs_canon.logits[0, -1, :].clone().to(torch.float32)
            first_probs = torch.softmax(first_logits, dim=-1)

            # 2. Generate completion
            out_ids_canon = model.generate(**inputs_canon, **decoding_config)
        torch.cuda.synchronize()
        latency_sec = time.perf_counter() - t0_inf
        peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 * 1024)

        # Explicit token slicing
        new_token_ids_canon = out_ids_canon[0, prompt_len_canon:].tolist()
        # Decode pure completion
        completion_str_canon = processor.tokenizer.decode(new_token_ids_canon, skip_special_tokens=True).strip()
        parsed_decision_canon = parse_completion(completion_str_canon)

        # Candidate stats
        p_yes = (first_probs[cand_ids["yes_space"]] + first_probs[cand_ids["yes_nospace"]]).item()
        p_no = (first_probs[cand_ids["no_space"]] + first_probs[cand_ids["no_nospace"]]).item()
        p_und = (first_probs[cand_ids["und_space"]] + first_probs[cand_ids["und_nospace"]]).item()

        logit_yes = max(first_logits[cand_ids["yes_space"]].item(), first_logits[cand_ids["yes_nospace"]].item())
        logit_no = max(first_logits[cand_ids["no_space"]].item(), first_logits[cand_ids["no_nospace"]].item())
        logit_und = max(first_logits[cand_ids["und_space"]].item(), first_logits[cand_ids["und_nospace"]].item())

        topk_vals, topk_inds = torch.topk(first_probs, k=3)
        top3_tokens = [
            {"token_str": processor.tokenizer.decode([topk_inds[i].item()]), "prob": round(topk_vals[i].item(), 4)}
            for i in range(3)
        ]

        conforms_canon = (parsed_decision_canon == gt_decision)
        if conforms_canon:
            correct_count_v2 += 1
            fail_cat = "none"
        else:
            if top == "underdetermined_fork" and parsed_decision_canon in ["Yes", "No"]:
                abstention_failures_v2 += 1
                fail_cat = "abstention_failure"
            else:
                deductive_errors_v2 += 1
                fail_cat = "deductive_error"

        # --- B: PERMUTED PROMPT (No, Yes, Undetermined) ---
        user_text_perm = prompt_template_permuted.format(text_premise=text_premise, query=query)
        msg_perm = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text_perm}]}
        ]
        prompt_perm = processor.apply_chat_template(msg_perm, add_generation_prompt=True)
        inputs_perm = processor(text=prompt_perm, images=[img], return_tensors="pt").to(device)
        prompt_len_perm = inputs_perm["input_ids"].shape[1]

        with torch.inference_mode():
            outputs_perm = model(**inputs_perm)
            first_logits_perm = outputs_perm.logits[0, -1, :].clone().to(torch.float32)
            first_probs_perm = torch.softmax(first_logits_perm, dim=-1)
            out_ids_perm = model.generate(**inputs_perm, **decoding_config)

        new_token_ids_perm = out_ids_perm[0, prompt_len_perm:].tolist()
        completion_str_perm = processor.tokenizer.decode(new_token_ids_perm, skip_special_tokens=True).strip()
        parsed_decision_perm = parse_completion(completion_str_perm)

        p_yes_perm = (first_probs_perm[cand_ids["yes_space"]] + first_probs_perm[cand_ids["yes_nospace"]]).item()
        p_no_perm = (first_probs_perm[cand_ids["no_space"]] + first_probs_perm[cand_ids["no_nospace"]]).item()
        p_und_perm = (first_probs_perm[cand_ids["und_space"]] + first_probs_perm[cand_ids["und_nospace"]]).item()

        # V1 baseline record comparison
        v1_rec = v1_records_by_id[sid]
        v1_decision = v1_rec["parsed_decision"]
        v1_conforms = v1_rec["conforms_to_ground_truth"]

        is_flipped_by_prompt = (parsed_decision_canon != parsed_decision_perm)

        rec_entry = {
            "test_index": idx,
            "sample_id": sid,
            "topology": top,
            "ground_truth_decision": gt_decision,
            "epistemic_status": epistemic_status,
            "image_file": img_fn,
            "image_sha256": sample["image_sha256"],
            "text_premise": text_premise,
            "query": query,
            "v1_baseline": {
                "raw_output": v1_rec["raw_output"],
                "parsed_decision": v1_decision,
                "conforms": v1_conforms
            },
            "v2_canonical": {
                "prompt_tokens": prompt_len_canon,
                "completion_tokens": len(new_token_ids_canon),
                "completion_token_ids": new_token_ids_canon,
                "decoded_completion": completion_str_canon,
                "parsed_decision": parsed_decision_canon,
                "conforms": conforms_canon,
                "failure_category": fail_cat,
                "latency_sec": round(latency_sec, 3),
                "peak_vram_mib": round(peak_vram_mb, 2),
                "first_token_probabilities": {
                    "P(Yes)": round(p_yes, 5),
                    "P(No)": round(p_no, 5),
                    "P(Undetermined)": round(p_und, 6)
                },
                "first_token_logits": {
                    "logit(Yes)": round(logit_yes, 3),
                    "logit(No)": round(logit_no, 3),
                    "logit(Undetermined)": round(logit_und, 3),
                    "margin_yes_minus_no": round(logit_yes - logit_no, 3),
                    "margin_yes_minus_und": round(logit_yes - logit_und, 3)
                },
                "top3_tokens": top3_tokens
            },
            "v2_permuted_order": {
                "prompt_order": "No, Yes, or Undetermined",
                "decoded_completion": completion_str_perm,
                "parsed_decision": parsed_decision_perm,
                "conforms": (parsed_decision_perm == gt_decision),
                "first_token_probabilities": {
                    "P(Yes)": round(p_yes_perm, 5),
                    "P(No)": round(p_no_perm, 5),
                    "P(Undetermined)": round(p_und_perm, 6)
                }
            },
            "diagnostic_flags": {
                "v1_v2_canonical_identical": (v1_decision == parsed_decision_canon),
                "choice_order_sensitive": is_flipped_by_prompt,
                "undetermined_suppressed": (p_und < 0.01)
            }
        }
        results_canonical.append(rec_entry)

        status_mark = "PASS" if conforms_canon else "FAIL"
        flip_mark = "FLIPPED_BY_ORDER" if is_flipped_by_prompt else "STABLE"
        print(f"[{idx:02d}/15] [{status_mark}] {sid} ({top}) | GT: {gt_decision:12} | V1: {v1_decision:4} | V2_canon: {parsed_decision_canon:4} (P_yes={p_yes:.3f}, P_no={p_no:.3f}) | V2_perm: {parsed_decision_perm:4} [{flip_mark}]", flush=True)

    target_topologies = [
        "forward_chain",
        "reverse_chain",
        "underdetermined_fork",
        "unimodal_visual",
        "unimodal_text"
    ]

    by_top = {}
    for top in target_topologies:
        recs = [r for r in results_canonical if r["topology"] == top]
        by_top[top] = {
            "n": len(recs),
            "v1_conforming": sum(1 for r in recs if r["v1_baseline"]["conforms"]),
            "v2_canon_conforming": sum(1 for r in recs if r["v2_canonical"]["conforms"]),
            "v2_canon_accuracy": round(sum(1 for r in recs if r["v2_canonical"]["conforms"]) / len(recs), 3),
            "decisions_canon": [r["v2_canonical"]["parsed_decision"] for r in recs],
            "decisions_perm": [r["v2_permuted_order"]["parsed_decision"] for r in recs],
            "order_flipped_count": sum(1 for r in recs if r["diagnostic_flags"]["choice_order_sensitive"]),
            "mean_p_yes": round(sum(r["v2_canonical"]["first_token_probabilities"]["P(Yes)"] for r in recs) / len(recs), 4),
            "mean_p_no": round(sum(r["v2_canonical"]["first_token_probabilities"]["P(No)"] for r in recs) / len(recs), 4),
            "mean_p_und": round(sum(r["v2_canonical"]["first_token_probabilities"]["P(Undetermined)"] for r in recs) / len(recs), 6)
        }

    v2_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": model_id,
        "revision": revision,
        "total_evaluated": len(target_sample_ids),
        "v1_total_conforming": sum(1 for r in results_canonical if r["v1_baseline"]["conforms"]),
        "v2_total_conforming": correct_count_v2,
        "v2_overall_accuracy": round(correct_count_v2 / len(target_sample_ids), 4),
        "abstention_failures": abstention_failures_v2,
        "deductive_errors": deductive_errors_v2,
        "total_order_flipped_cases": sum(1 for r in results_canonical if r["diagnostic_flags"]["choice_order_sensitive"]),
        "by_topology": by_top,
        "records": results_canonical
    }

    out_file = Path("artifacts/evaluations/pilot_15_corrected_v2.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(v2_summary, indent=2), encoding="utf-8")
    print(f"\nCorrected pilot V2 results saved to: {out_file.resolve()}", flush=True)

if __name__ == "__main__":
    run_pilot_v2()
