"""CrossModalLab — Phase 0C.2 Response Bias, Evidence Sensitivity, and Answer Scoring.

Executes comprehensive scientific diagnostic suite:
- Part 1: Full candidate-sequence scoring across 15 pilot cases (Yes vs No vs Undetermined)
- Part 2: Systematic 6-permutation + 2-wording prompt battery (Task C)
- Part 3: 6-condition visual evidence diagnostic battery with image hash verification (Task D)
- Part 4: Matched evidence-insufficiency vs contradiction triad (Task E)
- Part 5: Full resource and memory profiling (Task F)

Outputs structured empirical records to artifacts/evaluations/phase_0c2_response_bias_results.json.
"""

import os
import sys
import json
import time
import hashlib
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

from src.evaluation.candidate_scorer import score_candidate_answers
from scripts.run_corrected_pilot_v2 import parse_completion

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(64 * 1024):
            h.update(chunk)
    return h.hexdigest()

def execute_single_case(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt_text: str,
    candidate_strings: List[str],
    device: str = "cuda:0",
    max_new_tokens: int = 25
) -> Dict[str, Any]:
    """Runs generation and candidate-sequence scoring for a single image-prompt pair."""
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    base_inputs = processor(text=formatted_prompt, images=[image], return_tensors="pt").to(device)
    prompt_len = base_inputs["input_ids"].shape[1]

    # 1. Greedy Generation
    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out_ids = model.generate(
            **base_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    torch.cuda.synchronize()
    gen_time = time.perf_counter() - t0

    new_token_ids = out_ids[0, prompt_len:].tolist()
    completion_text = processor.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    parsed_decision = parse_completion(completion_text)

    # 2. Candidate Sequence Scoring
    t0_score = time.perf_counter()
    cand_scorer_res = score_candidate_answers(
        model=model,
        processor=processor,
        base_inputs=base_inputs,
        candidate_strings=candidate_strings,
        device=device
    )
    score_time = time.perf_counter() - t0_score
    peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 * 1024)

    # Clean intermediate references
    del base_inputs, out_ids
    torch.cuda.empty_cache()

    return {
        "prompt_length_tokens": prompt_len,
        "completion_token_ids": new_token_ids,
        "completion_text": completion_text,
        "parsed_decision": parsed_decision,
        "generation_latency_sec": round(gen_time, 3),
        "scoring_latency_sec": round(score_time, 3),
        "peak_vram_mib": round(peak_vram_mb, 2),
        "candidate_scoring": cand_scorer_res.to_dict()
    }

def main():
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
    revision = "a7da5b986cb59b408707209984f360a5f4ad7e47"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_id} on {device}...", flush=True)
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

    results_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": model_id,
        "revision": revision,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0),
        "experiments": {}
    }

    candidate_set_primary = [" Yes", " No", " Undetermined"]
    candidate_set_terminated = [" Yes.", " No.", " Undetermined."]

    # =========================================================================
    # PART 1: Full Candidate-Sequence Scoring across 15 Pilot Cases (Task B)
    # =========================================================================
    print("=================================================================", flush=True)
    print("PART 1: Full Candidate-Sequence Scoring on 15 Pilot Cases", flush=True)
    print("=================================================================", flush=True)

    manifest_path = Path("datasets/controlled/manifests/controlled_180_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    records_by_id = {r["sample_id"]: r for r in manifest["records"]}

    v1_results_path = Path("artifacts/evaluations/initial_pilot_15_results.json")
    with open(v1_results_path, "r", encoding="utf-8") as f:
        v1_data = json.load(f)
    pilot_15_ids = [r["sample_id"] for r in v1_data["records"]]

    images_dir = Path("datasets/controlled/images")
    canonical_prompt_fmt = (
        "{text_premise}\n\n"
        "Question: {query}\n"
        "Carefully evaluate whether the question is entailed (Yes), contradicted (No), or underdetermined from the evidence (Undetermined).\n"
        "Answer directly with: Yes, No, or Undetermined."
    )

    part1_records = []
    length_flip_count = 0

    for idx, sid in enumerate(pilot_15_ids, 1):
        rec = records_by_id[sid]
        img_path = images_dir / rec["image_file"]
        img = Image.open(img_path).convert("RGB")
        prompt = canonical_prompt_fmt.format(text_premise=rec["text_premise"], query=rec["query"])

        eval_res = execute_single_case(model, processor, img, prompt, candidate_set_primary, device)

        cs = eval_res["candidate_scoring"]
        raw_top = cs["raw_ranking"][0].strip()
        norm_top = cs["length_normalized_ranking"][0].strip()
        is_length_flip = cs["length_rank_flip"]
        if is_length_flip:
            length_flip_count += 1

        part1_records.append({
            "test_index": idx,
            "sample_id": sid,
            "topology": rec["topology"],
            "ground_truth": rec["ground_truth_decision"],
            "epistemic_status": rec["epistemic_status"],
            "image_file": rec["image_file"],
            "image_sha256": rec["image_sha256"],
            **eval_res
        })

        gt = rec["ground_truth_decision"]
        gen_dec = eval_res["parsed_decision"]
        raw_prob_und = cs["candidate_probabilities_raw"][" Undetermined"]
        norm_prob_und = cs["candidate_probabilities_normalized"][" Undetermined"]
        print(f"[{idx:02d}/15] {sid:30} | GT: {gt:12} | Gen: {gen_dec:4} | RawTop: {raw_top:4} | NormTop: {norm_top:4} | P_raw(Und)={raw_prob_und:.5f} | P_norm(Und)={norm_prob_und:.5f} | LengthFlip: {is_length_flip}", flush=True)

    results_payload["experiments"]["part1_candidate_scoring_15_pilot"] = {
        "total_evaluated": len(pilot_15_ids),
        "total_length_rank_flips": length_flip_count,
        "records": part1_records
    }

    # =========================================================================
    # PART 2: Systematic 6-Permutation + 2-Wording Prompt Battery (Task C)
    # =========================================================================
    print("\n=================================================================", flush=True)
    print("PART 2: Systematic 6-Permutation + 2-Wording Prompt Battery", flush=True)
    print("=================================================================", flush=True)

    # 3 Representative cases: 1 Forward, 1 Reverse, 1 Fork
    diag_cases = [
        {"sample_id": "ctrl_forward_chain_0001", "name": "Forward Chain (Entailed, GT: Yes)"},
        {"sample_id": "ctrl_reverse_chain_0038", "name": "Reverse Chain (Contradicted, GT: No)"},
        {"sample_id": "ctrl_underdetermined_fork_0073", "name": "Underdetermined Fork (Underdetermined, GT: Undetermined)"}
    ]

    prompt_variations = [
        {"pid": "P1_yes_no_und", "desc": "Canonical order (Yes, No, Undetermined)", "fmt": "{text_premise}\n\nQuestion: {query}\nCarefully evaluate whether the question is entailed (Yes), contradicted (No), or underdetermined from the evidence (Undetermined).\nAnswer directly with: Yes, No, or Undetermined."},
        {"pid": "P2_yes_und_no", "desc": "Permuted order (Yes, Undetermined, No)", "fmt": "{text_premise}\n\nQuestion: {query}\nCarefully evaluate whether the question is entailed (Yes), underdetermined (Undetermined), or contradicted (No).\nAnswer directly with: Yes, Undetermined, or No."},
        {"pid": "P3_no_yes_und", "desc": "Permuted order (No, Yes, Undetermined)", "fmt": "{text_premise}\n\nQuestion: {query}\nCarefully evaluate whether the question is contradicted (No), entailed (Yes), or underdetermined (Undetermined).\nAnswer directly with: No, Yes, or Undetermined."},
        {"pid": "P4_no_und_yes", "desc": "Permuted order (No, Undetermined, Yes)", "fmt": "{text_premise}\n\nQuestion: {query}\nCarefully evaluate whether the question is contradicted (No), underdetermined (Undetermined), or entailed (Yes).\nAnswer directly with: No, Undetermined, or Yes."},
        {"pid": "P5_und_yes_no", "desc": "Permuted order (Undetermined, Yes, No)", "fmt": "{text_premise}\n\nQuestion: {query}\nCarefully evaluate whether the question is underdetermined (Undetermined), entailed (Yes), or contradicted (No).\nAnswer directly with: Undetermined, Yes, or No."},
        {"pid": "P6_und_no_yes", "desc": "Permuted order (Undetermined, No, Yes)", "fmt": "{text_premise}\n\nQuestion: {query}\nCarefully evaluate whether the question is underdetermined (Undetermined), contradicted (No), or entailed (Yes).\nAnswer directly with: Undetermined, No, or Yes."},
        {"pid": "P7_open_unlisted", "desc": "Open prompt without enumerated options list", "fmt": "{text_premise}\n\nQuestion: {query}\nCarefully evaluate whether the question is entailed, contradicted, or underdetermined from the evidence.\nState your evaluation directly."},
        {"pid": "P8_alternate_wording", "desc": "Alternate lexical wording (Supported, Refuted, Unresolvable)", "fmt": "{text_premise}\n\nQuestion: {query}\nAssess whether the query statement is logically supported (Yes), refuted (No), or cannot be determined from the premises (Undetermined).\nAnswer directly with: Yes, No, or Undetermined."}
    ]

    part2_records = []

    for c in diag_cases:
        sid = c["sample_id"]
        rec = records_by_id[sid]
        img_path = images_dir / rec["image_file"]
        img = Image.open(img_path).convert("RGB")
        print(f"\nEvaluating Prompt Battery on {sid} ({c['name']}):", flush=True)

        case_prompts_log = []
        for pv in prompt_variations:
            prompt_str = pv["fmt"].format(text_premise=rec["text_premise"], query=rec["query"])
            eval_res = execute_single_case(model, processor, img, prompt_str, candidate_set_primary, device)

            p_yes = eval_res["candidate_scoring"]["candidate_probabilities_raw"][" Yes"]
            p_no = eval_res["candidate_scoring"]["candidate_probabilities_raw"][" No"]
            p_und = eval_res["candidate_scoring"]["candidate_probabilities_raw"][" Undetermined"]

            rec_item = {
                "prompt_id": pv["pid"],
                "prompt_desc": pv["desc"],
                "prompt_text": prompt_str,
                **eval_res
            }
            case_prompts_log.append(rec_item)
            print(f"  [{pv['pid']:20}] Gen: '{eval_res['completion_text']:15}' | Parsed: {eval_res['parsed_decision']:12} | P(Yes)={p_yes:.4f}, P(No)={p_no:.4f}, P(Und)={p_und:.6f}", flush=True)

        part2_records.append({
            "sample_id": sid,
            "case_name": c["name"],
            "ground_truth": rec["ground_truth_decision"],
            "prompts": case_prompts_log
        })

    results_payload["experiments"]["part2_prompt_order_controls"] = part2_records

    # =========================================================================
    # PART 3: Visual Evidence Controls (Task D)
    # =========================================================================
    print("\n=================================================================", flush=True)
    print("PART 3: Visual Evidence Controls (Task D)", flush=True)
    print("=================================================================", flush=True)

    diag_manifest_path = Path("datasets/controlled/manifests/diagnostic_controls_manifest.json")
    with open(diag_manifest_path, "r", encoding="utf-8") as f:
        diag_images = json.load(f)

    # Base Query and Text Premise
    fixed_text_premise = "The green cylinder is to the left of the magenta prism."
    fixed_query = "Based on the image and statement, is the cyan box to the left of the magenta prism?"

    # 6 Conditions
    visual_conditions = [
        {
            "condition_id": "cond1_original",
            "description": "Original image (cyan box left, green cylinder right). Visual: cyan < green. Combined: cyan < magenta (ENTAILED)",
            "image_path": Path(diag_images["original"]["path"]),
            "image_sha256": diag_images["original"]["sha256"],
            "text_premise": fixed_text_premise,
            "query": fixed_query,
            "expected_logical_status": "entailed",
            "ground_truth": "Yes"
        },
        {
            "condition_id": "cond2_reversed",
            "description": "Counterfactual reversed layout (green cylinder left, cyan box right). Visual: green < cyan. Combined: green < cyan & green < magenta (UNDERDETERMINED)",
            "image_path": Path(diag_images["reversed"]["path"]),
            "image_sha256": diag_images["reversed"]["sha256"],
            "text_premise": fixed_text_premise,
            "query": fixed_query,
            "expected_logical_status": "underdetermined",
            "ground_truth": "Undetermined"
        },
        {
            "condition_id": "cond3_object_removed",
            "description": "Ablated image (cyan box present, green cylinder REMOVED). Visual premise missing (UNDERDETERMINED)",
            "image_path": Path(diag_images["object_removed"]["path"]),
            "image_sha256": diag_images["object_removed"]["sha256"],
            "text_premise": fixed_text_premise,
            "query": fixed_query,
            "expected_logical_status": "underdetermined",
            "ground_truth": "Undetermined"
        },
        {
            "condition_id": "cond4_blank",
            "description": "Visual erasure (solid blank canvas). Visual premise completely missing (UNDERDETERMINED)",
            "image_path": Path(diag_images["blank"]["path"]),
            "image_sha256": diag_images["blank"]["sha256"],
            "text_premise": fixed_text_premise,
            "query": fixed_query,
            "expected_logical_status": "underdetermined",
            "ground_truth": "Undetermined"
        },
        {
            "condition_id": "cond5_text_only",
            "description": "Text-only version (blank canvas image + both premises explicit in text: cyan < green. green < magenta) (ENTAILED)",
            "image_path": Path(diag_images["blank"]["path"]),
            "image_sha256": diag_images["blank"]["sha256"],
            "text_premise": "The cyan box is to the left of the green cylinder. The green cylinder is to the left of the magenta prism.",
            "query": "Based on the statements, is the cyan box to the left of the magenta prism?",
            "expected_logical_status": "entailed",
            "ground_truth": "Yes"
        },
        {
            "condition_id": "cond6_incorrect_names",
            "description": "Lexical corruption (original image, but query asks about ungrounded entity: purple pyramid) (UNDERDETERMINED)",
            "image_path": Path(diag_images["original"]["path"]),
            "image_sha256": diag_images["original"]["sha256"],
            "text_premise": fixed_text_premise,
            "query": "Based on the image and statement, is the purple pyramid to the left of the magenta prism?",
            "expected_logical_status": "underdetermined",
            "ground_truth": "Undetermined"
        }
    ]

    part3_records = []
    for vc in visual_conditions:
        img = Image.open(vc["image_path"]).convert("RGB")
        prompt = canonical_prompt_fmt.format(text_premise=vc["text_premise"], query=vc["query"])

        eval_res = execute_single_case(model, processor, img, prompt, candidate_set_primary, device)

        cs = eval_res["candidate_scoring"]
        p_yes = cs["candidate_probabilities_raw"][" Yes"]
        p_no = cs["candidate_probabilities_raw"][" No"]
        p_und = cs["candidate_probabilities_raw"][" Undetermined"]

        ll_yes = cs["candidates"][" Yes"]["total_log_likelihood"]
        ll_no = cs["candidates"][" No"]["total_log_likelihood"]

        rec_item = {
            "condition_id": vc["condition_id"],
            "description": vc["description"],
            "image_file": vc["image_path"].name,
            "image_sha256": vc["image_sha256"],
            "ground_truth": vc["ground_truth"],
            "expected_logical_status": vc["expected_logical_status"],
            "log_likelihood_margin_yes_minus_no": round(ll_yes - ll_no, 4),
            **eval_res
        }
        part3_records.append(rec_item)
        print(f"  [{vc['condition_id']:22}] GT: {vc['ground_truth']:12} | Gen: '{eval_res['completion_text']:10}' | Parsed: {eval_res['parsed_decision']:12} | P(Yes)={p_yes:.4f}, P(No)={p_no:.4f}, P(Und)={p_und:.6f} | LL(Yes)-LL(No)={ll_yes-ll_no:+.3f}", flush=True)

    results_payload["experiments"]["part3_visual_evidence_controls"] = part3_records

    # =========================================================================
    # PART 4: Evidence Insufficiency vs Contradiction Dissection (Task E)
    # =========================================================================
    print("\n=================================================================", flush=True)
    print("PART 4: Evidence Insufficiency vs Contradiction Dissection (Task E)", flush=True)
    print("=================================================================", flush=True)

    # Matched Triad: Same Entities (cyan_box, green_cylinder, magenta_prism), Same Image (cyan < green)
    # Fixed Query: "Based on the image and statement, is the cyan box to the left of the magenta prism?"
    triad_image_path = Path(diag_images["original"]["path"])
    triad_img = Image.open(triad_image_path).convert("RGB")

    insufficiency_triad = [
        {
            "condition": "entailed_chain",
            "epistemic_status": "entailed",
            "ground_truth": "Yes",
            "text_premise": "The green cylinder is to the left of the magenta prism.",
            "description": "Forward Chain: cyan < green (image) & green < magenta (text) => cyan < magenta is ENTAILED"
        },
        {
            "condition": "contradicted_reverse",
            "epistemic_status": "contradicted",
            "ground_truth": "No",
            "text_premise": "The magenta prism is to the left of the cyan box.",
            "description": "Contradicted: magenta < cyan (text) directly contradicts query cyan < magenta"
        },
        {
            "condition": "underdetermined_fork",
            "epistemic_status": "underdetermined",
            "ground_truth": "Undetermined",
            "text_premise": "The magenta prism is to the left of the green cylinder.",
            "description": "Underdetermined Fork: cyan < green (image) & magenta < green (text) => cyan ? magenta is UNDERDETERMINED"
        }
    ]

    part4_records = []
    for item in insufficiency_triad:
        prompt = canonical_prompt_fmt.format(text_premise=item["text_premise"], query=fixed_query)
        eval_res = execute_single_case(model, processor, triad_img, prompt, candidate_set_primary, device)

        cs = eval_res["candidate_scoring"]
        p_yes = cs["candidate_probabilities_raw"][" Yes"]
        p_no = cs["candidate_probabilities_raw"][" No"]
        p_und = cs["candidate_probabilities_raw"][" Undetermined"]
        ll_yes = cs["candidates"][" Yes"]["total_log_likelihood"]
        ll_no = cs["candidates"][" No"]["total_log_likelihood"]

        rec_item = {
            "condition": item["condition"],
            "epistemic_status": item["epistemic_status"],
            "ground_truth": item["ground_truth"],
            "description": item["description"],
            "text_premise": item["text_premise"],
            "query": fixed_query,
            "margin_yes_minus_no": round(ll_yes - ll_no, 4),
            **eval_res
        }
        part4_records.append(rec_item)
        print(f"  [{item['condition']:22}] Status: {item['epistemic_status']:15} | GT: {item['ground_truth']:12} | Gen: '{eval_res['completion_text']:10}' | Parsed: {eval_res['parsed_decision']:12} | P(Yes)={p_yes:.4f}, P(No)={p_no:.4f}, P(Und)={p_und:.6f} | Margin={ll_yes-ll_no:+.3f}", flush=True)

    results_payload["experiments"]["part4_evidence_insufficiency_dissection"] = part4_records

    # Save complete JSON artifact
    out_json = Path("artifacts/evaluations/phase_0c2_response_bias_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    print(f"\nAll Phase 0C.2 diagnostic experiments completed. Artifact saved to: {out_json.resolve()}", flush=True)

if __name__ == "__main__":
    main()
