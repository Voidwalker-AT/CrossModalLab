"""CrossModalLab — Phase 0D.1: Matched Behavioral Diagnostics & Instrumentation Suite.

Executes:
Part C: 24 Matched Behavioral Diagnostics across 6 controlled groups:
  - Group 1: Directional Duality & Linguistic Symmetry (left-of vs right-of) [N=4]
  - Group 2: Position Swap vs Naming Swap [N=4]
  - Group 3: Degraded & Missing Evidence Controls (blank, missing source, missing target) [N=4]
  - Group 4: Cross-Modal Reasoning Triad (forward chain, reverse chain, fork) [N=3]
  - Group 5: Individual-Premise Unimodal Controls (unimodal visual, text-only) [N=3]
  - Group 6: Full 6-Way Choice-Order Permutations on Underdetermined Fork [N=6]
Part D: Full Candidate-Sequence Likelihood Scoring (raw and length-normalized) & Greedy Decode.
Part E: Instrumentation Validation on Layer 12:
  - Passive Observation Hook (tensor shape, dtype, CPU cast, logit invariance)
  - In-Flight No-Op Replacement Hook
  - Controlled Non-Identity Perturbation Hook (measuring logit shift without causal attribution)

Hardware: NVIDIA GeForce RTX 2050 (4 GB VRAM ceiling), 4-bit NF4.
Storage: portable cache resolution (E: quarantine preserved when present).
Output: artifacts/evaluations/phase_0d1_matched_diagnostics_results.json
"""

import os
import sys
import json
import time
import math
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple
from PIL import Image, ImageDraw


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

from scripts.generate_controlled_benchmark import ENTITY_PALETTE, CANVAS_SIZE, BG_COLOR, draw_shape
from src.evaluation.candidate_scorer import score_candidate_answers

def patch_smolvlm_inputs_merger():
    """Canonical masked_scatter implementation from parent Idefics3Model."""
    import torch
    from transformers.models.smolvlm import modeling_smolvlm

    def safe_inputs_merger(self, input_ids: torch.LongTensor, inputs_embeds: torch.Tensor, image_hidden_states: torch.Tensor):
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.full((), self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        special_image_mask = special_image_mask.unsqueeze(-1).to(inputs_embeds.device)
        image_hidden_states = image_hidden_states.to(inputs_embeds.device, inputs_embeds.dtype)
        return inputs_embeds.masked_scatter(special_image_mask, image_hidden_states)

    modeling_smolvlm.SmolVLMModel.inputs_merger = safe_inputs_merger

def compute_image_hash(image: Image.Image) -> str:
    """Computes SHA-256 hash of image bytes."""
    if image is None:
        return "None (Text-Only)"
    return hashlib.sha256(image.tobytes()).hexdigest()[:16]

def create_counterfactual_image(left_entity: str, right_entity: str) -> Image.Image:
    """Renders entities with positions swapped horizontally."""
    img = Image.new("RGB", CANVAS_SIZE, color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    draw_shape(draw, ENTITY_PALETTE[right_entity]["shape"], ENTITY_PALETTE[right_entity]["color"], (60, 160, 124, 224))
    draw_shape(draw, ENTITY_PALETTE[left_entity]["shape"], ENTITY_PALETTE[left_entity]["color"], (240, 160, 304, 224))
    return img

def create_single_entity_image(entity: str, at_left: bool = True) -> Image.Image:
    """Renders a single entity on canvas, leaving other half empty."""
    img = Image.new("RGB", CANVAS_SIZE, color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    box = (60, 160, 124, 224) if at_left else (240, 160, 304, 224)
    draw_shape(draw, ENTITY_PALETTE[entity]["shape"], ENTITY_PALETTE[entity]["color"], box)
    return img

def create_blank_canvas() -> Image.Image:
    """Renders blank gray canvas."""
    return Image.new("RGB", CANVAS_SIZE, color=BG_COLOR)

def run_evaluation_step(model, processor, prompt_text: str, image: Image.Image = None, device: str = "cuda:0", max_new_tokens: int = 20):
    """Executes forward pass, candidate scoring, and greedy decoding."""
    import torch

    img_hash = compute_image_hash(image)

    if image is not None:
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}
        ]
    else:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
        ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)

    prompt_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs)
        first_token_logits = outputs.logits[0, -1, :].clone().to(torch.float32)
        first_token_probs = torch.softmax(first_token_logits, dim=-1)

        # Candidate sequence likelihood scoring
        cand_result = score_candidate_answers(
            model=model,
            processor=processor,
            base_inputs=inputs,
            candidate_strings=[" Yes", " No", " Undetermined"],
            device=device
        )

        # Greedy decoding
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    elapsed = time.perf_counter() - t0

    new_token_ids = out_ids[0, prompt_len:].tolist()
    decoded = processor.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

    # Parse simple greedy label
    dec_lower = decoded.lower()
    if "undetermined" in dec_lower:
        greedy_pred = "Undetermined"
    elif "yes" in dec_lower:
        greedy_pred = "Yes"
    elif "no" in dec_lower:
        greedy_pred = "No"
    else:
        greedy_pred = "Ambiguous"

    top1_norm_cand = cand_result.length_normalized_ranking[0].strip()
    top1_raw_cand = cand_result.raw_ranking[0].strip()

    del outputs, first_token_logits, first_token_probs, out_ids
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "image_hash": img_hash,
        "prompt_tokens": prompt_len,
        "completion_tokens": len(new_token_ids),
        "completion_text": decoded,
        "greedy_prediction": greedy_pred,
        "elapsed_sec": round(elapsed, 3),
        "candidate_scores": {
            "top_candidate_length_norm": top1_norm_cand,
            "top_candidate_raw": top1_raw_cand,
            "length_rank_flip": cand_result.length_rank_flip,
            "candidate_probabilities_length_norm": cand_result.candidate_probabilities_normalized,
            "candidate_probabilities_raw": cand_result.candidate_probabilities_raw,
            "raw_log_likelihoods": {k: v.total_log_likelihood for k, v in cand_result.candidates.items()},
            "norm_log_likelihoods": {k: v.length_normalized_log_likelihood for k, v in cand_result.candidates.items()}
        }
    }

def main():
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    from src.config.experiment import get_experiment_config, require_gpu_capabilities
    from src.config.paths import get_datasets_dir

    require_gpu_capabilities()
    exp_cfg = get_experiment_config()
    patch_smolvlm_inputs_merger()

    snapshot_dir = get_smolvlm2_snapshot_dir()
    if not snapshot_dir.exists():
        raise FileNotFoundError(
            f"Pinned model snapshot not found: {snapshot_dir}. "
            "See docs/ASSET_MANIFEST.md. Do not silently substitute another revision."
        )
    device = "cuda:0"

    print("================================================================================")
    print("CrossModalLab — Phase 0D.1: Matched Behavioral Diagnostics (24 Cases)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Model: {exp_cfg.model.model_id} @ {exp_cfg.model.revision} (4-bit NF4)")
    print("================================================================================\n")

    print("[1/3] Loading processor and model...")
    processor = AutoProcessor.from_pretrained(str(snapshot_dir), local_files_only=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=exp_cfg.quantization.load_in_4bit,
        bnb_4bit_quant_type=exp_cfg.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=exp_cfg.quantization.bnb_4bit_use_double_quant
    )
    model = AutoModelForImageTextToText.from_pretrained(
        str(snapshot_dir),
        quantization_config=bnb_config,
        device_map=device,
        _attn_implementation="eager",
        local_files_only=True
    )
    model.eval()
    print("  -> Model ready.\n")

    # Load canonical images
    images_dir = get_datasets_dir() / "controlled" / "images"
    img_canon = Image.open(images_dir / "ctrl_forward_chain_0001.png").convert("RGB") # cyan left, green right
    img_triplet = Image.open(images_dir / "ctrl_unimodal_visual_0109.png").convert("RGB") # magenta < yellow < orange
    img_reverse = Image.open(images_dir / "ctrl_reverse_chain_0037.png").convert("RGB")
    img_fork = Image.open(images_dir / "ctrl_underdetermined_fork_0073.png").convert("RGB")

    diagnostic_cases: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # GROUP 1: Directional Duality & Linguistic Symmetry (N=4)
    # -------------------------------------------------------------------------
    g1 = [
        {
            "id": "case_01_left_direct_true",
            "group": "group_1_directional_duality",
            "description": "True left-of query on canonical image",
            "image": img_canon,
            "prompt": "Is the cyan box to the left of the green cylinder? Answer directly with: Yes, No, or Undetermined.",
            "ground_truth": "Yes"
        },
        {
            "id": "case_02_left_direct_false",
            "group": "group_1_directional_duality",
            "description": "False left-of query on canonical image",
            "image": img_canon,
            "prompt": "Is the green cylinder to the left of the cyan box? Answer directly with: Yes, No, or Undetermined.",
            "ground_truth": "No"
        },
        {
            "id": "case_03_right_direct_true",
            "group": "group_1_directional_duality",
            "description": "Logically equivalent true right-of query on canonical image",
            "image": img_canon,
            "prompt": "Is the green cylinder to the right of the cyan box? Answer directly with: Yes, No, or Undetermined.",
            "ground_truth": "Yes"
        },
        {
            "id": "case_04_right_direct_false",
            "group": "group_1_directional_duality",
            "description": "Logically equivalent false right-of query on canonical image",
            "image": img_canon,
            "prompt": "Is the cyan box to the right of the green cylinder? Answer directly with: Yes, No, or Undetermined.",
            "ground_truth": "No"
        }
    ]
    diagnostic_cases.extend(g1)

    # -------------------------------------------------------------------------
    # GROUP 2: Position Swap vs Naming Swap Controls (N=4)
    # -------------------------------------------------------------------------
    img_swap = create_counterfactual_image("cyan_box", "green_cylinder") # Green left, Cyan right
    g2 = [
        {
            "id": "case_05_position_original",
            "group": "group_2_swap_controls",
            "description": "Canonical image, baseline left query",
            "image": img_canon,
            "prompt": "Is the cyan box to the left of the green cylinder? Answer directly with: Yes, No, or Undetermined.",
            "ground_truth": "Yes"
        },
        {
            "id": "case_06_position_swapped",
            "group": "group_2_swap_controls",
            "description": "Visually swapped canvas, same prompt (evaluates visual sensitivity)",
            "image": img_swap,
            "prompt": "Is the cyan box to the left of the green cylinder? Answer directly with: Yes, No, or Undetermined.",
            "ground_truth": "No"
        },
        {
            "id": "case_07_naming_original",
            "group": "group_2_swap_controls",
            "description": "Canonical image, baseline left query",
            "image": img_canon,
            "prompt": "Is the cyan box to the left of the green cylinder? Answer directly with: Yes, No, or Undetermined.",
            "ground_truth": "Yes"
        },
        {
            "id": "case_08_naming_reversed",
            "group": "group_2_swap_controls",
            "description": "Canonical image, naming inverted in prompt (evaluates linguistic sensitivity)",
            "image": img_canon,
            "prompt": "Is the green cylinder to the left of the cyan box? Answer directly with: Yes, No, or Undetermined.",
            "ground_truth": "No"
        }
    ]
    diagnostic_cases.extend(g2)

    # -------------------------------------------------------------------------
    # GROUP 3: Degraded & Missing Evidence Controls (N=4)
    # -------------------------------------------------------------------------
    img_blank = create_blank_canvas()
    img_cyan_only = create_single_entity_image("cyan_box", at_left=True)
    img_green_only = create_single_entity_image("green_cylinder", at_left=False)

    g3 = [
        {
            "id": "case_09_canonical_evidence_present",
            "group": "group_3_missing_evidence",
            "description": "Both objects present and observable",
            "image": img_canon,
            "prompt": "Is the cyan box to the left of the green cylinder? Answer directly with: Yes, No, or Undetermined.",
            "ground_truth": "Yes"
        },
        {
            "id": "case_10_blank_canvas",
            "group": "group_3_missing_evidence",
            "description": "Blank canvas (zero entities visible; task semantics: insufficient evidence)",
            "image": img_blank,
            "prompt": "Is the cyan box to the left of the green cylinder? Answer directly with: Yes, No, or Undetermined.",
            "ground_truth": "Undetermined"
        },
        {
            "id": "case_11_missing_target_entity",
            "group": "group_3_missing_evidence",
            "description": "Target entity (green cylinder) ablated from scene",
            "image": img_cyan_only,
            "prompt": "Is the cyan box to the left of the green cylinder? Answer directly with: Yes, No, or Undetermined.",
            "ground_truth": "Undetermined"
        },
        {
            "id": "case_12_missing_source_entity",
            "group": "group_3_missing_evidence",
            "description": "Source entity (cyan box) ablated from scene",
            "image": img_green_only,
            "prompt": "Is the cyan box to the left of the green cylinder? Answer directly with: Yes, No, or Undetermined.",
            "ground_truth": "Undetermined"
        }
    ]
    diagnostic_cases.extend(g3)

    # -------------------------------------------------------------------------
    # GROUP 4: Cross-Modal Relational Reasoning Triad (N=3)
    # -------------------------------------------------------------------------
    g4 = [
        {
            "id": "case_13_crossmodal_forward_chain",
            "group": "group_4_crossmodal_triad",
            "description": "Visual: cyan_box < green_cylinder, Text: green_cylinder < magenta_prism. Query: cyan_box < magenta_prism?",
            "image": img_canon,
            "prompt": (
                "The green cylinder is to the left of the magenta prism.\n\n"
                "Question: Based on the image and statement, is the cyan box to the left of the magenta prism?\n"
                "Carefully evaluate whether the question is entailed (Yes), contradicted (No), or underdetermined from the evidence (Undetermined).\n"
                "Answer directly with: Yes, No, or Undetermined."
            ),
            "ground_truth": "Yes"
        },
        {
            "id": "case_14_crossmodal_reverse_chain",
            "group": "group_4_crossmodal_triad",
            "description": "Visual: magenta_prism < green_cylinder, Text: green_cylinder < cyan_box. Query: cyan_box < magenta_prism?",
            "image": img_reverse,
            "prompt": (
                "The green cylinder is to the left of the cyan box.\n\n"
                "Question: Based on the image and statement, is the cyan box to the left of the magenta prism?\n"
                "Carefully evaluate whether the question is entailed (Yes), contradicted (No), or underdetermined from the evidence (Undetermined).\n"
                "Answer directly with: Yes, No, or Undetermined."
            ),
            "ground_truth": "No"
        },
        {
            "id": "case_15_crossmodal_underdetermined_fork",
            "group": "group_4_crossmodal_triad",
            "description": "Visual: cyan_box < green_cylinder, Text: orange_torus < green_cylinder. Query: cyan_box < orange_torus?",
            "image": img_fork,
            "prompt": (
                "The orange torus is to the left of the green cylinder.\n\n"
                "Question: Based on the image and statement, is the cyan box to the left of the orange torus?\n"
                "Carefully evaluate whether the question is entailed (Yes), contradicted (No), or underdetermined from the evidence (Undetermined).\n"
                "Answer directly with: Yes, No, or Undetermined."
            ),
            "ground_truth": "Undetermined"
        }
    ]
    diagnostic_cases.extend(g4)

    # -------------------------------------------------------------------------
    # GROUP 5: Individual-Premise Unimodal Controls (N=3)
    # -------------------------------------------------------------------------
    g5 = [
        {
            "id": "case_16_unimodal_visual_transitive",
            "group": "group_5_unimodal_controls",
            "description": "Image shows magenta < yellow < orange. Query: magenta < orange? (0 text premises)",
            "image": img_triplet,
            "prompt": (
                "Question: Based on the image, is the magenta prism to the left of the orange torus?\n"
                "Carefully evaluate whether the question is entailed (Yes), contradicted (No), or underdetermined from the evidence (Undetermined).\n"
                "Answer directly with: Yes, No, or Undetermined."
            ),
            "ground_truth": "Yes"
        },
        {
            "id": "case_17_unimodal_visual_contradicted",
            "group": "group_5_unimodal_controls",
            "description": "Image shows magenta < yellow < orange. Query: orange < magenta? (0 text premises)",
            "image": img_triplet,
            "prompt": (
                "Question: Based on the image, is the orange torus to the left of the magenta prism?\n"
                "Carefully evaluate whether the question is entailed (Yes), contradicted (No), or underdetermined from the evidence (Undetermined).\n"
                "Answer directly with: Yes, No, or Undetermined."
            ),
            "ground_truth": "No"
        },
        {
            "id": "case_18_unimodal_text_transitive",
            "group": "group_5_unimodal_controls",
            "description": "Text: cyan < green, green < magenta. Query: cyan < magenta? (0 images)",
            "image": None,
            "prompt": (
                "The cyan box is to the left of the green cylinder.\n"
                "The green cylinder is to the left of the magenta prism.\n\n"
                "Question: Based on the statements, is the cyan box to the left of the magenta prism?\n"
                "Carefully evaluate whether the question is entailed (Yes), contradicted (No), or underdetermined from the evidence (Undetermined).\n"
                "Answer directly with: Yes, No, or Undetermined."
            ),
            "ground_truth": "Yes"
        }
    ]
    diagnostic_cases.extend(g5)

    # -------------------------------------------------------------------------
    # GROUP 6: Full 6-Way Choice-Order Permutations on Fork (N=6)
    # -------------------------------------------------------------------------
    fork_preamble = "The orange torus is to the left of the green cylinder."
    fork_q = "Based on the image and statement, is the cyan box to the left of the orange torus?"
    permutations = [
        ("YNU", "Yes, No, or Undetermined", "entailed (Yes), contradicted (No), or underdetermined from the evidence (Undetermined)"),
        ("YUN", "Yes, Undetermined, or No", "entailed (Yes), underdetermined from the evidence (Undetermined), or contradicted (No)"),
        ("NYU", "No, Yes, or Undetermined", "contradicted (No), entailed (Yes), or underdetermined from the evidence (Undetermined)"),
        ("NUY", "No, Undetermined, or Yes", "contradicted (No), underdetermined from the evidence (Undetermined), or entailed (Yes)"),
        ("UYN", "Undetermined, Yes, or No", "underdetermined from the evidence (Undetermined), entailed (Yes), or contradicted (No)"),
        ("UNY", "Undetermined, No, or Yes", "underdetermined from the evidence (Undetermined), contradicted (No), or entailed (Yes)")
    ]

    g6 = []
    for code, order_str, expl_str in permutations:
        g6.append({
            "id": f"case_{len(diagnostic_cases) + len(g6) + 1:02d}_order_{code}",
            "group": "group_6_choice_order_spectrum",
            "description": f"Underdetermined fork with candidate order: [{code}] {order_str}",
            "image": img_fork,
            "prompt": (
                f"{fork_preamble}\n\n"
                f"Question: {fork_q}\n"
                f"Carefully evaluate whether the question is {expl_str}.\n"
                f"Answer directly with: {order_str}."
            ),
            "ground_truth": "Undetermined",
            "permutation_code": code
        })
    diagnostic_cases.extend(g6)

    print(f"[2/3] Executing 24 Matched Behavioral Diagnostics (N={len(diagnostic_cases)})...")
    results_records = []
    group_stats: Dict[str, Dict[str, Any]] = {}

    for idx, c in enumerate(diagnostic_cases, start=1):
        print(f"  [{idx:02d}/24] {c['id']} ({c['group']})...", flush=True)
        res = run_evaluation_step(model, processor, c["prompt"], image=c["image"], device=device)

        is_correct = (res["greedy_prediction"] == c["ground_truth"])
        record = {
            "case_id": c["id"],
            "group": c["group"],
            "description": c["description"],
            "ground_truth": c["ground_truth"],
            "greedy_prediction": res["greedy_prediction"],
            "completion_text": res["completion_text"],
            "is_correct": is_correct,
            "image_hash": res["image_hash"],
            "elapsed_sec": res["elapsed_sec"],
            "candidate_scores": res["candidate_scores"]
        }
        results_records.append(record)

        grp = c["group"]
        if grp not in group_stats:
            group_stats[grp] = {"total": 0, "correct": 0, "predictions": []}
        group_stats[grp]["total"] += 1
        if is_correct:
            group_stats[grp]["correct"] += 1
        group_stats[grp]["predictions"].append(res["greedy_prediction"])

        print(f"      -> Expected: {c['ground_truth']:<12} | Greedy: {res['greedy_prediction']:<12} | Top Cand: {res['candidate_scores']['top_candidate_length_norm']:<12} | Correct: {is_correct}")

    # -------------------------------------------------------------------------
    # PART E: Instrumentation Validation on Layer 12
    # -------------------------------------------------------------------------
    print("\n[3/3] PART E: Instrumentation Validation on Layer 12...")
    target_layer_idx = 12
    target_layer = model.model.text_model.layers[target_layer_idx]

    hook_img = Image.open("datasets/controlled/images/ctrl_forward_chain_0001.png").convert("RGB")
    hook_prompt = "The green cylinder is to the left of the magenta prism.\n\nQuestion: Is the cyan box to the left of the magenta prism?\nAnswer directly with: Yes, No, or Undetermined."
    hook_messages = [{"role": "user", "content": [{"type": "image", "image": hook_img}, {"type": "text", "text": hook_prompt}]}]
    hook_inputs = processor.apply_chat_template(hook_messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(device)

    # 1. Baseline unhooked pass
    with torch.inference_mode():
        unhooked_out = model(**hook_inputs)
        base_logits = unhooked_out.logits[0, -1, :].clone()

    # 2. Passive observation hook
    passive_data = {}
    def passive_hook(module, args, output):
        act = output[0] if isinstance(output, tuple) else output
        passive_data["raw_gpu_dtype"] = str(act.dtype)
        passive_data["raw_gpu_device"] = str(act.device)
        passive_data["shape"] = list(act.shape)
        # Detach and move to CPU with float32 cast
        cpu_act = act.detach().cpu().to(torch.float32)
        passive_data["cpu_dtype"] = str(cpu_act.dtype)
        passive_data["mean"] = round(cpu_act.mean().item(), 4)
        passive_data["std"] = round(cpu_act.std().item(), 4)
        passive_data["has_nan"] = torch.isnan(cpu_act).any().item()
        passive_data["has_inf"] = torch.isinf(cpu_act).any().item()

    h_passive = target_layer.register_forward_hook(passive_hook)
    with torch.inference_mode():
        passive_out = model(**hook_inputs)
        passive_logits = passive_out.logits[0, -1, :].clone()
    h_passive.remove()

    passive_diff = (base_logits - passive_logits).abs().max().item()

    # 3. In-flight no-op replacement hook
    def noop_hook(module, args, output):
        act = output[0] if isinstance(output, tuple) else output
        cloned_act = act.clone()
        if isinstance(output, tuple):
            return (cloned_act, *output[1:])
        return cloned_act

    h_noop = target_layer.register_forward_hook(noop_hook)
    with torch.inference_mode():
        noop_out = model(**hook_inputs)
        noop_logits = noop_out.logits[0, -1, :].clone()
    h_noop.remove()

    noop_diff = (base_logits - noop_logits).abs().max().item()

    # 4. Controlled non-identity perturbation hook (Engineering controllability test)
    # Adds a small bounded perturbation to the final token coordinate 0: delta = +0.5
    pert_data = {}
    def perturbation_hook(module, args, output):
        act = output[0] if isinstance(output, tuple) else output
        pert_act = act.clone()
        pert_act[:, -1, 0] = pert_act[:, -1, 0] + 0.5
        pert_data["applied_delta"] = 0.5
        pert_data["perturbed_shape"] = list(pert_act.shape)
        if isinstance(output, tuple):
            return (pert_act, *output[1:])
        return pert_act

    h_pert = target_layer.register_forward_hook(perturbation_hook)
    with torch.inference_mode():
        pert_out = model(**hook_inputs)
        pert_logits = pert_out.logits[0, -1, :].clone()
    h_pert.remove()

    pert_diff = (base_logits - pert_logits).abs().max().item()
    pert_responsive = pert_diff > 1e-4

    instrumentation_report = {
        "target_module": f"model.model.text_model.layers[{target_layer_idx}]",
        "passive_hook": {
            "tensor_shape": passive_data["shape"],
            "raw_gpu_dtype": passive_data["raw_gpu_dtype"],
            "cpu_transferred_dtype": passive_data["cpu_dtype"],
            "mean": passive_data["mean"],
            "std": passive_data["std"],
            "has_nan": passive_data["has_nan"],
            "has_inf": passive_data["has_inf"],
            "max_logit_difference_vs_unhooked": passive_diff,
            "logits_identical": (passive_diff == 0.0)
        },
        "noop_replacement_hook": {
            "max_logit_difference_vs_unhooked": noop_diff,
            "logits_identical": (noop_diff == 0.0)
        },
        "controlled_perturbation_hook": {
            "perturbation_description": "Added delta = +0.5 to token[-1], coordinate 0 at Layer 12",
            "max_logit_difference": pert_diff,
            "controllability_confirmed": pert_responsive,
            "epistemic_qualification": "ENGINEERING CONTROLLABILITY ONLY: Confirms in-flight tensor modification path; does NOT constitute causal attribution of reasoning circuits."
        },
        "status": "PASS" if passive_diff == 0.0 and noop_diff == 0.0 and pert_responsive else "FAIL"
    }

    print(f"  Passive Hook Logit Max Diff:     {passive_diff:.6f} (GPU dtype: {passive_data['raw_gpu_dtype']}, CPU dtype: {passive_data['cpu_dtype']})")
    print(f"  No-Op Hook Logit Max Diff:       {noop_diff:.6f}")
    print(f"  Perturbation Logit Max Shift:    {pert_diff:.4f} (Responsive: {pert_responsive})")
    print(f"  Instrumentation Verdict:         {instrumentation_report['status']}")

    # Compile Final Structured Result
    total_correct = sum(1 for r in results_records if r["is_correct"])
    final_output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
            "revision": "482adb537c021c86670beed01cd58990d01e72e4",
            "quantization": "4-bit NF4 (bitsandbytes)",
            "total_diagnostic_cases": len(diagnostic_cases),
            "total_correct": total_correct,
            "overall_accuracy": f"{total_correct}/{len(diagnostic_cases)} ({total_correct / len(diagnostic_cases) * 100:.1f}%)"
        },
        "group_summary": group_stats,
        "diagnostic_cases": results_records,
        "instrumentation_validation": instrumentation_report
    }

    out_file = Path("artifacts/evaluations/phase_0d1_matched_diagnostics_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(final_output, indent=2), encoding="utf-8")
    print(f"\nPhase 0D.1 Matched Diagnostics complete. Overall Accuracy: {final_output['metadata']['overall_accuracy']}")
    print(f"Detailed results saved to: {out_file.resolve()}")

if __name__ == "__main__":
    main()
