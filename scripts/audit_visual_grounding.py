"""CrossModalLab — Visual Grounding and Prompt Bias Audit Battery.

Phase 0C.1 Experimental Confound Diagnostic:
Tests SmolVLM-500M-Instruct on:
1. Entity Identity Recognition: Color and shape recognition of left/right objects.
2. Direct Pairwise Spatial Relations: Left-of and right-of on rendered pairwise images without reasoning chains.
3. Image-Swap Counterfactual Control: Swapping visual entities while keeping prompt constant.
4. Triplet Image Perception: Adjacent vs distant pairs in 3-entity images.
5. Preamble Confound Check: Testing effect of "No additional statement is provided."
6. Choice-Order Sensitivity: Permuting (Yes, No, Undetermined) options in prompt.
7. First-Token Logit Extraction: Precise logit margin analysis for Yes vs No vs Undetermined.

Hardware: RTX 2050 (4 GB VRAM ceiling), bfloat16, batch size 1.
Cache: resolved via HF_HOME / CROSSMODAL_HF_HOME.
Output: artifacts/evaluations/visual_grounding_audit_results.json.
"""

import os
import sys
import json
import time
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

def get_candidate_token_ids(processor):
    """Finds token IDs for Yes, No, Undetermined variants."""
    tok = processor.tokenizer
    return {
        "yes_space": tok.encode(" Yes", add_special_tokens=False)[0],
        "yes_nospace": tok.encode("Yes", add_special_tokens=False)[0],
        "no_space": tok.encode(" No", add_special_tokens=False)[0],
        "no_nospace": tok.encode("No", add_special_tokens=False)[0],
        "und_space": tok.encode(" Und", add_special_tokens=False)[0],
        "und_nospace": tok.encode("Und", add_special_tokens=False)[0],
    }

def run_single_inference(model, processor, image: Image.Image, prompt_text: str, device: str):
    """Executes single inference, returning generated completion, first-token logits, and latency."""
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
    inputs = processor(text=formatted_prompt, images=[image], return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.inference_mode():
        # First get 1-step logits for first token analysis
        outputs = model(**inputs)
        first_token_logits = outputs.logits[0, -1, :].clone().to(torch.float32)
        first_token_probs = torch.softmax(first_token_logits, dim=-1)

        # Now generate completion tokens
        out_ids = model.generate(
            **inputs,
            max_new_tokens=25,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    elapsed = time.perf_counter() - t0

    # Strictly separate newly generated tokens from prompt tokens
    new_token_ids = out_ids[0, prompt_len:].tolist()
    decoded_completion = processor.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

    cand_ids = get_candidate_token_ids(processor)
    cand_stats = {}
    for k, tid in cand_ids.items():
        cand_stats[k] = {
            "token_id": tid,
            "token_str": processor.tokenizer.decode([tid]),
            "logit": round(first_token_logits[tid].item(), 3),
            "prob": round(first_token_probs[tid].item(), 6)
        }

    topk_vals, topk_inds = torch.topk(first_token_probs, k=5)
    top5_tokens = [
        {
            "token_id": topk_inds[i].item(),
            "token_str": processor.tokenizer.decode([topk_inds[i].item()]),
            "prob": round(topk_vals[i].item(), 4),
            "logit": round(first_token_logits[topk_inds[i].item()].item(), 3)
        }
        for i in range(5)
    ]

    return {
        "prompt_length_tokens": prompt_len,
        "completion_length_tokens": len(new_token_ids),
        "completion_token_ids": new_token_ids,
        "decoded_completion": decoded_completion,
        "elapsed_seconds": round(elapsed, 3),
        "top5_first_tokens": top5_tokens,
        "candidate_stats": cand_stats
    }

def create_counterfactual_image(left_entity: str, right_entity: str) -> Image.Image:
    """Renders an image where entities are horizontally swapped."""
    img = Image.new("RGB", CANVAS_SIZE, color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    # Put left_entity at right [240, 160], right_entity at left [60, 160]
    draw_shape(draw, ENTITY_PALETTE[right_entity]["shape"], ENTITY_PALETTE[right_entity]["color"], (60, 160, 124, 224))
    draw_shape(draw, ENTITY_PALETTE[left_entity]["shape"], ENTITY_PALETTE[left_entity]["color"], (240, 160, 304, 224))
    return img

def main():
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
    revision = "a7da5b986cb59b408707209984f360a5f4ad7e47"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_id} on {device}...")
    processor = AutoProcessor.from_pretrained(model_id, revision=revision, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",
        local_files_only=True
    ).to(device)
    model.eval()
    print("Model loaded.\n")

    audit_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": model_id,
        "battery_tests": {}
    }

    # -------------------------------------------------------------
    # BATTERY 1: Entity Identity Recognition (Color and Shape)
    # -------------------------------------------------------------
    print("=== BATTERY 1: Entity Identity Recognition ===")
    img1_path = Path("datasets/controlled/images/ctrl_forward_chain_0001.png")
    # In ctrl_forward_chain_0001.png: cyan_box on left (x~60), green_cylinder on right (x~250)
    img1 = Image.open(img1_path).convert("RGB")

    battery1_cases = [
        {"id": "b1_color_left", "prompt": "What color is the shape on the left? Answer with just the color name.", "expected": "cyan"},
        {"id": "b1_shape_left", "prompt": "What shape is on the left? Answer with just the shape name.", "expected": "box / square / rectangle"},
        {"id": "b1_color_right", "prompt": "What color is the shape on the right? Answer with just the color name.", "expected": "green"},
        {"id": "b1_shape_right", "prompt": "What shape is on the right? Answer with just the shape name.", "expected": "cylinder / pill / oval"},
    ]

    b1_results = []
    for c in battery1_cases:
        res = run_single_inference(model, processor, img1, c["prompt"], device)
        c_record = {**c, **res}
        b1_results.append(c_record)
        print(f"  [{c['id']}] Output: '{res['decoded_completion']}' | Top 1: {res['top5_first_tokens'][0]['token_str']} ({res['top5_first_tokens'][0]['prob']:.3f})")

    audit_results["battery_tests"]["entity_identity"] = b1_results

    # -------------------------------------------------------------
    # BATTERY 2: Direct Pairwise Spatial Relations (Left-of vs Right-of)
    # -------------------------------------------------------------
    print("\n=== BATTERY 2: Direct Pairwise Spatial Relations ===")
    # Using ctrl_forward_chain_0001.png (cyan box is on left, green cylinder is on right)
    battery2_cases = [
        {
            "id": "b2_left_direct_true",
            "prompt": "Is the cyan box to the left of the green cylinder? Answer directly with: Yes or No.",
            "ground_truth": "Yes"
        },
        {
            "id": "b2_left_direct_false",
            "prompt": "Is the green cylinder to the left of the cyan box? Answer directly with: Yes or No.",
            "ground_truth": "No"
        },
        {
            "id": "b2_right_direct_true",
            "prompt": "Is the green cylinder to the right of the cyan box? Answer directly with: Yes or No.",
            "ground_truth": "Yes"
        },
        {
            "id": "b2_right_direct_false",
            "prompt": "Is the cyan box to the right of the green cylinder? Answer directly with: Yes or No.",
            "ground_truth": "No"
        },
    ]

    b2_results = []
    for c in battery2_cases:
        res = run_single_inference(model, processor, img1, c["prompt"], device)
        c_record = {**c, **res}
        b2_results.append(c_record)
        print(f"  [{c['id']}] Expected: {c['ground_truth']} | Output: '{res['decoded_completion']}' | Top 1: {res['top5_first_tokens'][0]['token_str']} ({res['top5_first_tokens'][0]['prob']:.3f})")

    audit_results["battery_tests"]["pairwise_spatial"] = b2_results

    # -------------------------------------------------------------
    # BATTERY 3: Counterfactual Image Swap
    # -------------------------------------------------------------
    print("\n=== BATTERY 3: Counterfactual Image Swap ===")
    # Query: "Is the cyan box to the left of the green cylinder? Answer directly with: Yes or No."
    swap_prompt = "Is the cyan box to the left of the green cylinder? Answer directly with: Yes or No."
    cf_img = create_counterfactual_image("cyan_box", "green_cylinder") # cyan box is placed on RIGHT, green cylinder on LEFT!

    res_orig = run_single_inference(model, processor, img1, swap_prompt, device)
    res_swap = run_single_inference(model, processor, cf_img, swap_prompt, device)

    b3_results = {
        "query": swap_prompt,
        "original_image": {
            "layout": "cyan_box(left) < green_cylinder(right)",
            "ground_truth": "Yes",
            **res_orig
        },
        "swapped_image": {
            "layout": "green_cylinder(left) < cyan_box(right)",
            "ground_truth": "No",
            **res_swap
        },
        "sensitivity_confirmed": (res_orig["decoded_completion"] != res_swap["decoded_completion"])
    }
    audit_results["battery_tests"]["image_swap_control"] = b3_results
    print(f"  Original Image (Cyan left, Green right) -> Expected: Yes | Model: '{res_orig['decoded_completion']}'")
    print(f"  Swapped Image  (Green left, Cyan right) -> Expected: No  | Model: '{res_swap['decoded_completion']}'")
    print(f"  Sensitivity Confirmed: {b3_results['sensitivity_confirmed']}")

    # -------------------------------------------------------------
    # BATTERY 4: Triplet Unimodal Visual Confound Check
    # -------------------------------------------------------------
    print("\n=== BATTERY 4: Triplet Unimodal Visual Confound Check ===")
    # Test on ctrl_unimodal_visual_0109.png:
    # layout: magenta_prism (left) < yellow_cone (mid) < orange_torus (right)
    img_triplet = Image.open("datasets/controlled/images/ctrl_unimodal_visual_0109.png").convert("RGB")

    b4_cases = [
        {
            "id": "b4_with_preamble",
            "prompt": "No additional statement is provided.\n\nQuestion: Based on the image, is the magenta prism to the left of the orange torus?\nCarefully evaluate whether the question is entailed (Yes), contradicted (No), or underdetermined from the evidence (Undetermined).\nAnswer directly with: Yes, No, or Undetermined.",
            "ground_truth": "Yes"
        },
        {
            "id": "b4_without_preamble",
            "prompt": "Question: Based on the image, is the magenta prism to the left of the orange torus?\nCarefully evaluate whether the question is entailed (Yes), contradicted (No), or underdetermined from the evidence (Undetermined).\nAnswer directly with: Yes, No, or Undetermined.",
            "ground_truth": "Yes"
        },
        {
            "id": "b4_adjacent_ab",
            "prompt": "Based on the image, is the magenta prism to the left of the yellow cone? Answer directly with: Yes or No.",
            "ground_truth": "Yes"
        },
        {
            "id": "b4_adjacent_bc",
            "prompt": "Based on the image, is the yellow cone to the left of the orange torus? Answer directly with: Yes or No.",
            "ground_truth": "Yes"
        },
        {
            "id": "b4_distant_ac_binary",
            "prompt": "Based on the image, is the magenta prism to the left of the orange torus? Answer directly with: Yes or No.",
            "ground_truth": "Yes"
        }
    ]

    b4_results = []
    for c in b4_cases:
        res = run_single_inference(model, processor, img_triplet, c["prompt"], device)
        c_record = {**c, **res}
        b4_results.append(c_record)
        print(f"  [{c['id']}] Expected: {c['ground_truth']} | Output: '{res['decoded_completion']}' | Top 1: {res['top5_first_tokens'][0]['token_str']} ({res['top5_first_tokens'][0]['prob']:.3f})")

    audit_results["battery_tests"]["triplet_unimodal_visual"] = b4_results

    # -------------------------------------------------------------
    # BATTERY 5: Choice-Order Sensitivity & Affirmation Bias
    # -------------------------------------------------------------
    print("\n=== BATTERY 5: Choice-Order Sensitivity & Affirmation Bias ===")
    # Test on an underdetermined fork: ctrl_underdetermined_fork_0073
    # Ground truth is Undetermined
    img_fork = Image.open("datasets/controlled/images/ctrl_underdetermined_fork_0073.png").convert("RGB")
    fork_premise = "The cyan box is to the left of the green cylinder."
    fork_query = "Based on the image and statement, is the orange torus to the left of the cyan box?"

    b5_cases = [
        {
            "id": "b5_order_yes_no_und",
            "order": "Yes, No, or Undetermined",
            "prompt": f"{fork_premise}\n\nQuestion: {fork_query}\nCarefully evaluate whether the question is entailed (Yes), contradicted (No), or underdetermined from the evidence (Undetermined).\nAnswer directly with: Yes, No, or Undetermined."
        },
        {
            "id": "b5_order_und_no_yes",
            "order": "Undetermined, No, or Yes",
            "prompt": f"{fork_premise}\n\nQuestion: {fork_query}\nCarefully evaluate whether the question is underdetermined (Undetermined), contradicted (No), or entailed (Yes).\nAnswer directly with: Undetermined, No, or Yes."
        },
        {
            "id": "b5_order_no_yes_und",
            "order": "No, Yes, or Undetermined",
            "prompt": f"{fork_premise}\n\nQuestion: {fork_query}\nCarefully evaluate whether the question is contradicted (No), entailed (Yes), or underdetermined from the evidence (Undetermined).\nAnswer directly with: No, Yes, or Undetermined."
        }
    ]

    b5_results = []
    for c in b5_cases:
        res = run_single_inference(model, processor, img_fork, c["prompt"], device)
        c_record = {**c, **res}
        b5_results.append(c_record)
        top1 = res['top5_first_tokens'][0]
        c_stats = res['candidate_stats']
        yes_p = c_stats['yes_space']['prob'] + c_stats['yes_nospace']['prob']
        no_p = c_stats['no_space']['prob'] + c_stats['no_nospace']['prob']
        und_p = c_stats['und_space']['prob'] + c_stats['und_nospace']['prob']
        print(f"  [{c['id']}] Output: '{res['decoded_completion']}' | Top 1: '{top1['token_str']}' ({top1['prob']:.3f}) | P(Yes)={yes_p:.4f}, P(No)={no_p:.4f}, P(Und)={und_p:.4f}")

    audit_results["battery_tests"]["choice_order_sensitivity"] = b5_results

    # Save structured audit results
    out_path = Path("artifacts/evaluations/visual_grounding_audit_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit_results, indent=2), encoding="utf-8")
    print(f"\nVisual grounding and bias audit complete. Results saved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
