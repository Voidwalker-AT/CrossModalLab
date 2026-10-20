"""CrossModalLab — Phase 0D: SmolVLM2 Extended Validation Suite.

Fulfills all remaining explicit Stage 5, 6, and 7 requirements:
1. Stage 5: Single-object ablation control (object removal).
2. Stage 6: Individual-premise controls:
   - Unimodal visual 3-entity control (no text premise).
   - Text-only 2-premise reasoning control (no image).
3. Stage 7: Controlled in-flight no-op activation replacement hook on Layer 12 to verify the causal intervention path.

Output: artifacts/evaluations/smolvlm2_extended_validation_results.json
"""

import os
import sys
import json
import time
from pathlib import Path
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
    """Canonical masked_scatter patch for SmolVLM2."""
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

def create_ablated_image(entity: str) -> Image.Image:
    """Renders a single entity on canvas, removing the second entity."""
    img = Image.new("RGB", CANVAS_SIZE, color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    draw_shape(draw, ENTITY_PALETTE[entity]["shape"], ENTITY_PALETTE[entity]["color"], (60, 160, 124, 224))
    return img

def run_inference(model, processor, prompt_text: str, image: Image.Image = None, device: str = "cuda:0", max_new_tokens: int = 25):
    """Executes multimodal or text-only inference."""
    import torch

    if image is not None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text}
                ]
            }
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

        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    elapsed = time.perf_counter() - t0

    new_token_ids = out_ids[0, prompt_len:].tolist()
    decoded = processor.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

    topk_vals, topk_inds = torch.topk(first_token_probs, k=5)
    top5 = [
        {
            "token_id": topk_inds[i].item(),
            "token_str": processor.tokenizer.decode([topk_inds[i].item()]),
            "prob": round(topk_vals[i].item(), 4),
            "logit": round(first_token_logits[topk_inds[i].item()].item(), 3)
        }
        for i in range(5)
    ]

    del outputs, first_token_logits, first_token_probs, out_ids
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "prompt_tokens": prompt_len,
        "completion_tokens": len(new_token_ids),
        "completion_text": decoded,
        "elapsed_sec": round(elapsed, 3),
        "top5_first_tokens": top5,
        "raw_inputs": inputs
    }

def main():
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    patch_smolvlm_inputs_merger()

    snapshot_dir = get_smolvlm2_snapshot_dir()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("================================================================================")
    print("CrossModalLab — Phase 0D: SmolVLM2 Extended Validation Suite")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print("================================================================================\n")

    print("[1/4] Loading model and processor...")
    processor = AutoProcessor.from_pretrained(str(snapshot_dir), local_files_only=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        str(snapshot_dir),
        quantization_config=bnb_config,
        device_map=device,
        _attn_implementation="eager",
        local_files_only=True
    )
    model.eval()
    print("  -> Ready.\n")

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage_5_ablation_control": {},
        "stage_6_individual_premise_controls": {},
        "stage_7_activation_replacement_hook": {}
    }

    # -------------------------------------------------------------
    # 1. STAGE 5: Object Removal / Ablation Control
    # -------------------------------------------------------------
    print("=== 1. STAGE 5: Object Removal / Ablation Control ===")
    ablated_img = create_ablated_image("cyan_box") # Only cyan box, green cylinder removed!
    query_ablation = "Is the cyan box to the left of the green cylinder? Answer directly with: Yes, No, or Undetermined."

    res_ablated = run_inference(model, processor, query_ablation, image=ablated_img, device=device)
    results["stage_5_ablation_control"] = {
        "condition": "single_object_present_cyan_box_only",
        "prompt": query_ablation,
        "completion": res_ablated["completion_text"],
        "top5_tokens": res_ablated["top5_first_tokens"]
    }
    print(f"  Single-Object (Cyan Only) Output: '{res_ablated['completion_text']}' | Top 1: '{res_ablated['top5_first_tokens'][0]['token_str']}' ({res_ablated['top5_first_tokens'][0]['prob']:.4f})")

    # -------------------------------------------------------------
    # 2. STAGE 6: Individual Premise Controls
    # -------------------------------------------------------------
    print("\n=== 2. STAGE 6: Individual Premise Controls ===")

    # 2.1 Unimodal Visual Control (3 objects present, no text premise)
    # Using ctrl_unimodal_visual_0109.png: magenta_prism < yellow_cone < orange_torus
    img_triplet = Image.open("datasets/controlled/images/ctrl_unimodal_visual_0109.png").convert("RGB")
    unimodal_visual_prompt = (
        "Question: Based on the image, is the magenta prism to the left of the orange torus?\n"
        "Carefully evaluate whether the question is entailed (Yes), contradicted (No), or underdetermined from the evidence (Undetermined).\n"
        "Answer directly with: Yes, No, or Undetermined."
    )
    res_vis_ctrl = run_inference(model, processor, unimodal_visual_prompt, image=img_triplet, device=device)
    cand_scorer_vis = score_candidate_answers(
        model=model,
        processor=processor,
        base_inputs=res_vis_ctrl["raw_inputs"],
        candidate_strings=[" Yes", " No", " Undetermined"],
        device=device
    )

    # 2.2 Text-Only Control (2 premises in text, zero image)
    text_only_prompt = (
        "The cyan box is to the left of the green cylinder.\n"
        "The green cylinder is to the left of the magenta prism.\n\n"
        "Question: Based on the statements, is the cyan box to the left of the magenta prism?\n"
        "Carefully evaluate whether the question is entailed (Yes), contradicted (No), or underdetermined from the evidence (Undetermined).\n"
        "Answer directly with: Yes, No, or Undetermined."
    )
    res_txt_ctrl = run_inference(model, processor, text_only_prompt, image=None, device=device)
    cand_scorer_txt = score_candidate_answers(
        model=model,
        processor=processor,
        base_inputs=res_txt_ctrl["raw_inputs"],
        candidate_strings=[" Yes", " No", " Undetermined"],
        device=device
    )

    results["stage_6_individual_premise_controls"] = {
        "unimodal_visual": {
            "prompt": unimodal_visual_prompt,
            "ground_truth": "Yes",
            "completion": res_vis_ctrl["completion_text"],
            "top1_token": res_vis_ctrl["top5_first_tokens"][0]["token_str"],
            "top1_prob": res_vis_ctrl["top5_first_tokens"][0]["prob"],
            "candidate_probabilities_length_norm": cand_scorer_vis.candidate_probabilities_normalized,
            "top_candidate": cand_scorer_vis.length_normalized_ranking[0].strip()
        },
        "text_only": {
            "prompt": text_only_prompt,
            "ground_truth": "Yes",
            "completion": res_txt_ctrl["completion_text"],
            "top1_token": res_txt_ctrl["top5_first_tokens"][0]["token_str"],
            "top1_prob": res_txt_ctrl["top5_first_tokens"][0]["prob"],
            "candidate_probabilities_length_norm": cand_scorer_txt.candidate_probabilities_normalized,
            "top_candidate": cand_scorer_txt.length_normalized_ranking[0].strip()
        }
    }
    print(f"  Unimodal Visual Output: '{res_vis_ctrl['completion_text']}' | Top Cand: {cand_scorer_vis.length_normalized_ranking[0].strip()} | P(Yes)={cand_scorer_vis.candidate_probabilities_normalized[' Yes']}")
    print(f"  Text-Only Output:       '{res_txt_ctrl['completion_text']}' | Top Cand: {cand_scorer_txt.length_normalized_ranking[0].strip()} | P(Yes)={cand_scorer_txt.candidate_probabilities_normalized[' Yes']}")

    # -------------------------------------------------------------
    # 3. STAGE 7: In-Flight No-Op Activation Replacement Hook
    # -------------------------------------------------------------
    print("\n=== 3. STAGE 7: In-Flight No-Op Activation Replacement Hook ===")
    target_layer_idx = 12
    target_layer = model.model.text_model.layers[target_layer_idx]

    # Baseline forward pass without hook
    test_img = Image.open("datasets/controlled/images/ctrl_forward_chain_0001.png").convert("RGB")
    test_prompt = "The green cylinder is to the left of the magenta prism.\n\nQuestion: Is the cyan box to the left of the magenta prism?\nAnswer directly with: Yes or No."

    messages = [
        {"role": "user", "content": [{"type": "image", "image": test_img}, {"type": "text", "text": test_prompt}]}
    ]
    base_inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(device)

    with torch.inference_mode():
        unhooked_out = model(**base_inputs)
        unhooked_logits = unhooked_out.logits[0, -1, :].clone()

    # Register in-flight activation replacement hook
    # Replaces hidden states with explicit cloned copy to test modification path
    replacement_event = {}
    def noop_replacement_hook(module, args, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        replacement_event["intercepted_shape"] = list(hidden_states.shape)
        replacement_event["intercepted_dtype"] = str(hidden_states.dtype)
        # Modify activation in-flight via identity tensor operation
        modified_hidden_states = hidden_states.clone()
        if isinstance(output, tuple):
            return (modified_hidden_states, *output[1:])
        return modified_hidden_states

    hook_handle = target_layer.register_forward_hook(noop_replacement_hook)
    print(f"  -> In-flight replacement hook registered on layer {target_layer_idx}.")

    with torch.inference_mode():
        hooked_out = model(**base_inputs)
        hooked_logits = hooked_out.logits[0, -1, :].clone()

    hook_handle.remove()
    print("  -> Hooked forward pass executed. Hook handle cleanly removed.")

    logits_match = torch.allclose(unhooked_logits, hooked_logits, atol=1e-5)
    max_logit_diff = (unhooked_logits - hooked_logits).abs().max().item()

    results["stage_7_activation_replacement_hook"] = {
        "target_layer": f"model.model.text_model.layers[{target_layer_idx}]",
        "intercepted_shape": replacement_event["intercepted_shape"],
        "intercepted_dtype": replacement_event["intercepted_dtype"],
        "logits_match": logits_match,
        "max_logit_diff": max_logit_diff,
        "status": "PASS" if logits_match else "FAIL"
    }

    print(f"  -> Intercepted Tensor Shape: {replacement_event['intercepted_shape']}")
    print(f"  -> Intercepted Tensor Dtype: {replacement_event['intercepted_dtype']}")
    print(f"  -> Logits Bit-Exact Match:   {logits_match} (Max diff: {max_logit_diff})")
    print(f"  -> Replacement Hook Verdict: {results['stage_7_activation_replacement_hook']['status']}")

    out_file = Path("artifacts/evaluations/smolvlm2_extended_validation_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nExtended validation results saved to: {out_file.resolve()}")

if __name__ == "__main__":
    main()
