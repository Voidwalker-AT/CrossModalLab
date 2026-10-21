"""Pinned SmolVLM2 loader with the verified masked_scatter inputs-merger patch.

The upstream transformers SmolVLM inputs_merger contains an unvalidated
modulo check against vision_config.patch_size (14). SmolVLM2 visual token
counts (e.g. 1377) are not divisible by 14, so the stock path fails.
Phase 0D.1 verified that restoring the canonical Idefics3 masked_scatter
implementation is required and scientifically safe for this model.

This module never downloads weights and never substitutes model revisions.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from src.config.experiment import get_experiment_config, require_gpu_capabilities
from src.config.paths import get_smolvlm2_snapshot_dir


def apply_smolvlm_inputs_merger_patch() -> str:
    """Install the verified canonical inputs_merger patch. Returns patch id."""
    import torch
    from transformers.models.smolvlm import modeling_smolvlm

    def safe_inputs_merger(
        self,
        input_ids: "torch.LongTensor | None",
        inputs_embeds: "torch.Tensor",
        image_hidden_states: "torch.Tensor",
    ):
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.full(
                    (),
                    self.config.image_token_id,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        special_image_mask = special_image_mask.unsqueeze(-1).to(inputs_embeds.device)
        image_hidden_states = image_hidden_states.to(inputs_embeds.device, inputs_embeds.dtype)
        return inputs_embeds.masked_scatter(special_image_mask, image_hidden_states)

    modeling_smolvlm.SmolVLMModel.inputs_merger = safe_inputs_merger
    return "smolvlm_masked_scatter_canonical"


def load_pinned_smolvlm2(*, require_gpu: bool = True) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load the pinned processor/model with NF4 quantization.

    Returns (processor, model, load_metadata). Raises CapabilityError or
    FileNotFoundError when hardware or assets are insufficient.
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    cfg = get_experiment_config()
    if require_gpu:
        require_gpu_capabilities()
    elif not torch.cuda.is_available():
        raise RuntimeError(
            "Pinned NF4 experiment requires CUDA. CPU-only symbolic workflows use tests/check_portability only."
        )

    snapshot_dir = get_smolvlm2_snapshot_dir()
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(
            f"Pinned model snapshot not found: {snapshot_dir}. "
            "See docs/ASSET_MANIFEST.md. Do not silently substitute another revision."
        )

    patch_id = apply_smolvlm_inputs_merger_patch()
    device = "cuda:0"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.quantization.load_in_4bit,
        bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=cfg.quantization.bnb_4bit_use_double_quant,
    )

    processor = AutoProcessor.from_pretrained(str(snapshot_dir), local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(snapshot_dir),
        quantization_config=bnb_config,
        device_map=device,
        _attn_implementation=cfg.generation["attn_implementation"],
        local_files_only=True,
    )
    model.eval()

    meta = {
        "model_id": cfg.model.model_id,
        "revision": cfg.model.revision,
        "snapshot_dir": str(snapshot_dir),
        "quantization": {
            "load_in_4bit": cfg.quantization.load_in_4bit,
            "bnb_4bit_quant_type": cfg.quantization.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": cfg.quantization.bnb_4bit_compute_dtype,
            "bnb_4bit_use_double_quant": cfg.quantization.bnb_4bit_use_double_quant,
        },
        "device": device,
        "inputs_merger_patch": patch_id,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    return processor, model, meta
