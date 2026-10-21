"""Minimal multi-model adapter interface (Phase 2C-A).

Does not download or load Qwen. Only defines the surface existing scientific
code needs so SmolVLM2 and future Qwen3-VL can share pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Sequence


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str
    architecture: str  # e.g. smolvlm | qwen3_vl
    image_token_id: int
    text_num_layers: int
    hidden_size: int
    vision_num_layers: Optional[int]
    requires_trust_remote_code: bool
    preferred_attn_implementation: str  # eager | sdpa | flash_attention_2
    quantization_supported: bool
    notes: str = ""


SMOLVLM2_2_2B = ModelSpec(
    model_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    revision="482adb537c021c86670beed01cd58990d01e72e4",
    architecture="smolvlm",
    image_token_id=-1,  # resolved at runtime from model.config
    text_num_layers=24,
    hidden_size=2048,
    vision_num_layers=None,
    requires_trust_remote_code=False,
    preferred_attn_implementation="eager",
    quantization_supported=True,
    notes="Primary CrossModalLab model; inputs_merger masked_scatter patch required.",
)

# Specs from official HF config.json (no weights downloaded in this phase).
QWEN3_VL_2B = ModelSpec(
    model_id="Qwen/Qwen3-VL-2B-Instruct",
    revision="UNPINNED_FEASIBILITY_ONLY",
    architecture="qwen3_vl",
    image_token_id=151655,
    text_num_layers=28,
    hidden_size=2048,
    vision_num_layers=24,
    requires_trust_remote_code=False,
    preferred_attn_implementation="sdpa",
    quantization_supported=True,
    notes=(
        "HF card: Qwen3VLForConditionalGeneration; transformers>=4.57; "
        "vision patch 16 merge 2; DeepStack indexes 5/11/17; "
        "BF16 2B weights ~4GB — NF4 required on 4GB GPU."
    ),
)


class VLMAdapter(Protocol):
    """Minimal interface for CrossModalLab evaluation/instrumentation."""

    spec: ModelSpec

    def load(self) -> Any: ...

    def encode_chat(self, messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]: ...

    def image_token_count(self, inputs: Dict[str, Any]) -> int: ...

    def layer_module(self, model: Any, layer_index: int) -> Any: ...

    def residual_dim(self, model: Any) -> int: ...


def describe_model_specs() -> List[Dict[str, Any]]:
    from dataclasses import asdict

    return [asdict(SMOLVLM2_2_2B), asdict(QWEN3_VL_2B)]


def behavioral_task_battery() -> List[Dict[str, Any]]:
    """Shared logical tasks; model-specific templates applied later."""
    return [
        {"task_id": "object_identity", "requires_image": True, "requires_text": False},
        {"task_id": "left_right_relation", "requires_image": True, "requires_text": False},
        {"task_id": "image_swap", "requires_image": True, "requires_text": True},
        {"task_id": "text_premise_reversal", "requires_image": True, "requires_text": True},
        {"task_id": "entailed_chain", "requires_image": True, "requires_text": True},
        {"task_id": "contradicted_chain", "requires_image": True, "requires_text": True},
        {"task_id": "underdetermined_fork", "requires_image": True, "requires_text": True},
        {"task_id": "missing_visual_evidence", "requires_image": False, "requires_text": True},
        {"task_id": "missing_textual_evidence", "requires_image": True, "requires_text": False},
    ]
