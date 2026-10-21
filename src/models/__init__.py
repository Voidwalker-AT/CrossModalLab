"""Shared model-loading and compatibility adapters for CrossModalLab GPU experiments."""

from src.models.model_loader import (
    apply_smolvlm_inputs_merger_patch,
    load_pinned_smolvlm2,
)

__all__ = [
    "apply_smolvlm_inputs_merger_patch",
    "load_pinned_smolvlm2",
]
