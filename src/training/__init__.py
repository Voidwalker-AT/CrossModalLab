"""CrossModalLab training design package.

Schema utilities are CPU-only (no torch). Losses require torch and are imported lazily.
"""

from src.training.schema import full_audit, validate_record_schema

__all__ = [
    "combined_loss",
    "counterfactual_consistency_loss",
    "cross_entropy_loss",
    "evidence_aware_loss",
    "full_audit",
    "validate_record_schema",
]


def __getattr__(name: str):
    if name in {
        "combined_loss",
        "counterfactual_consistency_loss",
        "cross_entropy_loss",
        "evidence_aware_loss",
    }:
        import src.training.losses as _losses

        return getattr(_losses, name)
    raise AttributeError(name)
