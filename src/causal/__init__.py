"""CrossModalLab causal design package (design + CPU validators only in Phase 2A)."""

from src.causal.interventions import (
    InterventionSpec,
    MarginOutcomes,
    apply_residual_replacement,
    config_hash,
    margin_change,
    validate_intervention_spec,
)

__all__ = [
    "InterventionSpec",
    "MarginOutcomes",
    "apply_residual_replacement",
    "config_hash",
    "margin_change",
    "validate_intervention_spec",
]
