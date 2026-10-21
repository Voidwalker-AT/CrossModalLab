"""CrossModalLab probe package (CPU linear probes; non-causal)."""

from src.probes.confounding import CONDITION_TO_LABEL, analyze_confounding
from src.probes.linear_probe import (
    LinearProbe,
    ProbeDataset,
    evaluate_probe,
    fit_standardized_probe,
)

__all__ = [
    "CONDITION_TO_LABEL",
    "analyze_confounding",
    "LinearProbe",
    "ProbeDataset",
    "evaluate_probe",
    "fit_standardized_probe",
]
