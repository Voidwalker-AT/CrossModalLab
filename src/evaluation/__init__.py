"""CrossModalLab evaluation package.

CPU-safe modules (metrics, manifest, harness helpers) import without torch.
Torch-dependent candidate scoring is available via
``src.evaluation.candidate_scorer`` (imports torch at module load).
"""

from src.evaluation.behavioral_metrics import (
    build_prompt,
    compute_behavioral_metrics,
    parse_model_answer,
)

__all__ = [
    "build_prompt",
    "compute_behavioral_metrics",
    "parse_model_answer",
]
