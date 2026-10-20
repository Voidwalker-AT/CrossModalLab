"""CrossModalLab evaluation module."""
from src.evaluation.candidate_scorer import (
    CandidateScore,
    CandidateScorerResult,
    compute_sequence_log_likelihood_from_logits,
    score_candidate_answers
)

__all__ = [
    "CandidateScore",
    "CandidateScorerResult",
    "compute_sequence_log_likelihood_from_logits",
    "score_candidate_answers"
]
