"""
Symbolic Reasoning Engine for CrossModalLab.

Provides formal directional relations, graph reachability, cycle detection,
linear extension enumeration, 3-way epistemic classification, deterministic
dataset generation, and independent artifact verification.
"""

from src.symbolic.relations import (
    DirectionalRelation,
    Modality,
    EpistemicStatus,
    InconsistentPremisesError,
    InvalidRelationError,
    Premise,
    Query,
)
from src.symbolic.graph import RelationGraph
from src.symbolic.solver import SymbolicSolver, SolverResult
from src.symbolic.generator import BenchmarkGenerator
from src.symbolic.verifier import BenchmarkVerifier, VerificationReport

__all__ = [
    "DirectionalRelation",
    "Modality",
    "EpistemicStatus",
    "InconsistentPremisesError",
    "InvalidRelationError",
    "Premise",
    "Query",
    "RelationGraph",
    "SymbolicSolver",
    "SolverResult",
    "BenchmarkGenerator",
    "BenchmarkVerifier",
    "VerificationReport",
]
