"""
Deductive solver and 3-way epistemic classifier with dual-algorithm verification.

Evaluates queries against premises using two independent methods:
1. Primary: Directed Graph Reachability & Partial Order Logic
2. Verification Oracle: Exhaustive Linear Extension Enumeration
"""

from dataclasses import dataclass
from typing import Any, Optional

from src.symbolic.graph import RelationGraph
from src.symbolic.relations import (
    EpistemicStatus,
    InconsistentPremisesError,
    Premise,
    Query,
)


@dataclass(frozen=True)
class SolverResult:
    """Outcome of symbolic deduction and epistemic classification."""
    epistemic_status: EpistemicStatus
    linear_extensions: list[list[str]]
    num_linear_extensions: int
    algorithm_agreement: bool
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert result to JSON-serializable dictionary."""
        return {
            "epistemic_status": self.epistemic_status.value,
            "num_linear_extensions": self.num_linear_extensions,
            "linear_extensions": self.linear_extensions,
            "algorithm_agreement": self.algorithm_agreement,
            "explanation": self.explanation,
        }


class SymbolicSolver:
    """
    Symbolic solver evaluating directional relational queries over premises.

    Enforces dual-algorithm cross-validation on every inference step.
    """

    @classmethod
    def solve(
        cls,
        premises: list[Premise],
        query: Query,
        verify_with_oracle: bool = True,
    ) -> SolverResult:
        """
        Evaluate the epistemic status of a query given a premise set.

        Args:
            premises: Sequence of asserted directional premises.
            query: The target relational query R(subject, object).
            verify_with_oracle: If True, executes Algorithm 2 (Linear Extension Oracle)
                                and asserts exact agreement with Algorithm 1.

        Returns:
            SolverResult with verified epistemic status and linear extensions.

        Raises:
            AssertionError: If Algorithm 1 and Algorithm 2 yield conflicting classifications.
        """
        # Step 1: Attempt to construct graph including query entities
        extra_entities = [query.subject, query.object]
        graph = RelationGraph(entities=extra_entities)

        is_consistent = True
        contradiction_reason = ""

        try:
            for p in premises:
                # Add edge, detecting immediate 2-cycles or transitive cycles
                graph.add_relation(p.subject, p.object, check_cycle=True)
        except InconsistentPremisesError as exc:
            is_consistent = False
            contradiction_reason = str(exc)

        # Check for any remaining cycles if edge additions didn't raise
        if is_consistent and graph.has_cycle():
            is_consistent = False
            cycles = graph.find_cycles()
            contradiction_reason = f"Cycle detected: {' -> '.join(cycles[0]) if cycles else 'unspecified cycle'}"

        # Algorithm 1: Reachability / Partial Order Logic
        if not is_consistent:
            status_algo1 = EpistemicStatus.INCONSISTENT
            explanation_algo1 = f"Inconsistent premises: {contradiction_reason}"
        else:
            subj_reaches_obj = graph.is_reachable(query.subject, query.object)
            obj_reaches_subj = graph.is_reachable(query.object, query.subject)

            if subj_reaches_obj:
                status_algo1 = EpistemicStatus.ENTAILED
                explanation_algo1 = (
                    f"Entailed: Transitive path exists from {query.subject!r} to {query.object!r} "
                    f"in the directed relation graph."
                )
            elif obj_reaches_subj:
                status_algo1 = EpistemicStatus.CONTRADICTED
                explanation_algo1 = (
                    f"Contradicted: Reverse transitive path exists from {query.object!r} to {query.subject!r}; "
                    f"therefore {query.object!r} < {query.subject!r}, contradicting {query.subject!r} < {query.object!r}."
                )
            else:
                status_algo1 = EpistemicStatus.UNDERDETERMINED
                explanation_algo1 = (
                    f"Underdetermined: Neither {query.subject!r} nor {query.object!r} reaches the other; "
                    f"their relative order is unconstrained by the premises."
                )

        # Algorithm 2: Linear Extension Verification Oracle
        linear_extensions: list[list[str]] = []
        status_algo2 = EpistemicStatus.INCONSISTENT
        explanation_algo2 = ""

        if is_consistent:
            linear_extensions = graph.get_linear_extensions()
            if not linear_extensions:
                status_algo2 = EpistemicStatus.INCONSISTENT
                explanation_algo2 = "Oracle found 0 valid linear extensions (contradiction)."
            else:
                # Evaluate query R(subject, object) across all linear extensions
                n_total = len(linear_extensions)
                n_holds = 0

                for ext in linear_extensions:
                    idx_subj = ext.index(query.subject)
                    idx_obj = ext.index(query.object)
                    if idx_subj < idx_obj:
                        n_holds += 1

                if n_holds == n_total:
                    status_algo2 = EpistemicStatus.ENTAILED
                    explanation_algo2 = (
                        f"Oracle confirmed: R({query.subject}, {query.object}) holds in "
                        f"100% of linear extensions ({n_holds}/{n_total})."
                    )
                elif n_holds == 0:
                    status_algo2 = EpistemicStatus.CONTRADICTED
                    explanation_algo2 = (
                        f"Oracle confirmed: R({query.subject}, {query.object}) holds in "
                        f"0% of linear extensions (reverse holds in {n_total}/{n_total})."
                    )
                else:
                    status_algo2 = EpistemicStatus.UNDERDETERMINED
                    explanation_algo2 = (
                        f"Oracle confirmed: R({query.subject}, {query.object}) holds in "
                        f"{n_holds}/{n_total} linear extensions, and reverse holds in "
                        f"{n_total - n_holds}/{n_total} extensions."
                    )
        else:
            status_algo2 = EpistemicStatus.INCONSISTENT
            explanation_algo2 = f"Oracle skipped due to detected inconsistency: {contradiction_reason}"

        # Cross-validation
        agreement = (status_algo1 == status_algo2)
        if verify_with_oracle and not agreement:
            raise AssertionError(
                f"DUAL-ALGORITHM CONCORDANCE FAILURE!\n"
                f"Algorithm 1 (Reachability): {status_algo1.value} ({explanation_algo1})\n"
                f"Algorithm 2 (Oracle): {status_algo2.value} ({explanation_algo2})\n"
                f"Premises: {premises}\n"
                f"Query: {query}"
            )

        return SolverResult(
            epistemic_status=status_algo1,
            linear_extensions=linear_extensions,
            num_linear_extensions=len(linear_extensions),
            algorithm_agreement=agreement,
            explanation=explanation_algo1 if is_consistent else explanation_algo2,
        )

    @classmethod
    def solve_from_raw(
        cls,
        premises_raw: list[dict[str, Any]],
        query_raw: dict[str, Any],
        verify_with_oracle: bool = True,
    ) -> SolverResult:
        """Parse raw dictionary inputs and evaluate query epistemic status."""
        premises = [
            Premise(
                subject=p["subject"],
                object=p["object"],
                relation=p.get("relation", "left_of"),
                modality=p.get("modality", "textual"),
            )
            for p in premises_raw
        ]
        query = Query(
            subject=query_raw["subject"],
            object=query_raw["object"],
            relation=query_raw.get("relation", "left_of"),
        )
        return cls.solve(premises, query, verify_with_oracle=verify_with_oracle)
