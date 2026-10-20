"""
Unit tests for SymbolicSolver and dual-algorithm 3-way epistemic classification.
"""

import itertools
import unittest

from src.symbolic.relations import (
    DirectionalRelation,
    EpistemicStatus,
    Modality,
    Premise,
    Query,
)
from src.symbolic.solver import SymbolicSolver


class TestSymbolicSolver(unittest.TestCase):
    """Test suite for logical deduction, epistemic states, and dual-algorithm cross-validation."""

    def test_forward_chain_entailed(self) -> None:
        """R(A, B) and R(B, C) => R(A, C) is ENTAILED."""
        premises = [
            Premise("blue_cube", "green_cylinder", DirectionalRelation.LEFT_OF, Modality.VISUAL),
            Premise("green_cylinder", "red_sphere", DirectionalRelation.LEFT_OF, Modality.TEXTUAL),
        ]
        query = Query("blue_cube", "red_sphere", DirectionalRelation.LEFT_OF)

        res = SymbolicSolver.solve(premises, query, verify_with_oracle=True)
        self.assertEqual(res.epistemic_status, EpistemicStatus.ENTAILED)
        self.assertTrue(res.algorithm_agreement)
        self.assertEqual(res.num_linear_extensions, 1)
        self.assertEqual(res.linear_extensions[0], ["blue_cube", "green_cylinder", "red_sphere"])

    def test_reverse_chain_contradicted(self) -> None:
        """R(B, A) and R(C, B) => R(A, C) is CONTRADICTED (since C < B < A)."""
        premises = [
            Premise("green_cylinder", "blue_cube", DirectionalRelation.LEFT_OF, Modality.VISUAL),
            Premise("red_sphere", "green_cylinder", DirectionalRelation.LEFT_OF, Modality.TEXTUAL),
        ]
        query = Query("blue_cube", "red_sphere", DirectionalRelation.LEFT_OF)

        res = SymbolicSolver.solve(premises, query, verify_with_oracle=True)
        self.assertEqual(res.epistemic_status, EpistemicStatus.CONTRADICTED)
        self.assertTrue(res.algorithm_agreement)
        self.assertEqual(res.num_linear_extensions, 1)
        self.assertEqual(res.linear_extensions[0], ["red_sphere", "green_cylinder", "blue_cube"])

    def test_underdetermined_fan_in_fork(self) -> None:
        """R(A, B) and R(C, B) => R(A, C) is UNDERDETERMINED."""
        premises = [
            Premise("blue_cube", "green_cylinder", DirectionalRelation.LEFT_OF, Modality.VISUAL),
            Premise("red_sphere", "green_cylinder", DirectionalRelation.LEFT_OF, Modality.TEXTUAL),
        ]
        query = Query("blue_cube", "red_sphere", DirectionalRelation.LEFT_OF)

        res = SymbolicSolver.solve(premises, query, verify_with_oracle=True)
        self.assertEqual(res.epistemic_status, EpistemicStatus.UNDERDETERMINED)
        self.assertTrue(res.algorithm_agreement)
        self.assertEqual(res.num_linear_extensions, 2)
        # blue_cube and red_sphere can be in either relative order
        ext_blue_first = ["blue_cube", "red_sphere", "green_cylinder"]
        ext_red_first = ["red_sphere", "blue_cube", "green_cylinder"]
        self.assertIn(ext_blue_first, res.linear_extensions)
        self.assertIn(ext_red_first, res.linear_extensions)

    def test_underdetermined_fan_out_fork(self) -> None:
        """R(B, A) and R(B, C) => R(A, C) is UNDERDETERMINED."""
        premises = [
            Premise("green_cylinder", "blue_cube", DirectionalRelation.LEFT_OF, Modality.VISUAL),
            Premise("green_cylinder", "red_sphere", DirectionalRelation.LEFT_OF, Modality.TEXTUAL),
        ]
        query = Query("blue_cube", "red_sphere", DirectionalRelation.LEFT_OF)

        res = SymbolicSolver.solve(premises, query, verify_with_oracle=True)
        self.assertEqual(res.epistemic_status, EpistemicStatus.UNDERDETERMINED)
        self.assertTrue(res.algorithm_agreement)
        self.assertEqual(res.num_linear_extensions, 2)

    def test_underdetermined_disconnected_components(self) -> None:
        """R(A, B) and R(C, D) => R(A, C) is UNDERDETERMINED."""
        premises = [
            Premise("A", "B", DirectionalRelation.LEFT_OF),
            Premise("C", "D", DirectionalRelation.LEFT_OF),
        ]
        query = Query("A", "C", DirectionalRelation.LEFT_OF)

        res = SymbolicSolver.solve(premises, query, verify_with_oracle=True)
        self.assertEqual(res.epistemic_status, EpistemicStatus.UNDERDETERMINED)
        self.assertTrue(res.algorithm_agreement)
        self.assertEqual(res.num_linear_extensions, 6)

    def test_inconsistent_premises_cycle(self) -> None:
        """Contradictory premises (cycle A -> B -> C -> A) must return INCONSISTENT."""
        premises = [
            Premise("A", "B"),
            Premise("B", "C"),
            Premise("C", "A"),
        ]
        query = Query("A", "B")

        res = SymbolicSolver.solve(premises, query, verify_with_oracle=True)
        self.assertEqual(res.epistemic_status, EpistemicStatus.INCONSISTENT)
        self.assertTrue(res.algorithm_agreement)
        self.assertEqual(res.num_linear_extensions, 0)
        self.assertEqual(res.linear_extensions, [])

    def test_direct_premise_evaluation(self) -> None:
        premises = [Premise("A", "B")]

        # Query R(A, B) should be ENTAILED
        q_entailed = Query("A", "B")
        res_ent = SymbolicSolver.solve(premises, q_entailed, verify_with_oracle=True)
        self.assertEqual(res_ent.epistemic_status, EpistemicStatus.ENTAILED)

        # Query R(B, A) should be CONTRADICTED
        q_contradicted = Query("B", "A")
        res_cont = SymbolicSolver.solve(premises, q_contradicted, verify_with_oracle=True)
        self.assertEqual(res_cont.epistemic_status, EpistemicStatus.CONTRADICTED)

    def test_solve_from_raw_dictionary(self) -> None:
        raw_premises = [
            {"subject": "blue_cube", "object": "green_cylinder", "modality": "visual", "relation": "left_of"},
            {"subject": "green_cylinder", "object": "red_sphere", "modality": "textual", "relation": "left_of"},
        ]
        raw_query = {"subject": "blue_cube", "object": "red_sphere", "relation": "left_of"}

        res = SymbolicSolver.solve_from_raw(raw_premises, raw_query, verify_with_oracle=True)
        self.assertEqual(res.epistemic_status, EpistemicStatus.ENTAILED)
        self.assertTrue(res.algorithm_agreement)

    def test_exhaustive_3_node_sweep(self) -> None:
        """
        Exhaustively test all 2^6 = 64 directed subgraphs over 3 entities (A, B, C).
        Verifies 100% concordance between Algorithm 1 (Reachability) and
        Algorithm 2 (Linear Extension Oracle) across all 64 * 6 = 384 queries.
        """
        entities = ["A", "B", "C"]
        all_possible_edges = [
            ("A", "B"), ("B", "A"),
            ("A", "C"), ("C", "A"),
            ("B", "C"), ("C", "B"),
        ]

        evaluated_combinations = 0
        for r in range(len(all_possible_edges) + 1):
            for edge_subset in itertools.combinations(all_possible_edges, r):
                premises = [Premise(u, v) for u, v in edge_subset]
                for u in entities:
                    for v in entities:
                        if u != v:
                            q = Query(u, v)
                            res = SymbolicSolver.solve(premises, q, verify_with_oracle=True)
                            self.assertTrue(res.algorithm_agreement)
                            evaluated_combinations += 1

        self.assertEqual(evaluated_combinations, 64 * 6)

    def test_exhaustive_4_node_linear_orders(self) -> None:
        """
        Exhaustively test all 24 linear orders of 4 entities (A, B, C, D).
        Asserts 100% agreement on all pair queries.
        """
        entities = ["A", "B", "C", "D"]
        for perm in itertools.permutations(entities):
            premises = [
                Premise(perm[i], perm[i + 1]) for i in range(len(perm) - 1)
            ]
            for u in entities:
                for v in entities:
                    if u != v:
                        q = Query(u, v)
                        res = SymbolicSolver.solve(premises, q, verify_with_oracle=True)
                        self.assertTrue(res.algorithm_agreement)
                        expected_status = (
                            EpistemicStatus.ENTAILED
                            if perm.index(u) < perm.index(v)
                            else EpistemicStatus.CONTRADICTED
                        )
                        self.assertEqual(res.epistemic_status, expected_status)


if __name__ == "__main__":
    unittest.main()
