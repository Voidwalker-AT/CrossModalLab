"""
Unit tests for RelationGraph, cycle detection, reachability, and linear extensions.
"""

import unittest

from src.symbolic.graph import RelationGraph
from src.symbolic.relations import (
    InconsistentPremisesError,
    InvalidRelationError,
    Premise,
)


class TestRelationGraph(unittest.TestCase):
    """Test suite for graph-theoretic partial order operations."""

    def test_node_and_edge_addition(self) -> None:
        graph = RelationGraph()
        graph.add_relation("A", "B")
        self.assertIn("A", graph.entities)
        self.assertIn("B", graph.entities)
        self.assertIn(("A", "B"), graph.edges)
        self.assertEqual(len(graph.edges), 1)

    def test_self_relation_rejection(self) -> None:
        graph = RelationGraph()
        with self.assertRaises(InvalidRelationError):
            graph.add_relation("A", "A")

    def test_asymmetry_violation(self) -> None:
        """Adding B -> A when A -> B exists must raise InconsistentPremisesError."""
        graph = RelationGraph()
        graph.add_relation("A", "B")
        with self.assertRaises(InconsistentPremisesError) as ctx:
            graph.add_relation("B", "A")
        self.assertIn("Asymmetry violation", str(ctx.exception))

    def test_transitive_cycle_detection_3_nodes(self) -> None:
        """A -> B -> C -> A must be flagged as a cycle."""
        graph = RelationGraph()
        graph.add_relation("A", "B")
        graph.add_relation("B", "C")
        with self.assertRaises(InconsistentPremisesError) as ctx:
            graph.add_relation("C", "A")
        self.assertIn("Cycle detected", str(ctx.exception))

    def test_transitive_cycle_detection_4_nodes(self) -> None:
        """A -> B -> C -> D -> A must be flagged as a cycle."""
        graph = RelationGraph()
        graph.add_relation("A", "B")
        graph.add_relation("B", "C")
        graph.add_relation("C", "D")
        with self.assertRaises(InconsistentPremisesError) as ctx:
            graph.add_relation("D", "A")
        self.assertIn("Cycle detected", str(ctx.exception))

    def test_has_cycle_predicate(self) -> None:
        # Acyclic graph
        graph = RelationGraph()
        graph.add_relation("A", "B", check_cycle=False)
        graph.add_relation("B", "C", check_cycle=False)
        self.assertFalse(graph.has_cycle())

        # Manually constructed cyclic graph (bypassing check_cycle on add)
        graph_cyclic = RelationGraph()
        graph_cyclic.add_relation("A", "B", check_cycle=False)
        graph_cyclic.add_relation("B", "C", check_cycle=False)
        graph_cyclic.add_relation("C", "A", check_cycle=False)
        self.assertTrue(graph_cyclic.has_cycle())
        cycles = graph_cyclic.find_cycles()
        self.assertGreaterEqual(len(cycles), 1)

    def test_reachability(self) -> None:
        graph = RelationGraph()
        graph.add_relation("A", "B")
        graph.add_relation("B", "C")
        graph.add_relation("C", "D")
        graph.add_entity("Z")  # Disconnected entity

        self.assertTrue(graph.is_reachable("A", "B"))
        self.assertTrue(graph.is_reachable("A", "C"))
        self.assertTrue(graph.is_reachable("A", "D"))
        self.assertFalse(graph.is_reachable("D", "A"))
        self.assertFalse(graph.is_reachable("C", "A"))
        self.assertFalse(graph.is_reachable("A", "Z"))
        self.assertFalse(graph.is_reachable("Z", "A"))

    def test_transitive_closure(self) -> None:
        graph = RelationGraph()
        graph.add_relation("A", "B")
        graph.add_relation("B", "C")
        closure = graph.transitive_closure()
        self.assertEqual(closure, {("A", "B"), ("B", "C"), ("A", "C")})

    def test_linear_extensions_chain(self) -> None:
        """Linear chain A -> B -> C has exactly 1 topological ordering: [A, B, C]."""
        graph = RelationGraph()
        graph.add_relation("A", "B")
        graph.add_relation("B", "C")
        exts = graph.get_linear_extensions()
        self.assertEqual(len(exts), 1)
        self.assertEqual(exts[0], ["A", "B", "C"])

    def test_linear_extensions_fan_in_fork(self) -> None:
        """Fork A -> B, C -> B has exactly 2 topological orderings: [A, C, B] and [C, A, B]."""
        graph = RelationGraph()
        graph.add_relation("A", "B")
        graph.add_relation("C", "B")
        exts = graph.get_linear_extensions()
        self.assertEqual(len(exts), 2)
        self.assertIn(["A", "C", "B"], exts)
        self.assertIn(["C", "A", "B"], exts)

    def test_linear_extensions_fan_out_fork(self) -> None:
        """Fork B -> A, B -> C has exactly 2 topological orderings: [B, A, C] and [B, C, A]."""
        graph = RelationGraph()
        graph.add_relation("B", "A")
        graph.add_relation("B", "C")
        exts = graph.get_linear_extensions()
        self.assertEqual(len(exts), 2)
        self.assertIn(["B", "A", "C"], exts)
        self.assertIn(["B", "C", "A"], exts)

    def test_linear_extensions_disjoint_pairs(self) -> None:
        """Disjoint pairs A -> B and C -> D have (4! / (2! * 2!)) = 6 topological orderings."""
        graph = RelationGraph()
        graph.add_relation("A", "B")
        graph.add_relation("C", "D")
        exts = graph.get_linear_extensions()
        self.assertEqual(len(exts), 6)
        # Verify edge constraints in every extension
        for ext in exts:
            self.assertLess(ext.index("A"), ext.index("B"))
            self.assertLess(ext.index("C"), ext.index("D"))

    def test_linear_extensions_cyclic_graph_empty(self) -> None:
        graph = RelationGraph()
        graph.add_relation("A", "B", check_cycle=False)
        graph.add_relation("B", "A", check_cycle=False)
        self.assertEqual(graph.get_linear_extensions(), [])


if __name__ == "__main__":
    unittest.main()
