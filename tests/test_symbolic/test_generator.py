"""
Unit tests for deterministic benchmark dataset generation.
"""

from collections import Counter
import unittest

from src.symbolic.generator import (
    BenchmarkGenerator,
    HELD_OUT_ENTITIES,
    IN_DISTRIBUTION_ENTITIES,
    TOPOLOGIES,
)


class TestBenchmarkGenerator(unittest.TestCase):
    """Test suite for benchmark suite assembly, determinism, and vocabulary partitioning."""

    def test_seed_determinism(self) -> None:
        """Identical seeds must produce bitwise identical sample dictionaries."""
        gen1 = BenchmarkGenerator(seed=42)
        gen2 = BenchmarkGenerator(seed=42)

        dataset1 = gen1.generate_dataset(num_samples=25, split="in_distribution")
        dataset2 = gen2.generate_dataset(num_samples=25, split="in_distribution")

        self.assertEqual(dataset1, dataset2)

    def test_seed_divergence(self) -> None:
        """Different seeds must produce different entity samples."""
        gen1 = BenchmarkGenerator(seed=42)
        gen2 = BenchmarkGenerator(seed=999)

        dataset1 = gen1.generate_dataset(num_samples=25, split="in_distribution")
        dataset2 = gen2.generate_dataset(num_samples=25, split="in_distribution")

        self.assertNotEqual(dataset1, dataset2)

    def test_split_vocabulary_disjointness(self) -> None:
        """In-Distribution and Held-Out samples must never share entity vocabularies."""
        gen = BenchmarkGenerator(seed=123)

        id_samples = gen.generate_dataset(num_samples=50, split="in_distribution")
        ho_samples = gen.generate_dataset(num_samples=50, split="held_out")

        id_entities_used: set[str] = set()
        for s in id_samples:
            for p in s["symbolic_ground_truth"]["premises"]:
                id_entities_used.add(p["subject"])
                id_entities_used.add(p["object"])

        ho_entities_used: set[str] = set()
        for s in ho_samples:
            for p in s["symbolic_ground_truth"]["premises"]:
                ho_entities_used.add(p["subject"])
                ho_entities_used.add(p["object"])

        # Check disjointness
        self.assertTrue(
            id_entities_used.isdisjoint(ho_entities_used),
            f"Leakage detected between splits! Shared entities: {id_entities_used & ho_entities_used}",
        )
        # Check subset of master pools
        self.assertTrue(id_entities_used.issubset(set(IN_DISTRIBUTION_ENTITIES)))
        self.assertTrue(ho_entities_used.issubset(set(HELD_OUT_ENTITIES)))

    def test_balanced_topology_distribution(self) -> None:
        """N=100 samples across 5 topologies must yield exactly 20 per topology."""
        gen = BenchmarkGenerator(seed=42)
        samples = gen.generate_dataset(num_samples=100, split="in_distribution")

        top_counts = Counter(s["topology"] for s in samples)
        self.assertEqual(len(top_counts), len(TOPOLOGIES))
        for top in TOPOLOGIES:
            self.assertEqual(top_counts[top], 20)

    def test_synthetic_demo_quarantine_invariant(self) -> None:
        """Benchmark datasets must have is_synthetic_demo: False to protect empirical metrics."""
        gen = BenchmarkGenerator(seed=42)
        samples = gen.generate_dataset(num_samples=50, split="in_distribution")
        for s in samples:
            self.assertFalse(s["is_synthetic_demo"])

    def test_topology_epistemic_invariants(self) -> None:
        """Forward chains must be entailed, reverse contradicted, forks underdetermined."""
        gen = BenchmarkGenerator(seed=42)
        samples = gen.generate_dataset(num_samples=50, split="in_distribution")

        for s in samples:
            top = s["topology"]
            stat = s["symbolic_ground_truth"]["epistemic_status"]

            if top == "forward_chain":
                self.assertEqual(stat, "entailed")
            elif top == "reverse_chain":
                self.assertEqual(stat, "contradicted")
            elif top == "underdetermined_fork":
                self.assertEqual(stat, "underdetermined")


if __name__ == "__main__":
    unittest.main()
