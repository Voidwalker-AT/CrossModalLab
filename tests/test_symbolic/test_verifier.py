"""
Unit and integration tests for BenchmarkVerifier and round-trip artifact validation.
"""

import json
from pathlib import Path
import tempfile
import unittest

from src.config.paths import REPO_ROOT
from src.symbolic.generator import BenchmarkGenerator
from src.symbolic.verifier import BenchmarkVerifier


class TestBenchmarkVerifier(unittest.TestCase):
    """Test suite for Tier 1 and Tier 2 artifact verification."""

    def setUp(self) -> None:
        # Repository-relative scratch dir (portable; avoids hardcoded E: paths).
        # datasets/test_scratch is gitignored.
        self.workspace_tmp_dir = REPO_ROOT / "datasets" / "test_scratch"
        self.workspace_tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        # Clean up temporary test files in test_scratch
        if self.workspace_tmp_dir.exists():
            for f in self.workspace_tmp_dir.glob("*"):
                try:
                    f.unlink()
                except OSError:
                    pass
            try:
                self.workspace_tmp_dir.rmdir()
            except OSError:
                pass

    def test_roundtrip_1000_samples_verification(self) -> None:
        """
        Integration test:
        Generates 1,000 samples across seeds, serializes to JSONL, and audits
        with BenchmarkVerifier, asserting 100% concordance.
        """
        artifact_path = self.workspace_tmp_dir / "bench_1000.jsonl"
        gen = BenchmarkGenerator(seed=42)
        gen.generate_and_save(artifact_path, num_samples=1000, split="in_distribution")

        self.assertTrue(artifact_path.exists())

        report = BenchmarkVerifier.verify_file(artifact_path)
        self.assertTrue(report.is_valid, f"Verification failed with errors: {report.errors[:5]}")
        self.assertEqual(report.total_samples, 1000)
        self.assertEqual(report.valid_samples, 1000)
        self.assertEqual(report.status_match_count, 1000)
        self.assertEqual(report.extensions_match_count, 1000)
        self.assertEqual(len(report.errors), 0)

        # Confirm topology balance (200 of each topology)
        for top, count in report.topology_counts.items():
            self.assertEqual(count, 200, f"Expected 200 for {top}, got {count}")

    def test_detection_of_tampered_status(self) -> None:
        """Verifier must catch discrepancy between recorded and re-derived status."""
        gen = BenchmarkGenerator(seed=42)
        samples = gen.generate_dataset(num_samples=1, split="in_distribution")
        sample = samples[0]

        # Deliberately invert epistemic status
        original_status = sample["symbolic_ground_truth"]["epistemic_status"]
        tampered_status = "contradicted" if original_status != "contradicted" else "entailed"
        sample["symbolic_ground_truth"]["epistemic_status"] = tampered_status

        errors = BenchmarkVerifier.verify_sample(sample, line_num=1)
        self.assertGreaterEqual(len(errors), 1)
        self.assertIn("Epistemic status discrepancy", errors[0])

    def test_detection_of_insufficient_premises(self) -> None:
        """Verifier must catch records with fewer than 2 premises."""
        gen = BenchmarkGenerator(seed=42)
        samples = gen.generate_dataset(num_samples=1, split="in_distribution")
        sample = samples[0]

        # Truncate premises to 1 item
        sample["symbolic_ground_truth"]["premises"] = sample["symbolic_ground_truth"]["premises"][:1]

        errors = BenchmarkVerifier.verify_sample(sample, line_num=1)
        self.assertGreaterEqual(len(errors), 1)
        self.assertIn("must have at least 2 items", errors[0])

    def test_detection_of_empty_sample_id(self) -> None:
        """Verifier must catch empty sample_id."""
        gen = BenchmarkGenerator(seed=42)
        samples = gen.generate_dataset(num_samples=1, split="in_distribution")
        sample = samples[0]

        sample["sample_id"] = ""
        errors = BenchmarkVerifier.verify_sample(sample, line_num=1)
        self.assertGreaterEqual(len(errors), 1)
        self.assertIn("'sample_id' must be a non-empty string", errors[0])

    def test_detection_of_tampered_extensions(self) -> None:
        """Verifier must catch mismatch in linear extensions."""
        gen = BenchmarkGenerator(seed=42)
        samples = gen.generate_dataset(num_samples=1, split="in_distribution")
        sample = samples[0]

        # Tamper linear extensions
        sample["symbolic_ground_truth"]["consistent_linear_extensions"] = [["fake_a", "fake_b", "fake_c"]]
        errors = BenchmarkVerifier.verify_sample(sample, line_num=1)
        self.assertGreaterEqual(len(errors), 1)
        self.assertIn("Linear extensions mismatch", errors[0])

    def test_detection_of_nonexistent_file(self) -> None:
        missing = REPO_ROOT / "datasets" / "non_existent.jsonl"
        report = BenchmarkVerifier.verify_file(str(missing))
        self.assertFalse(report.is_valid)
        self.assertIn("File not found", report.errors[0])


if __name__ == "__main__":
    unittest.main()
