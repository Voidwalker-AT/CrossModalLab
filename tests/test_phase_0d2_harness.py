"""Phase 0D.2 harness unit tests (CPU-only; no torch/GPU required for most tests)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.config.paths import REPO_ROOT, get_datasets_dir
from src.evaluation.behavioral_harness import (
    append_jsonl,
    load_completed_sample_ids,
    load_jsonl,
    sha256_file,
    validate_resume_compatibility,
)
from src.evaluation.behavioral_metrics import (
    balanced_accuracy,
    bootstrap_ci_binary,
    build_prompt,
    compute_behavioral_metrics,
    confusion_3x3,
    parse_model_answer,
    unsupported_definite_rate,
)
from src.evaluation.phase_0d2_manifest import build_phase_0d2_manifest
from src.symbolic.solver import SymbolicSolver
from src.symbolic.relations import DirectionalRelation, Modality, Premise, Query


class TestAnswerParsing(unittest.TestCase):
    def test_yes(self):
        self.assertEqual(parse_model_answer(" Yes.")["parsed_label"], "Yes")

    def test_no(self):
        self.assertEqual(parse_model_answer("No")["parsed_label"], "No")

    def test_undetermined(self):
        self.assertEqual(parse_model_answer(" Undetermined.")["parsed_label"], "Undetermined")

    def test_invalid(self):
        self.assertEqual(parse_model_answer("purple elephant")["parsed_label"], "Invalid")

    def test_empty(self):
        self.assertEqual(parse_model_answer("")["parsed_label"], "Invalid")

    def test_missing(self):
        self.assertEqual(parse_model_answer(None)["parsed_label"], "Missing")


class TestPromptTemplate(unittest.TestCase):
    def test_visual_prompt_mentions_image(self):
        p = build_prompt("A is left of B.", "is A to the left of B?", modality="visual")
        self.assertIn("image and statement", p)
        self.assertIn("Undetermined", p)

    def test_text_only_prompt(self):
        p = build_prompt("A is left of B.", "is A to the left of B?", modality="text_only")
        self.assertIn("statements", p)


class TestMetrics(unittest.TestCase):
    def test_confusion_and_balanced_acc(self):
        y_true = ["Yes", "No", "Undetermined", "Yes"]
        y_pred = ["Yes", "No", "Yes", "No"]
        conf = confusion_3x3(y_true, y_pred)
        self.assertEqual(conf["Yes"]["Yes"], 1)
        self.assertEqual(conf["Undetermined"]["Yes"], 1)
        ba = balanced_accuracy(y_true, y_pred)
        self.assertGreater(ba, 0.0)
        self.assertLessEqual(ba, 1.0)

    def test_unsupported_definite_rate(self):
        rate = unsupported_definite_rate(
            ["Undetermined", "Undetermined", "Yes"],
            ["Yes", "Undetermined", "Yes"],
        )
        self.assertEqual(rate["n_underdetermined"], 2)
        self.assertEqual(rate["n_definite"], 1)
        self.assertAlmostEqual(rate["rate"], 0.5)

    def test_bootstrap_ci(self):
        ci = bootstrap_ci_binary([True, True, False, False], n_boot=1000, seed=1)
        self.assertEqual(ci["n"], 4)
        self.assertAlmostEqual(ci["mean"], 0.5)
        self.assertLessEqual(ci["ci_low"], ci["mean"])
        self.assertGreaterEqual(ci["ci_high"], ci["mean"])

    def test_metrics_from_fake_records(self):
        recs = [
            {
                "sample_id": "a",
                "prediction_status": "ok",
                "ground_truth_label": "Yes",
                "parsed_answer": "Yes",
                "condition": "forward_chain",
                "modality": "cross_modal",
                "relation": "left_of",
                "matched_group_id": "g1",
                "latency_sec": 0.5,
                "peak_vram_mib": 100.0,
            },
            {
                "sample_id": "b",
                "prediction_status": "ok",
                "ground_truth_label": "No",
                "parsed_answer": "Yes",
                "condition": "forward_chain",
                "modality": "cross_modal",
                "relation": "left_of",
                "matched_group_id": "g1",
                "latency_sec": 0.4,
                "peak_vram_mib": 100.0,
            },
            {
                "sample_id": "c",
                "prediction_status": "ok",
                "ground_truth_label": "Undetermined",
                "parsed_answer": "Undetermined",
                "condition": "underdetermined_fork",
                "modality": "cross_modal",
                "relation": "left_of",
                "matched_group_id": None,
                "latency_sec": 0.6,
                "peak_vram_mib": 120.0,
            },
        ]
        m = compute_behavioral_metrics(recs)
        self.assertEqual(m["n_scored"], 3)
        self.assertEqual(m["matched_counterfactual_consistency"]["n_groups"], 1)
        self.assertEqual(m["matched_counterfactual_consistency"]["n_groups_all_correct"], 0)


class TestResumeHelpers(unittest.TestCase):
    def test_append_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preds.jsonl"
            append_jsonl(path, {"sample_id": "s1", "prediction_status": "ok"})
            append_jsonl(path, {"sample_id": "s1", "prediction_status": "ok"})
            append_jsonl(path, {"sample_id": "s2", "prediction_status": "failed"})
            done = load_completed_sample_ids(path)
            self.assertIn("s1", done)
            self.assertNotIn("s2", done)
            rows = load_jsonl(path)
            self.assertEqual(len(rows), 3)

    def test_resume_validation_detects_mismatch(self):
        rows = [
            {
                "sample_id": "s1",
                "prediction_status": "ok",
                "manifest_sha256": "abc",
                "model": {"revision": "rev1", "model_id": "m"},
                "candidate_scores": {
                    " Yes": {"total_log_likelihood": -1.0, "length_normalized_log_likelihood": -1.0}
                },
            }
        ]
        ok = validate_resume_compatibility(
            existing_rows=rows,
            manifest_sha256="abc",
            expected_revision="rev1",
            expected_model_id="m",
        )
        self.assertTrue(ok["safe_to_resume"])
        bad = validate_resume_compatibility(
            existing_rows=rows,
            manifest_sha256="zzz",
            expected_revision="rev2",
            expected_model_id="m",
        )
        self.assertFalse(bad["safe_to_resume"])

    def test_resume_validation_detects_duplicate_ok(self):
        row = {
            "sample_id": "s1",
            "prediction_status": "ok",
            "manifest_sha256": "abc",
            "model": {"revision": "rev1", "model_id": "m"},
            "candidate_scores": {},
        }
        check = validate_resume_compatibility(
            existing_rows=[row, dict(row)],
            manifest_sha256="abc",
            expected_revision="rev1",
            expected_model_id="m",
        )
        self.assertFalse(check["safe_to_resume"])
        self.assertEqual(check["duplicate_ok_ids"], ["s1"])


class TestManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build_phase_0d2_manifest()

    def test_counts(self):
        self.assertEqual(self.manifest["n_samples"], 36)
        self.assertEqual(
            self.manifest["condition_counts"],
            {
                "forward_chain": 6,
                "reverse_chain": 6,
                "underdetermined_fork": 6,
                "visual_premise_control": 6,
                "text_premise_control": 6,
                "directional_reversal_control": 6,
            },
        )

    def test_unique_ids(self):
        ids = [s["sample_id"] for s in self.manifest["samples"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_determinism(self):
        again = build_phase_0d2_manifest()
        self.assertEqual(again["samples_sha256"], self.manifest["samples_sha256"])

    def test_images_exist_and_hash_source(self):
        for s in self.manifest["samples"]:
            self.assertTrue(s["image_exists"], s["sample_id"])
            if s.get("image_file"):
                path = REPO_ROOT / s["image_file"]
                self.assertTrue(path.is_file())
                if s["condition"] != "text_premise_control":
                    self.assertEqual(sha256_file(path), s["image_sha256"])

    def test_label_correctness_forward_and_fork(self):
        by_cond = {}
        for s in self.manifest["samples"]:
            by_cond.setdefault(s["condition"], []).append(s)
        for s in by_cond["forward_chain"]:
            self.assertEqual(s["ground_truth_label"], "Yes")
            self.assertEqual(s["symbolic_epistemic_status"], "entailed")
        for s in by_cond["reverse_chain"]:
            self.assertEqual(s["ground_truth_label"], "No")
            self.assertEqual(s["symbolic_epistemic_status"], "contradicted")
        for s in by_cond["underdetermined_fork"]:
            self.assertEqual(s["ground_truth_label"], "Undetermined")
        for s in by_cond["directional_reversal_control"]:
            self.assertEqual(s["ground_truth_label"], "No")

    def test_source_jsonl_unchanged(self):
        # Manifest builder must not mutate source ground truth file hash content
        src = get_datasets_dir() / "controlled" / "controlled_benchmark_180.jsonl"
        self.assertEqual(sha256_file(src), self.manifest["source_jsonl_sha256"])


class TestSymbolicAgreement(unittest.TestCase):
    def test_forward_chain_entailed(self):
        premises = [
            Premise("cyan_box", "green_cylinder", DirectionalRelation.LEFT_OF, Modality.VISUAL),
            Premise("green_cylinder", "magenta_prism", DirectionalRelation.LEFT_OF, Modality.TEXTUAL),
        ]
        q = Query("cyan_box", "magenta_prism", DirectionalRelation.LEFT_OF)
        res = SymbolicSolver.solve(premises, q)
        self.assertEqual(res.epistemic_status.value, "entailed")


if __name__ == "__main__":
    unittest.main()
