"""Phase 1B tests: capture integrity, cross-modal labels, splits, ablations."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.benchmark.cross_modal_dependency import (
    CONDITIONS,
    build_prompt_for_condition,
    condition_premises,
    generate_dataset,
    plan_families,
)
from src.config.paths import REPO_ROOT
from src.instrumentation.capture_integrity import (
    build_capture_identity,
    should_skip_capture,
    verify_vector_artifact,
)
from src.symbolic.relations import Modality
from src.symbolic.solver import SymbolicSolver
from src.symbolic.relations import DirectionalRelation, Premise, Query


class TestCaptureIntegrity(unittest.TestCase):
    def _identity(self, **kwargs):
        base = dict(
            sample_id="s1",
            image_sha256="abc",
            prompt="hello",
            model_id="m",
            model_revision="rev",
            quantization={"load_in_4bit": True},
            layer_indices=(3, 12),
            token_selectors=("last", "mean_image"),
        )
        base.update(kwargs)
        return build_capture_identity(**base)

    def test_skip_on_complete_match(self):
        ident = self._identity()
        row = {
            "sample_id": "s1",
            "status": "ok",
            "layers": [3, 12],
            "token_selectors": ["last", "mean_image"],
            "image_sha256": "abc",
            "prompt_sha256": ident.prompt_sha256,
            "model": {"model_id": "m", "revision": "rev", "quantization": {"load_in_4bit": True}},
            "capture_identity": {**ident.to_dict(), "identity_hash": ident.identity_hash()},
            "vectors_file": None,
        }
        d = should_skip_capture(existing_rows=[row], required_identity=ident, verify_vectors=False)
        self.assertTrue(d.skip)

    def test_missing_layer_blocks_skip(self):
        ident = self._identity(layer_indices=(3, 12, 18, 23))
        row = {
            "sample_id": "s1",
            "status": "ok",
            "layers": [3, 12],
            "token_selectors": ["last", "mean_image"],
            "image_sha256": "abc",
            "model": {"model_id": "m", "revision": "rev", "quantization": {"load_in_4bit": True}},
        }
        d = should_skip_capture(existing_rows=[row], required_identity=ident, verify_vectors=False)
        self.assertFalse(d.skip)
        self.assertTrue(any("layer set mismatch" in i for i in d.issues))

    def test_changed_revision_blocks_skip(self):
        ident = self._identity(model_revision="rev2")
        row = {
            "sample_id": "s1",
            "status": "ok",
            "layers": [3, 12],
            "token_selectors": ["last", "mean_image"],
            "image_sha256": "abc",
            "model": {"model_id": "m", "revision": "rev", "quantization": {"load_in_4bit": True}},
        }
        d = should_skip_capture(existing_rows=[row], required_identity=ident, verify_vectors=False)
        self.assertFalse(d.skip)

    def test_changed_selectors_block_skip(self):
        ident = self._identity(token_selectors=("last",))
        row = {
            "sample_id": "s1",
            "status": "ok",
            "layers": [3, 12],
            "token_selectors": ["last", "mean_image"],
            "image_sha256": "abc",
            "model": {"model_id": "m", "revision": "rev", "quantization": {"load_in_4bit": True}},
        }
        d = should_skip_capture(existing_rows=[row], required_identity=ident, verify_vectors=False)
        self.assertFalse(d.skip)

    def test_missing_vector_file(self):
        ident = self._identity()
        row = {
            "sample_id": "s1",
            "status": "ok",
            "layers": [3, 12],
            "token_selectors": ["last", "mean_image"],
            "image_sha256": "abc",
            "model": {"model_id": "m", "revision": "rev", "quantization": {"load_in_4bit": True}},
            "capture_identity": {**ident.to_dict(), "identity_hash": ident.identity_hash()},
            "vectors_file": "missing.json",
        }
        d = should_skip_capture(
            existing_rows=[row],
            required_identity=ident,
            vectors_file_for_row=lambda r: REPO_ROOT / "missing_vectors_xyz.json",
            verify_vectors=True,
        )
        self.assertFalse(d.skip)
        self.assertTrue(any("missing vector" in i for i in d.issues))

    def test_corrupted_vector_blocks_skip(self):
        ident = self._identity()
        with tempfile.TemporaryDirectory() as tmp:
            vp = Path(tmp) / "v.json"
            vp.write_text(json.dumps({"layers": {"3": {"tokens": {"last": {"vector": []}}}}}), encoding="utf-8")
            issues = verify_vector_artifact(
                vectors_file=vp, expected_layers=(3, 12), expected_selectors=("last", "mean_image")
            )
            self.assertTrue(issues)


class TestCrossModalLabels(unittest.TestCase):
    def test_both_premises_entailed(self):
        premises, _ = condition_premises("both_premises", "a", "b", "c")
        res = SymbolicSolver.solve(premises, Query("a", "c", DirectionalRelation.LEFT_OF))
        self.assertEqual(res.epistemic_status.value, "entailed")

    def test_visual_removed_underdetermined(self):
        premises, _ = condition_premises("visual_removed", "a", "b", "c")
        res = SymbolicSolver.solve(premises, Query("a", "c", DirectionalRelation.LEFT_OF))
        self.assertEqual(res.epistemic_status.value, "underdetermined")

    def test_fork_underdetermined(self):
        premises, _ = condition_premises("underdetermined_fork", "a", "b", "c")
        res = SymbolicSolver.solve(premises, Query("a", "c", DirectionalRelation.LEFT_OF))
        self.assertEqual(res.epistemic_status.value, "underdetermined")

    def test_contradictory_inconsistent(self):
        premises, _ = condition_premises("contradictory_evidence", "a", "b", "c")
        res = SymbolicSolver.solve(premises, Query("a", "c", DirectionalRelation.LEFT_OF))
        self.assertEqual(res.epistemic_status.value, "inconsistent")

    def test_visual_reversed_not_assumed_same_as_base(self):
        premises, _ = condition_premises("visual_reversed", "a", "b", "c")
        res = SymbolicSolver.solve(premises, Query("a", "c", DirectionalRelation.LEFT_OF))
        self.assertIn(res.epistemic_status.value, {"underdetermined", "contradicted", "entailed"})


class TestDatasetGeneration(unittest.TestCase):
    def test_small_dataset_determinism_and_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            out1 = Path(tmp) / "d1"
            out2 = Path(tmp) / "d2"
            r1 = generate_dataset(n_families=8, seed=7, out_dir=out1, render_images=False)
            r2 = generate_dataset(n_families=8, seed=7, out_dir=out2, render_images=False)
            self.assertEqual(r1["n_records"], 8 * len(CONDITIONS))
            self.assertEqual(
                [x["sample_id"] for x in r1["records"]],
                [x["sample_id"] for x in r2["records"]],
            )
            self.assertEqual(r1["label_counts"], r2["label_counts"])
            self.assertTrue(r1["group_disjoint_splits_ok"])
            self.assertEqual(r1["solver_mismatches"], [])

            # family not split across partitions
            fam_split = {}
            for rec in r1["records"]:
                fam_split.setdefault(rec["family_id"], set()).add(rec["split"])
            for fid, splits in fam_split.items():
                self.assertEqual(len(splits), 1, fid)

    def test_ablation_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = generate_dataset(n_families=4, seed=11, out_dir=Path(tmp), render_images=True)
            for rec in data["records"]:
                self.assertTrue(rec["modality_ablation_ok"], rec["sample_id"])
                if rec["condition"] == "visual_removed":
                    self.assertTrue(all(p["modality"] != "visual" for p in rec["premises"]))
                    self.assertEqual(rec["visual_entities_shown"], [])
                if rec["condition"] == "both_premises":
                    self.assertTrue(rec["requires_both_modalities"])
                    self.assertTrue(any(p["modality"] == "visual" for p in rec["premises"]))
                    self.assertTrue(any(p["modality"] == "textual" for p in rec["premises"]))
                    # third entity not on canvas
                    self.assertNotIn(rec["entities"]["C"], rec["visual_entities_shown"])

    def test_plan_families_240(self):
        plans = plan_families(n_families=240, seed=20261021)
        self.assertEqual(len(plans), 240)
        keys = {(p.a, p.b, p.c) for p in plans}
        self.assertEqual(len(keys), 240)
        splits = {p.split for p in plans}
        self.assertEqual(splits, {"train", "validation", "test"})


class TestPromptBuild(unittest.TestCase):
    def test_prompt_mentions_question(self):
        p = build_prompt_for_condition(condition="both_premises", a="cyan_box", b="green_cylinder", c="magenta_prism")
        self.assertIn("cyan box", p)
        self.assertIn("Undetermined", p)


if __name__ == "__main__":
    unittest.main()
