"""Phase 2B recovery regression tests (CPU-only)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.causal.recovery import (
    compare_control_vs_targeted,
    missing_interventions,
    validate_intervention_jsonl,
)
from src.config.paths import REPO_ROOT

try:
    import numpy as np
    import torch
    from src.causal.patch_runner import residual_patch_context
    _TORCH_OK = True
except ImportError:
    np = None
    torch = None
    residual_patch_context = None
    _TORCH_OK = False


class TestArtifactValidation(unittest.TestCase):
    def test_real_artifact_validates(self):
        path = REPO_ROOT / "artifacts/causal/phase_2b/intervention_records_v1.jsonl"
        if not path.is_file():
            self.skipTest("no phase 2b artifact")
        report = validate_intervention_jsonl(path)
        self.assertTrue(report["ok"])
        self.assertEqual(report["n_bad_lines"], 0)
        self.assertEqual(report["duplicate_ids"], [])

    def test_immutable_copy_detects_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.jsonl"
            p.write_text(
                '{"intervention_id":"a","status":"ok","role":"baseline_source"}\n'
                '{"intervention_id":"b","status":"ok","role":"interven',
                encoding="utf-8",
            )
            report = validate_intervention_jsonl(p)
            self.assertEqual(report["n_valid"], 1)
            self.assertEqual(report["n_bad_lines"], 1)
            self.assertFalse(report["ok"])


class TestResumeMatrix(unittest.TestCase):
    def test_missing_computation(self):
        rows = [
            {"intervention_id": "f1:entailed_vs_underdetermined::baseline_source", "status": "ok"},
            {"intervention_id": "f1:entailed_vs_underdetermined::source_to_target", "status": "ok"},
        ]
        missing = missing_interventions(
            rows,
            ["f1"],
            ["entailed_vs_underdetermined"],
            ["source_to_target", "same_label"],
        )
        self.assertIn("f1:entailed_vs_underdetermined::baseline_target", missing)
        self.assertIn("f1:entailed_vs_underdetermined::same_label", missing)
        self.assertNotIn("f1:entailed_vs_underdetermined::source_to_target", missing)


class TestControlComparison(unittest.TestCase):
    def test_flag_when_controls_larger(self):
        rows = [
            {"role": "intervention", "direction": "source_to_target", "delta_m": {"delta_m": 0.1}},
            {"role": "intervention", "direction": "random_source", "delta_m": {"delta_m": -2.0}},
            {"role": "intervention", "direction": "unrelated_token", "delta_m": {"delta_m": -1.5}},
        ]
        out = compare_control_vs_targeted(rows)
        self.assertEqual(out["scientific_flag"], "GENERIC_INTERVENTION_SENSITIVITY")


class TestHookLifetime(unittest.TestCase):
    def test_patch_context_counts_multiple_forwards(self):
        if not _TORCH_OK:
            self.skipTest("torch required")

        class Block(torch.nn.Module):
            def forward(self, x):
                return x

        class Wrap(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.text_model = torch.nn.Module()
                self.model.text_model.layers = torch.nn.ModuleList([Block(), Block()])

        m = Wrap()
        row = np.ones(8, dtype=np.float32) * 3.0
        with residual_patch_context(m, layer_index=1, token_position=1, replacement_row=row) as ctx:
            for _ in range(3):
                _ = m.model.text_model.layers[1](torch.zeros(1, 4, 8))
            self.assertEqual(ctx.applied, 3)
        _ = m.model.text_model.layers[1](torch.zeros(1, 4, 8))
        self.assertEqual(ctx.applied, 3)

    def test_direction_and_margin_definition(self):
        if not _TORCH_OK:
            self.skipTest("torch required for margin helpers")
        from src.causal.patch_runner import delta_margin, score_margins_from_logprobs

        lp = {" Yes": -0.2, " Undetermined": -1.2}
        m = score_margins_from_logprobs(lp, " Yes", " Undetermined")
        self.assertAlmostEqual(m["margin_source_minus_target"], 1.0)
        d = delta_margin(1.0, 1.5)
        self.assertAlmostEqual(d["delta_m"], 0.5)


class TestConfigVersion(unittest.TestCase):
    def test_exec_spec_version(self):
        spec = json.loads(
            (REPO_ROOT / "configs/phase_2b/causal_pilot_exec_spec_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(spec["schema_version"], "phase_2b_exec_spec_v1")


if __name__ == "__main__":
    unittest.main()
