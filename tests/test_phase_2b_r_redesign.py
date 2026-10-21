"""Phase 2B-R tests: exact pairs, scoring hook lifetime, telemetry schema."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.causal.pair_types import classify_pair
from src.causal.telemetry import MemorySnapshot, should_stop_on_memory
from src.config.paths import REPO_ROOT

try:
    import torch
    from src.causal.scoring import score_complete_candidates
    from src.causal.patch_runner import residual_patch_context
    _TORCH_OK = True
except ImportError:
    torch = None
    score_complete_candidates = None
    residual_patch_context = None
    _TORCH_OK = False


class TestPairClassification(unittest.TestCase):
    def test_exact_image_changed_text(self):
        s = {
            "family_id": "f",
            "image_sha256": "aaa",
            "premises": [
                {"modality": "visual", "subject": "a", "object": "b"},
                {"modality": "textual", "subject": "b", "object": "c"},
            ],
            "query": {"subject": "a", "object": "c"},
            "entities": {"A": "a", "B": "b", "C": "c"},
            "prompt": "p1",
            "sample_id": "f::entailed",
            "ground_truth_label": "Yes",
        }
        t = dict(s)
        t["premises"] = [
            {"modality": "visual", "subject": "a", "object": "b"},
            {"modality": "textual", "subject": "c", "object": "b"},
        ]
        t["prompt"] = "p2"
        t["sample_id"] = "f::underdetermined"
        t["ground_truth_label"] = "Undetermined"
        out = classify_pair(s, t)
        self.assertEqual(out["pair_type"], "EXACT_IMAGE_CHANGED_TEXT")
        self.assertTrue(out["eligible_for_primary_causal"])

    def test_multi_factor_when_both_change(self):
        s = {
            "family_id": "f",
            "image_sha256": "aaa",
            "premises": [
                {"modality": "visual", "subject": "a", "object": "b"},
                {"modality": "textual", "subject": "b", "object": "c"},
            ],
            "query": {"subject": "a", "object": "c"},
            "entities": {"A": "a", "B": "b", "C": "c"},
            "prompt": "p1",
            "sample_id": "s",
            "ground_truth_label": "Yes",
        }
        t = dict(s)
        t["image_sha256"] = "bbb"
        t["premises"][1]["subject"] = "c"
        t["premises"][1]["object"] = "b"
        t["prompt"] = "p2"
        out = classify_pair(s, t)
        self.assertEqual(out["pair_type"], "MULTI_FACTOR")
        self.assertFalse(out["eligible_for_primary_causal"])


class TestTelemetrySchema(unittest.TestCase):
    def test_stop_conditions(self):
        snap = MemorySnapshot(
            torch_allocated_mib=3900.0,
            torch_reserved_mib=3900.0,
            torch_max_allocated_mib=3900.0,
            torch_max_reserved_mib=3900.0,
            nvidia_smi_used_mib=3500.0,
            nvidia_smi_total_mib=4096.0,
            host_available_mib=2000.0,
            warnings=[],
        )
        out = should_stop_on_memory(snap)
        self.assertTrue(out["stop"])


class TestScoringHook(unittest.TestCase):
    @unittest.skipUnless(_TORCH_OK, "torch required")
    def test_prompt_position_guard(self):
        class Block(torch.nn.Module):
            def forward(self, x):
                return x

        class Wrap(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.text_model = torch.nn.Module()
                self.model.text_model.layers = torch.nn.ModuleList([Block()])

        m = Wrap()
        inputs = {"input_ids": torch.ones(1, 5, dtype=torch.long)}
        # token_position == prompt_len should be rejected
        with self.assertRaises(ValueError):
            score_complete_candidates(
                m,
                processor=None,
                base_inputs=inputs,
                candidate_strings=[" Yes"],
                layer_index=0,
                token_position=5,
                replacement_row=np.zeros(4, dtype=np.float32),
                no_op=True,
            )


class TestExactPairsArtifact(unittest.TestCase):
    def test_dataset_if_present(self):
        meta = REPO_ROOT / "datasets/exact_pairs_v1/exact_pairs_meta_v1.json"
        if not meta.is_file():
            self.skipTest("exact pairs not built")
        data = json.loads(meta.read_text(encoding="utf-8"))
        self.assertTrue(data["image_hash_shared_per_family"])


class TestHistoricalImmutable(unittest.TestCase):
    def test_phase2b_jsonl_sha_stable_if_present(self):
        p = REPO_ROOT / "artifacts/causal/phase_2b/intervention_records_v1.jsonl"
        if not p.is_file():
            self.skipTest("missing")
        import hashlib

        h = hashlib.sha256(p.read_bytes()).hexdigest()
        self.assertEqual(h, "50bb225b97e29b394b9a746a1c35134c02cf95d1203ad1126cecb0fcd512e337")


if __name__ == "__main__":
    unittest.main()
