"""Phase 2I-A tests: schedule coverage, gold masking, strict parser."""

from __future__ import annotations

import json
import unittest

try:
    import torch
    _TORCH_OK = True
except ImportError:
    torch = None
    _TORCH_OK = False

from src.config.paths import REPO_ROOT
from src.training.answer_parser import evaluation_accounting, gold_answer_string, strict_parse_answer
from src.training.sampling_and_masking import (
    answer_span_prompt_length,
    maximize_family_coverage_schedule,
)


class TestSchedule(unittest.TestCase):
    def _recs(self, n_families=40, per=3):
        recs = []
        for f in range(n_families):
            for i, lab in enumerate(("Yes", "No", "Undetermined")):
                recs.append(
                    {
                        "family_id": f"fam{f:03d}",
                        "sample_id": f"fam{f:03d}::{lab}",
                        "ground_truth_label": lab,
                        "split": "train",
                    }
                )
        return recs

    def test_coverage_maximizes_families(self):
        recs = self._recs(40)
        sched = maximize_family_coverage_schedule(recs, max_steps=50, seed=7)
        self.assertEqual(len(sched), 50)
        labs = [r["ground_truth_label"] for r in sched]
        # balanced as much as possible
        self.assertEqual(labs.count("Yes"), 17)
        self.assertEqual(labs.count("No"), 17)
        self.assertEqual(labs.count("Undetermined"), 16)
        fams = {r["family_id"] for r in sched}
        # should cover many families (not ~17 only)
        self.assertGreaterEqual(len(fams), 30)
        # unique samples in first 39 should be 39
        first39 = [r["sample_id"] for r in sched[:39]]
        self.assertEqual(len(first39), len(set(first39)))

    def test_deterministic(self):
        recs = self._recs()
        a = maximize_family_coverage_schedule(recs, max_steps=30, seed=3)
        b = maximize_family_coverage_schedule(recs, max_steps=30, seed=3)
        self.assertEqual([r["sample_id"] for r in a], [r["sample_id"] for r in b])

    def test_missing_class(self):
        recs = [{"family_id": "f", "sample_id": "s", "ground_truth_label": "Yes"}]
        with self.assertRaises(ValueError):
            maximize_family_coverage_schedule(recs, max_steps=3, seed=0)


@unittest.skipUnless(_TORCH_OK, "torch required")
class TestAnswerSpan(unittest.TestCase):
    def test_suffix_span(self):
        ids = torch.tensor([10, 11, 12, 5, 6])
        ans = [5, 6]
        self.assertEqual(answer_span_prompt_length(ids, ans), 3)

    def test_missing_span(self):
        ids = torch.tensor([10, 11, 12])
        with self.assertRaises(ValueError):
            answer_span_prompt_length(ids, [99, 98])


class TestStrictParser(unittest.TestCase):
    def test_parses_exact(self):
        self.assertEqual(strict_parse_answer(" Yes")["parsed_label"], "Yes")
        self.assertEqual(strict_parse_answer("No.")["parsed_label"], "No")
        self.assertEqual(strict_parse_answer(" Undetermined")["parsed_label"], "Undetermined")

    def test_invalid_not_coerced(self):
        r = strict_parse_answer("purple elephant flying")
        self.assertEqual(r["parsed_label"], "Invalid")
        self.assertEqual(r["parser_version"], "strict_answer_v1")

    def test_conflict(self):
        r = strict_parse_answer("Yes or No maybe")
        self.assertEqual(r["parsed_label"], "Conflicted")

    def test_empty(self):
        self.assertEqual(strict_parse_answer("")["parsed_label"], "Invalid")
        self.assertEqual(strict_parse_answer(None)["parsed_label"], "Missing")


class TestAccounting(unittest.TestCase):
    def test_invalid_rate(self):
        y_t = ["Yes", "No", "Undetermined", "Yes"]
        y_p = ["Yes", "Invalid", "Undetermined", "Conflicted"]
        m = evaluation_accounting(y_t, y_p)
        self.assertEqual(m["invalid_output_count"], 2)
        self.assertAlmostEqual(m["invalid_output_rate"], 0.5)


class TestProtocolFrozen(unittest.TestCase):
    def test_ce_only(self):
        cfg = json.loads((REPO_ROOT / "configs/phase_2i_a/ce_control_protocol_v1.json").read_text(encoding="utf-8"))
        self.assertFalse(cfg["training"]["custom_losses"])
        self.assertFalse(cfg["data"]["holdout_allowed"])
        self.assertEqual(cfg["training"]["max_steps"], 50)


if __name__ == "__main__":
    unittest.main()
