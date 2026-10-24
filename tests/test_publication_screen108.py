import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.config.paths import REPO_ROOT
from src.evaluation.screen_budget import ScreenBudget

spec=importlib.util.spec_from_file_location('screen108',REPO_ROOT/'scripts/run_publication_screen108.py')
screen=importlib.util.module_from_spec(spec);spec.loader.exec_module(screen)


class ApprovedScreenTests(unittest.TestCase):
    def test_exact_frozen_cases_and_content_order(self):
        rows=screen.selected_cases()
        self.assertEqual(len(rows),18)
        self.assertEqual(len({r['sample_id'] for r in rows}),18)
        self.assertEqual(len({r['family_id'] for r in rows}),6)
        self.assertTrue(all(sorted(r['content_order'])==['image','text'] for r in rows))

    def test_worker_quota_is_persistent_and_exactly_54(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT/'.cache') as name:
            path=Path(name)/'ledger';budget=ScreenBudget(path)
            for i in range(54):budget.consume(candidate_index=i%3,model='fixture')
            with self.assertRaisesRegex(RuntimeError,'exhausted'):budget.consume()
            with self.assertRaisesRegex(ValueError,'reset'):ScreenBudget(path)
            self.assertEqual(len(list(path.glob('forward_*.json'))),54)

    def test_free_vram_gate_uses_reported_free_not_total_minus_used(self):
        snapshot={'host_available_mib':5000,'nvidia_total_mib':4096,'nvidia_used_mib':500,'nvidia_free_mib':3463}
        with self.assertRaisesRegex(RuntimeError,'dedicated'):screen.gate(snapshot,loading=True)

    def test_modified_approved_plan_refused_before_loading(self):
        with patch.object(screen,'file_hash',return_value='changed'):
            with self.assertRaisesRegex(ValueError,'byte identity'):screen.selected_cases()

    def test_failed_gate_retains_triggering_sample_in_exception(self):
        for sample in ({'host_available_mib':4095,'nvidia_used_mib':100},
                       {'host_available_mib':5000,'nvidia_used_mib':3901}):
            with self.assertRaisesRegex(RuntimeError,str(sample['host_available_mib'])):screen.gate(sample)


if __name__=='__main__':unittest.main()
