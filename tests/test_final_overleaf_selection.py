"""CPU evidence-selection regressions; no model load or restricted dataset read."""
import csv
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.audit_dual_model_evidence import compare_inputs, load, ROOT
from scripts.verify_final_overleaf_selection import verify, archive

PACKAGE = ROOT / 'paper/overleaf_submission_selection_20261025'


class FinalSelectionTests(unittest.TestCase):
    def test_same_inputs_different_endpoint_not_direct_comparison(self):
        row = dict(sample_id='fixture', image_sha256='fixture', prompt_sha256='fixture', ground_truth_label='No')
        self.assertEqual(compare_inputs(row, row, left_method='generation', right_method='sum_LL'), 'PARTIALLY COMPARABLE')
        self.assertEqual(compare_inputs(row, row, left_method='sum_LL', right_method='sum_LL'), 'DIRECTLY COMPARABLE')
        self.assertEqual(compare_inputs(row, dict(row, prompt_sha256='other'), left_method='sum_LL', right_method='sum_LL'), 'NOT COMPARABLE')

    def test_acquisition_roles_not_mistaken_for_probe_evaluation_split(self):
        data = load(PACKAGE / 'audit/dual_model_audit.json')
        split = data['neural']['split_metadata']
        self.assertEqual(split['acquisition_roles'], dict(train=105, validation=3))
        self.assertEqual(split['evaluation_rows'], dict(train=81, validation=27))
        self.assertEqual(split['role_membership_disagreements'], 30)
        probe = data['neural']['probes']
        self.assertFalse(set(probe['train_families']) & set(probe['val_families']))

    def test_selected_metrics_and_sources(self):
        # A contributor can validate the committed selection without ignored NPZs.
        # Full original-source validation is a separate primary-machine check.
        result = verify(PACKAGE, original_sources=False, require_manifest=False)
        self.assertEqual(result['smol_cases'], 36)
        self.assertEqual(result['smol_matched_groups'], 30)
        self.assertEqual(result['qwen_generation_rows'], 108)
        self.assertEqual(result['new_model_forwards'], 0)
        self.assertEqual(result['bibliography_entries'], 14)

    def test_tampered_behavior_table_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as temp:
            copy = Path(temp) / 'selection'
            shutil.copytree(PACKAGE, copy)
            p = copy / 'tables/t3_qwen_behavior.csv'
            with p.open(encoding='utf-8', newline='') as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            rows[0]['BA'] = '0.999999'
            with p.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'table source mismatch'):
                verify(copy, original_sources=False, require_manifest=False)

    def test_original_archive_cannot_be_overwritten(self):
        original = ROOT / 'submission/CrossModalLab_Overleaf_Evidence_20261025.zip'
        before = hashlib.sha256(original.read_bytes()).hexdigest()
        with self.assertRaises(FileExistsError):
            archive(PACKAGE, original)
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), before)


if __name__ == '__main__':
    unittest.main()
