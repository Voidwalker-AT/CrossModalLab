"""CPU-only checks for frozen development selection and fail-closed admission."""
import tempfile
import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_qwen_ram_recovery as recovery
from scripts.verify_qwen_ram_recovery import verify


class RecoveryTests(unittest.TestCase):
    evidence = recovery.ROOT / 'artifacts/publication/20261025/qwen_replication_20261025_464531de'

    def test_completed_evidence(self):
        result = verify(self.evidence)
        self.assertEqual(result['actual_forwards'], 9)
        self.assertEqual(result['parameter_tensors_unchanged'], 625)
        self.assertEqual(result['generation_metrics'], 'NOT_APPLICABLE')

    def test_corrupted_accounting_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'evidence'
            shutil.copytree(self.evidence, target)
            path = target / 'execution.json'
            data = json.loads(path.read_text(encoding='utf-8'))
            data['attempted_forwards'] = 8
            path.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'accounting mismatch'):
                verify(target)

    def test_frozen_development_identity(self):
        protocol, rows = recovery.selected_cases()
        self.assertEqual([r['sample_id'] for r in rows], [
            'xc_ho_0011::entailed', 'xc_ho_0011::contradicted', 'xc_ho_0011::underdetermined'])
        self.assertEqual(len(rows) * len(protocol['candidates']), 9)
        self.assertLessEqual(9, protocol['hard_total_cap'])

    def test_protocol_tampering_rejected_before_model_import(self):
        with patch.object(recovery, 'PROTOCOL_HASH', '0' * 64):
            with self.assertRaisesRegex(ValueError, 'protocol identity'):
                recovery.selected_cases()

    def test_resource_gates(self):
        with patch('scripts.run_publication_isolated_pilot.shutil.disk_usage') as disk:
            disk.return_value.free = 8 * 1024**3
            safe = dict(host_available_mib=4096, nvidia_total_mib=4096, nvidia_used_mib=500)
            recovery.gate(safe, loading=True)
            for changes in ({'host_available_mib': 4095}, {'nvidia_used_mib': 3901},
                            {'nvidia_used_mib': 597}):
                with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                    recovery.gate(dict(safe, **changes), loading=True)

    def test_failed_attempt_is_spent_and_ledger_cannot_reset(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'ledger'
            budget = recovery.ForwardBudget(path, limit=1)
            try:
                budget.consume(sample_id='cpu_fixture', candidate_index=0)
                raise RuntimeError('simulated failure inside model')
            except RuntimeError:
                pass
            self.assertEqual(budget.used, 1)
            self.assertTrue((path / 'forward_001.json').is_file())
            with self.assertRaises(RuntimeError):
                budget.consume()
            with self.assertRaises(ValueError):
                recovery.ForwardBudget(path, limit=1)


if __name__ == '__main__':
    unittest.main()
