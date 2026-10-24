import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from dashboard.backend.publication import load
from src.config.paths import REPO_ROOT


class PublicationDashboardTests(unittest.TestCase):
    def test_stored_update_has_exact_development_inputs_and_model_outputs(self):
        result=load(REPO_ROOT)
        self.assertTrue(result['available'])
        self.assertEqual(result['new_model_forwards'],12)
        self.assertEqual(len(result['cases']),2)
        for row in result['cases']:
            self.assertTrue(row['image_data_url'].startswith('data:image/png;base64,'))
        self.assertEqual([r['argmax_answer'] for r in result['models']['smol']['cases']],['Yes','Yes'])
        self.assertEqual([r['argmax_answer'] for r in result['models']['qwen']['cases']],['No','Yes'])
        self.assertEqual(result['screen']['status'],'FAILED')
        self.assertEqual(result['screen']['attempts'],1)
        self.assertEqual(result['screen']['completed_rows'],0)

    def test_missing_artifacts_are_explicit_not_synthetic_fallback(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT/'.cache') as name:
            self.assertEqual(load(Path(name)),{'available':False,'status':'NOT PACKAGED'})

    def test_changed_image_refused(self):
        real=Path.read_bytes
        def changed(path):return b'changed' if path.suffix=='.png' else real(path)
        with patch.object(Path,'read_bytes',changed):
            with self.assertRaisesRegex(ValueError,'hash mismatch'):load(REPO_ROOT)


if __name__=='__main__':unittest.main()
