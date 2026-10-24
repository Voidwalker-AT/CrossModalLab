import copy
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from src.benchmark.publication_factorial import build,shortcut_audit,feature
from src.config.paths import REPO_ROOT


class PublicationFactorialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir=REPO_ROOT/'.cache')
        cls.rows,cls.manifest=build(Path(cls.temp.name)/'benchmark',families=2)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_complete_factorial_support_and_oracles(self):
        self.assertEqual(len(self.rows),3072)
        self.assertEqual(Counter(r['ground_truth_label'] for r in self.rows),{'Yes':768,'No':768,'Undetermined':1536})
        self.assertTrue(all(r['oracle_agreement'] and r['split']=='development' for r in self.rows))
        for family in self.manifest['family_partitions']:
            rows=[r for r in self.rows if r['family_id']==family]
            self.assertEqual(len({tuple(r['factors'].values()) for r in rows}),1536)
            for key in ('roles','visual_edge','layout','query_first','choices_reversed','content_reversed'):
                for value in {r['factors'][key] for r in rows}:
                    counts=Counter(r['ground_truth_label'] for r in rows if r['factors'][key]==value)
                    self.assertEqual(counts['Yes'],counts['No'])
                    self.assertEqual(counts['Undetermined'],2*counts['Yes'])

    def test_query_counterfactuals_preserve_or_flip_exactly(self):
        indexed={(r['family_id'],tuple((k,v) for k,v in r['factors'].items() if k!='query_forward')):r for r in self.rows if r['factors']['query_forward']==0}
        reverse={'Yes':'No','No':'Yes','Undetermined':'Undetermined'}
        for row in self.rows:
            if not row['factors']['query_forward']:continue
            other=indexed[(row['family_id'],tuple((k,v) for k,v in row['factors'].items() if k!='query_forward'))]
            self.assertEqual(row['ground_truth_label'],reverse[other['ground_truth_label']])
            self.assertEqual(row['image_sha256'],other['image_sha256'])

    def test_baseline_ceilings_and_family_hash_separation(self):
        report=shortcut_audit(self.rows)
        for name,result in report['baselines'].items():
            target=2/3 if name in ('text_only_oracle_roles','image_only_oracle_roles') else 1/3
            self.assertAlmostEqual(result['in_sample_balanced_accuracy_ceiling'],target,places=10,msg=name)
            self.assertAlmostEqual(result['balanced_accuracy'],target,places=10,msg=name)
        self.assertAlmostEqual(report['oracle_minus_best_single_modality_ba'],1/3)
        self.assertEqual(report['cross_partition_image_hash_overlap'],0)
        self.assertEqual(report['cross_partition_prompt_hash_overlap'],0)

    def test_existing_output_and_family_leakage_refused(self):
        with self.assertRaisesRegex(ValueError,'fresh'):build(Path(self.temp.name)/'benchmark',families=2)
        bad=copy.deepcopy(self.rows)
        bad[-1]['family_id']=bad[0]['family_id']
        with self.assertRaisesRegex(ValueError,'family split'):shortcut_audit(bad)


if __name__=='__main__':unittest.main()
