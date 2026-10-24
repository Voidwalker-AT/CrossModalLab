import copy
import tempfile
from pathlib import Path
import unittest
from src.benchmark.factorial_development import build_factorial
from src.config.paths import REPO_ROOT
from src.evaluation.research_audit import audit_records, lookup_audit, verified_label, role_ambiguity, verify_partial_pilot
from src.evaluation.forward_budget import ForwardBudget


class Phase3AuditTests(unittest.TestCase):
    def test_recorded_failure_remains_partial_not_cross_model_success(self):
        path=REPO_ROOT/'artifacts/phase3/20261024_v2/pilot'
        if not path.exists():self.skipTest('bounded pilot artifacts unavailable')
        result=verify_partial_pilot(path)
        self.assertEqual(result['forward_attempts'],6)
        self.assertEqual(result['valid_scored_rows'],2)
        self.assertEqual(result['execution_status'],'FAILED')
        self.assertFalse(result['complete_cross_model'])

    def test_budget_persists_attempts_and_refuses_extra_forward(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT/'.cache') as temp:
            path=Path(temp)/'ledger';budget=ForwardBudget(path,limit=2)
            budget.consume(sample_id='fixture');budget.consume(sample_id='fixture')
            with self.assertRaisesRegex(RuntimeError,'exhausted'):budget.consume(sample_id='never_executed')
            self.assertEqual(len(list(path.glob('forward_*.json'))),2)
            with self.assertRaisesRegex(ValueError,'reset'):ForwardBudget(path)
            with self.assertRaises(ValueError):ForwardBudget(Path(temp)/'too_many',limit=13)

    def test_factorial_counterbalances_roles_and_shares_exact_images(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT/'.cache') as temp:
            rows,_=build_factorial(Path(temp)/'new',families=2)
            audit=audit_records(rows)
            self.assertEqual(audit['labels'],{'Yes':4,'No':4,'Undetermined':8})
            families=sorted({r['family_id']for r in rows})
            train=[r for r in rows if r['family_id']==families[0]]
            val=[r for r in rows if r['family_id']==families[1]]
            # A single premise can exclude one label without determining truth.
            # Chance-level BA is not guaranteed even after full counterbalancing.
            self.assertAlmostEqual(lookup_audit(train,val,'textual')['balanced_accuracy'],2/3)
            self.assertAlmostEqual(lookup_audit(train,val,'visual')['balanced_accuracy'],2/3)
            for modality in ('textual','visual'):
                self.assertEqual(role_ambiguity(rows,modality)['deterministic_groups'],0)
                self.assertEqual(role_ambiguity(rows,modality)['maximum_label_purity'],.5)
            for family in families:
                selected=[r for r in rows if r['family_id']==family]
                self.assertEqual(len({r['image_sha256']for r in selected}),2)
                a=next(r for r in selected if r['sample_id'].endswith('v1t1q0'))
                b=next(r for r in selected if r['sample_id'].endswith('v1t0q0'))
                self.assertEqual((a['ground_truth_label'],b['ground_truth_label']),('No','Undetermined'))
                self.assertEqual(a['image_sha256'],b['image_sha256'])
                self.assertEqual(a['query'],b['query'])

    def test_restricted_split_and_wrong_gold_refused(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT/'.cache') as temp:
            rows,_=build_factorial(Path(temp)/'new',families=2)
            for split in ('test','final_holdout','unknown'):
                row=copy.deepcopy(rows[0]);row['split']=split
                with self.assertRaisesRegex(ValueError,'restricted'):verified_label(row)
            row=copy.deepcopy(rows[0]);row['ground_truth_label']='No'
            with self.assertRaisesRegex(ValueError,'mismatch'):verified_label(row)
            row=copy.deepcopy(rows[0]);row['is_synthetic_demo']=True
            with self.assertRaisesRegex(ValueError,'demo'):verified_label(row)

    def test_duplicates_and_group_leakage_refused(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT/'.cache') as temp:
            rows,_=build_factorial(Path(temp)/'new',families=2)
            with self.assertRaisesRegex(ValueError,'duplicate'):audit_records(rows+[rows[0]])
            with self.assertRaisesRegex(ValueError,'family leakage'):lookup_audit(rows,rows,'textual')
            row=copy.deepcopy(rows[0]);row['split']='train'
            with self.assertRaisesRegex(ValueError,'crosses'):audit_records([row]+rows[1:])

    def test_generation_preserves_existing_output(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT/'.cache') as temp:
            target=Path(temp)/'new';build_factorial(target,families=2)
            before=(target/'records.jsonl').read_bytes()
            with self.assertRaisesRegex(ValueError,'preserved'):build_factorial(target,families=2)
            self.assertEqual((target/'records.jsonl').read_bytes(),before)


if __name__=='__main__':unittest.main()
