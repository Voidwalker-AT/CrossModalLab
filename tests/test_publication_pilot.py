"""CPU checks of admission, evidence integrity and failure dispatch; no VLM."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.config.paths import REPO_ROOT

spec=importlib.util.spec_from_file_location('publication_pilot',REPO_ROOT/'scripts/run_publication_isolated_pilot.py')
pilot=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class IsolatedPilotTests(unittest.TestCase):
    def test_resource_gates(self):
        good={'host_available_mib':5000,'nvidia_total_mib':4096,'nvidia_used_mib':150}
        with patch.object(pilot.shutil,'disk_usage',return_value=type('Disk',(),{'free':10*1024**3})()):
            pilot.gate(good,loading=True)
            for field,value in [('host_available_mib',4095),('nvidia_used_mib',3901)]:
                bad={**good,field:value}
                with self.assertRaises(RuntimeError):pilot.gate(bad)
            with self.assertRaisesRegex(RuntimeError,'dedicated'):pilot.gate({**good,'nvidia_used_mib':700},loading=True)

    def test_fixed_cases_remain_development(self):
        rows=pilot.selected_cases()
        self.assertEqual([r['ground_truth_label'] for r in rows],['No','Undetermined'])
        self.assertEqual(rows[0]['image_sha256'],rows[1]['image_sha256'])

    def test_existing_admission_refuses_reset(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT/'.cache') as name,patch.object(pilot.sys,'argv',['pilot']):
            with patch.object(pilot,'OUTPUT',Path(name)):
                with self.assertRaisesRegex(RuntimeError,'reset'):pilot.main()

    def test_worker_integrity_rejects_missing_rows_and_parameter_changes(self):
        rows=pilot.selected_cases()
        with tempfile.TemporaryDirectory(dir=REPO_ROOT/'.cache') as name:
            out=Path(name);ledger=out/'forward_ledger';ledger.mkdir()
            execution={'status':'PASSED','forward_attempts':6,'optimizer_steps':0,'backward_passes':0,'causal_patching':False,'holdout_accessed':False,
                       'parameter_audit':{'parameters':[{'passed':True}]},'telemetry':{'sampling_errors':[],'minimum_sampled_host_available_mib':5000,'maximum_sampled_nvidia_used_mib':3000}}
            scored=[]
            for index,row in enumerate(rows):
                scored.append({**row,'model':'smol','summed_loglikelihoods':[-3.,-1.,-2.],'argmax_answer':'No','is_generated_completion':False})
                for candidate in range(3):
                    attempt=index*3+candidate+1
                    (ledger/f'forward_{attempt:03d}.json').write_text(json.dumps({'attempt':attempt,'model':'smol','sample_id':row['sample_id'],'candidate_index':candidate}))
            def save():
                (out/'execution.json').write_text(json.dumps(execution))
                (out/'candidate_scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
            save();self.assertEqual(pilot.validate_worker_output(out,'smol',rows)['valid_scored_rows'],2)
            execution['parameter_audit']['parameters'][0]['passed']=False;save()
            with self.assertRaisesRegex(ValueError,'invariance'):pilot.validate_worker_output(out,'smol',rows)
            execution['parameter_audit']['parameters'][0]['passed']=True
            scored.pop();save()
            with self.assertRaisesRegex(ValueError,'incomplete'):pilot.validate_worker_output(out,'smol',rows)


if __name__=='__main__':unittest.main()
