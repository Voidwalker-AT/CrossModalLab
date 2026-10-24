"""CPU analysis of recorded publication pilot, dataset and local asset provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.benchmark.publication_factorial import shortcut_audit
from src.causal.publication_controls import eligible_pair
from src.training.run_state import file_hash,write_once


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT) or out.exists():raise ValueError('fresh checkout-local analysis output required')
    base=ROOT/'artifacts/publication/20261025'
    pilot=base/'pilot';plan=json.loads((pilot/'plan.json').read_text())
    execution=json.loads((pilot/'execution.json').read_text())
    if execution['status']!='PASSED' or execution['additional_forward_attempts']!=12:raise ValueError('pilot not complete')
    if file_hash(ROOT/'scripts/run_publication_isolated_pilot.py')!=plan['source_sha256']:raise ValueError('executed source changed')
    import importlib.util
    spec=importlib.util.spec_from_file_location('isolated_pilot',ROOT/'scripts/run_publication_isolated_pilot.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    result={'schema':'publication_analysis_v1','new_model_forwards':12,'historical_model_forwards':6,'optimizer_steps':0,
            'holdout_accessed':False,'causal_interventions':0,'models':{}}
    import psutil
    for tag in ('smol','qwen'):
        validation=runner.validate_worker_output(pilot/tag,tag,plan['cases'])
        details=json.loads((pilot/tag/'execution.json').read_text())
        scored=[json.loads(line) for line in (pilot/tag/'candidate_scores.jsonl').read_text().splitlines()]
        if psutil.pid_exists(details['pid']):raise RuntimeError('worker PID exists; independently inspect identity before asserting no orphan')
        for row in scored:
            row['length_normalized_answer']=['Yes','No','Undetermined'][max(range(3),key=lambda i:row['length_normalized_diagnostic'][i])]
            row['rank_flip']=row['length_normalized_answer']!=row['argmax_answer']
        result['models'][tag]={'validation':validation,'cases':scored,
                              'subset_correct':sum(r['argmax_answer']==r['ground_truth_label'] for r in scored),
                              'n':2,'families':1,'underdetermined_unsupported_definite_count':sum(r['ground_truth_label']=='Undetermined' and r['argmax_answer']!='Undetermined' for r in scored),
                              'causal_pair_screen':eligible_pair(*[r['summed_loglikelihoods'] for r in scored]),
                              'actual_worker_pid_absent':True,'load_sec':details['load_sec'],'worker_elapsed_sec':details['elapsed_sec'],
                              'telemetry':{k:v for k,v in details['telemetry'].items() if k!='samples'},
                              'parameter_comparisons':validation['parameter_comparisons']}
    old=[json.loads(line) for line in (ROOT/'artifacts/phase3/20261024_v2/pilot/candidate_scores.jsonl').read_text().splitlines()]
    result['qwen_scores_exactly_replicated']=all(a['summed_loglikelihoods']==b['summed_loglikelihoods'] for a,b in zip(old,result['models']['qwen']['cases']))
    declaration=json.loads((base/'benchmark/manifest.json').read_text())
    if file_hash(base/'benchmark/records.jsonl')!=declaration['records_sha256']:raise ValueError('benchmark identity mismatch')
    rows=[json.loads(line) for line in (base/'benchmark/records.jsonl').read_text().splitlines()]
    result['benchmark']=shortcut_audit(rows)
    for row in {r['image_file']:r for r in rows}.values():
        if file_hash(ROOT/row['image_file'])!=row['image_sha256']:raise ValueError('benchmark image changed')
    result['natural_assets']={}
    for dataset in ('coco','gqa'):
        directory=ROOT/f'datasets/real_world/{dataset}'
        manifest=json.loads((directory/f'manifests/{dataset}_30_manifest.json').read_text())
        checked=[]
        for row in manifest['records']:
            path=directory/'images'/row['filename']
            if file_hash(path)!=row['sha256']:raise ValueError('local natural image identity mismatch')
            checked.append({'sample_id':row['sample_id'],'source_id':row['source_id'],'image_sha256':row['sha256'],
                            'source_url':row['url'],'license_status':'INDIVIDUAL_IMAGE_RIGHTS_UNVERIFIED',
                            'annotation_status':'UNANNOTATED_NOT_EVALUATION','independent_human_verification':False})
        result['natural_assets'][dataset]={'local_hash_verified':len(checked),'records':checked,'images_redistributed':False,
                                          'license_note':'Historical dataset-level license string is not proof of individual image rights.'}
    neural=json.loads((ROOT/'dashboard/assets/neural_evidence_v1.json').read_text())
    for item in neural['sources']:
        if file_hash(ROOT/item['path'])!=item['sha256']:raise ValueError('historical neural source changed')
    result['neural_source_hashes_unchanged']=len(neural['sources'])
    result['interpretation']='One exposed development family, two fixed cases, candidate scoring only. No three-class BA/CI, generated-answer claim, causal mechanism, independent generalization or method benefit. New benchmark has no model outcomes.'
    out.parent.mkdir(parents=True,exist_ok=True)
    write_once(out,result)
    print(json.dumps({'status':'VERIFIED','models':{k:{key:v[key] for key in ('subset_correct','n','causal_pair_screen','parameter_comparisons')} for k,v in result['models'].items()},'qwen_exact_replication':result['qwen_scores_exactly_replicated'],'natural_assets':{k:v['local_hash_verified'] for k,v in result['natural_assets'].items()}}))


if __name__=='__main__':main()
