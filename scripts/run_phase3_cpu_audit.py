"""Audit allowlisted development evidence without opening mixed/final manifests.

Reconstruct only scheduled train + displayed validation families from the pinned
generator, then require their exact historical prompt/image/label identities.
All output is versioned; no research source is overwritten. Optional checkpoint
checks read trusted local binaries with weights_only=True on CPU, zero steps.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.config.paths import REPO_ROOT
from src.training.run_state import digest, file_hash, RunStore
from src.evaluation.research_audit import audit_records, lookup_audit, role_ambiguity
from src.benchmark.contrast_v2 import plan_families, build_family_records
from src.benchmark.factorial_development import build_factorial

RUN='artifacts/training/phase_2i_b/runs/56e277e421e442b08d33b107fc934494'


def reconstruct_known(out,identity):
    selected={r['sample_id']:r for r in identity['schedule']+identity['validation']}
    families={r['family_id'] for r in selected.values()}
    plans=[p for p in plan_families(seed=20261031) if p.split in {'train','validation'} and p.family_id in families]
    if len(plans)!=len(families):raise ValueError('selected development family missing from pinned plan')
    rows=[]
    for plan in plans:
        for row in build_family_records(plan,out/'reconstructed_images'):
            if row['sample_id'] not in selected:continue
            historical=selected[row['sample_id']]
            for key in ('family_id','ground_truth_label','prompt_sha256','image_sha256'):
                if row[key]!=historical[key]:raise ValueError('reconstruction disagrees with historical identity: '+key)
            original=REPO_ROOT/'datasets/cross_modal_contrast_v2/images'/f"{plan.family_id}__{row['variant_id']}.png"
            if file_hash(original)!=row['image_sha256']:raise ValueError('original selected image changed')
            row['image_file']=original.relative_to(REPO_ROOT).as_posix()
            rows.append(row)
    if len(rows)!=len(selected):raise ValueError('selected record coverage mismatch')
    return rows


def checkpoint_audit(run,identity):
    import torch
    from src.training.run_state import validate_optimizer, validate_rng
    before={p.relative_to(run).as_posix():file_hash(p) for p in run.rglob('*') if p.is_file()}
    results=[]
    with RunStore(run,identity) as store:
        inspected=store.inspect()
        if inspected['step']!=5:raise ValueError('saved pilot boundary unexpectedly changed')
        for step,path in enumerate(store.checkpoints):
            blob=torch.load(path/'optimizer.pt',map_location='cpu',weights_only=True)
            params=[torch.nn.Parameter(torch.empty(x['shape'],dtype=getattr(torch,x['dtype'].split('.')[-1])))for x in blob['layout']]
            opt=torch.optim.AdamW(params,lr=identity['optimizer']['learning_rate'])
            validate_optimizer(blob,opt,blob['layout'],step)
            rng=torch.load(path/'rng.pt',map_location='cpu',weights_only=True);validate_rng(rng)
            opt.load_state_dict(blob['state'])
            for key,slot in blob['state']['state'].items():
                if not all(torch.equal(v,opt.state_dict()['state'][key][k]) for k,v in slot.items()):raise ValueError('CPU optimizer reload mismatch')
            results.append({'step':step,'components_sha256_verified':True,'optimizer_state_valid':True,
                            'cpu_optimizer_moments_reloaded_exactly':True,'rng_state_valid':True,
                            'cuda_rng_states':len(rng['cuda']),'trainable_tensors':len(blob['layout'])})
    after={p.relative_to(run).as_posix():file_hash(p) for p in run.rglob('*') if p.is_file()}
    if before!=after:raise ValueError('historical run mutated during audit')
    return {'status':'PASS','checkpoints':results,'historical_files_unchanged':True,'optimizer_steps':0,
            'restoration_limit':'CPU state reload/validation only. No fresh real-model adapter or CUDA RNG restoration; prior GPU restoration remains historical evidence.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoints',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT) or out.exists():parser.error('fresh checkout-local output required')
    out.mkdir(parents=True)
    run=ROOT/RUN;run_manifest=json.loads((run/'run.json').read_text(encoding='utf-8'));identity=run_manifest['identity']
    if digest(identity)!=run_manifest['identity_hash']:raise ValueError('saved identity hash mismatch')
    rows=reconstruct_known(out,identity)
    train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation']
    report={'schema':'phase3_cpu_research_audit_v1','audit_scope':'50 frozen scheduled train examples + 18 displayed validation cases; original mixed manifests and final-holdout records not opened',
            'source_run_identity':run_manifest['identity_hash'],'selected':audit_records(rows),
            'train':audit_records(train),'validation':audit_records(val),
            'text_role_lookup':lookup_audit(train,val,'textual'),'image_role_lookup':lookup_audit(train,val,'visual'),
            'text_role_ambiguity':role_ambiguity(val,'textual'),'image_role_ambiguity':role_ambiguity(val,'visual'),
            'entity_overlap_train_validation':sorted(set(e for r in train for e in r['entities'].values()) & set(e for r in val for e in r['entities'].values())),
            'cross_split_identical_image_hashes':sorted({r['image_sha256']for r in train}&{r['image_sha256']for r in val}),
            'cross_split_identical_prompt_hashes':sorted({r['prompt_sha256']for r in train}&{r['prompt_sha256']for r in val}),
            'unordered_entity_set_overlap':sorted([sorted(t)for t in {tuple(sorted(r['entities'].values()))for r in train}&{tuple(sorted(r['entities'].values()))for r in val}]),
            'source_hash_checks':{name:file_hash(ROOT/name)==expected for name,expected in identity['source_sha256'].items()},
            'config_hash_valid':file_hash(ROOT/'configs/phase_2i_b/es_ablation_config_v1.json')==identity['config_sha256'],
            'schedule_hash_valid':digest(identity['schedule'])==identity['schedule_sha256'],
            'dataset_combined_hash_status':'NOT RECOMPUTED: mixed train/validation/test source kept closed; historical identity retained; selected records independently matched',
            'holdout_records_opened':False,'new_model_forwards':0,'new_optimizer_steps':0}
    if not all(report['source_hash_checks'].values()) or not report['config_hash_valid'] or not report['schedule_hash_valid']:raise ValueError('frozen run source/config/schedule identity changed')
    for step in range(1,6):
        audit=json.loads((run/f'parameter_audit_step_{step:04d}.json').read_text(encoding='utf-8'))
        if not audit['acceptance']['passed']:raise ValueError('historical frozen integrity audit failed')
    report['frozen_parameter_integrity']='All five stored exact parameter audits pass; no new model load or fresh parameter comparison claimed'
    if args.checkpoints:report['checkpoint_audit']=checkpoint_audit(run,identity)
    journal=[json.loads(line)for line in (run/'train_metrics_append.jsonl').read_text().splitlines()]
    report['pilot_step_seconds']=[r['duration_sec']for r in journal]
    report['remaining_45_train_steps_seconds_linear_projection']=45*sum(report['pilot_step_seconds'])/5
    (out/'selected_development_records.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n'for r in rows),encoding='utf-8')
    fresh,meta=build_factorial(out/'factorial_development')
    report['factorial']=audit_records(fresh)
    report['factorial_text_ambiguity']=role_ambiguity(fresh,'textual')
    report['factorial_image_ambiguity']=role_ambiguity(fresh,'visual')
    family_ids=sorted({r['family_id']for r in fresh});first=set(family_ids[:6])
    report['factorial_text_lookup']=lookup_audit([r for r in fresh if r['family_id']in first],[r for r in fresh if r['family_id']not in first],'textual')
    report['factorial_image_lookup']=lookup_audit([r for r in fresh if r['family_id']in first],[r for r in fresh if r['family_id']not in first],'visual')
    paired=[json.loads(line)for line in (ROOT/'artifacts/paired/phase_2c_c/paired_evaluations_v1.jsonl').read_text().splitlines()]
    report['historical_cross_model']={}
    for tag in sorted({r['model_tag']for r in paired}):
        selected=[r for r in paired if r['model_tag']==tag]
        report['historical_cross_model'][tag]={'cases':len(selected),'correct':sum(r['parsed_answer']==r['ground_truth_label']for r in selected),
            'unsupported_definite_forks':sum(r['ground_truth_label']=='Undetermined' and r['parsed_answer']in {'Yes','No'}for r in selected),
            'strict_format_compliant':sum(r['raw_completion'].strip()in {'Yes','No','Undetermined'}for r in selected),
            'mean_generation_latency_sec':sum(r['latency_sec']for r in selected)/len(selected)}
    by_model={tag:{r['sample_id']:r for r in paired if r['model_tag']==tag}for tag in report['historical_cross_model']}
    tags=list(by_model);shared=set(by_model[tags[0]])&set(by_model[tags[1]])
    report['historical_cross_model']['paired_identity_valid']=len(shared)==18 and all(all(by_model[tags[0]][sid][k]==by_model[tags[1]][sid][k]for k in ('image_sha256','prompt_sha256','ground_truth_label'))for sid in shared)
    manifest={'schema':'phase3_evaluation_partitions_v1','TRAIN':{'status':'HISTORICAL_SCHEDULED_TRAIN_NOT_EVALUATION','records':identity['schedule']},
              'DEVELOPMENT':{'status':'REPEATEDLY_INSPECTED','records':identity['validation'],'new_factorial_manifest':(out/'factorial_development/manifest.json').relative_to(ROOT).as_posix()},
              'LOCKED_FINAL_HOLDOUT':{'status':'CLOSED_REQUIRES_SEPARATE_APPROVAL','reference':'configs/phase_1c/frozen_split_manifest_v1.json','record_files_opened':False,
                                      'historical_checksum_registry':'configs/phase_1c/checksums_v1.json','independence_currently_certified':False},
              'mixed_manifest_policy':'Do not use existing build_plan() for this phase: it parses every source line before filtering test. Use the selected extract and verify original selected hashes.'}
    (out/'evaluation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k]for k in ('selected','text_role_lookup','image_role_lookup','factorial','factorial_text_lookup','factorial_image_lookup','historical_cross_model')},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
