"""Read-only supplemental historical checks; no model execution or new fitting."""
import json,math,sys,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.evaluation.research_audit import audit_records,lookup_audit,role_key,verify_partial_pilot
from scripts.build_overleaf_evidence import recompute,json_write,RUN

def main():
    out=ROOT/'paper/overleaf_evidence_20261025';hashes={}
    def load(name,lines=False):
        path=ROOT/name;hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest();s=path.read_text(encoding='utf-8');return [json.loads(l)for l in s.splitlines()]if lines else json.loads(s)
    selected=load('artifacts/phase3/20261024_v2/selected_development_records.jsonl',True)
    train=[r for r in selected if r['split']=='train'];val=[r for r in selected if r['split']=='validation']
    result={'selected_development':audit_records(selected),'train_label_counts':dict(Counter(r['ground_truth_label']for r in train)),'legacy_shortcuts':{m:lookup_audit(train,val,m)for m in ('visual','textual')}}
    result['concrete_role_strata']={m:{str(key):dict(Counter(r['ground_truth_label']for r in selected if role_key(r,m)==key))for key in set(role_key(r,m)for r in selected)}for m in ('visual','textual')}
    factorial=load('artifacts/phase3/20261024_v2/factorial_development/records.jsonl',True);result['earlier_factorial']=audit_records(factorial);assert result['earlier_factorial']['cases']==96
    paired=load('artifacts/paired/phase_2c_c/paired_evaluations_v1.jsonl',True)
    result['historical_paired']={tag:recompute([r for r in paired if r['model_tag']==tag])for tag in ('smolvlm2','qwen3vl2')}
    assert result['historical_paired']['smolvlm2']['correct']==8 and result['historical_paired']['qwen3vl2']['correct']==10
    old=load('artifacts/phase3/20261024_v2/pilot/candidate_scores.jsonl',True)
    result['partial_qwen_pilot']=verify_partial_pilot(ROOT/'artifacts/phase3/20261024_v2/pilot')
    result['isolated_pilot']={}
    for model in ['smol','qwen']:
        scores=load(f'artifacts/publication/20261025/pilot/{model}/candidate_scores.jsonl',True)
        execution=load(f'artifacts/publication/20261025/pilot/{model}/execution.json')
        assert execution['forward_attempts']==6
        for r in scores:
            vals=r['summed_loglikelihoods'];assert all(math.isfinite(x)for x in vals)
            assert r['argmax_answer']==['Yes','No','Undetermined'][max(range(3),key=lambda i:vals[i])]
        result['isolated_pilot'][model]={'scores':scores,'recorded_execution':execution}
    result['qwen_historical_scores_exact_match']=all(a['summed_loglikelihoods']==b['summed_loglikelihoods']for a,b in zip(old,result['isolated_pilot']['qwen']['scores']))
    assert result['qwen_historical_scores_exact_match']
    screen=load('artifacts/publication/20261025/screen108/execution.json');result['failed_screen']=screen
    run=json.loads((out/'audit/recomputed.json').read_text());rows=run['training'];samples=run['execution']['resource_samples']
    result['training_aggregates']={'step_seconds':sum(r['duration_sec']for r in rows),'run_elapsed_seconds':run['execution']['elapsed_sec'],'max_allocator_mib':max(r['memory']['torch_max_allocated_mib']for r in rows),'max_reserved_mib':max(r['memory']['torch_max_reserved_mib']for r in rows),'sampled_physical_max_mib':max(r['nvidia_smi_used_mib']for r in samples),'sampled_available_ram_min_mib':min(r['host_available_mib']for r in samples)}
    result['frozen_parameter_records']=[]
    for step in range(1,6):
        audit=load(RUN+f'/parameter_audit_step_{step:04d}.json')
        # The global comparison 'passed' is false when expected adapter changes occur.
        records=next(v for v in audit.values()if isinstance(v,list)and v and isinstance(v[0],dict)and 'requires_grad_before'in v[0])
        frozen=[r for r in records if not r['requires_grad_before']];lora=[r for r in records if r['requires_grad_before']]
        assert len(frozen)==625 and all(r['numerically_equal']and r['finite']and not r['unexpected_frozen_gradient']for r in frozen)
        assert len(lora)==224 and all(r['finite']for r in lora)
        result['frozen_parameter_records'].append({'step':step,'frozen_exactly_equal':len(frozen),'adapter_finite':len(lora)})
    ce50=[r for r in json.loads((out/'audit/raw_predictions.json').read_text())if r['condition']=='ce50']
    result['ce50_suffix_audit']={'records':18,'with_replacement_character':sum('\ufffd'in r['raw_completion']for r in ce50),'replacement_counts':dict(Counter(r['raw_completion'].count('\ufffd')for r in ce50)),'semantic_invalid':run['conditions']['ce50']['semantic_invalid'],'exact_label_compliant':run['conditions']['ce50']['exact_format']}
    json_write(out/'audit/supplementary_recomputation.json',result);json_write(out/'manifest/supplementary_source_hashes.json',hashes)
    print(json.dumps({'status':'PASS','training_aggregates':result['training_aggregates'],'ce50_suffix_audit':result['ce50_suffix_audit'],'new_model_forwards':0,'new_optimizer_steps':0}))

if __name__=='__main__':main()
