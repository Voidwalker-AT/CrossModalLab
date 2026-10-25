"""CPU-only paper evidence extraction. No model loads, forwards or optimizer steps.

Run in the existing research environment. Refuses to replace a completed audit.
The packaged copy also recomputes metrics using --verify-packaged-metrics.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,shutil,sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LABELS=('Yes','No','Undetermined')
RUN='artifacts/training/phase_2i_b/runs/56e277e421e442b08d33b107fc934494'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def recompute(rows):
    if not rows or len({r['sample_id']for r in rows})!=len(rows):raise ValueError('empty/duplicate evaluation')
    if set(r['ground_truth_label']for r in rows)!=set(LABELS):raise ValueError('all three classes required')
    if any(r.get('is_synthetic_demo') is True for r in rows):raise ValueError('demo data prohibited')
    cm={a:{b:0 for b in (*LABELS,'Invalid')}for a in LABELS}
    for r in rows:cm[r['ground_truth_label']][r['parsed_answer']if r['parsed_answer']in LABELS else 'Invalid']+=1
    supports={a:sum(cm[a].values())for a in LABELS};tp={a:cm[a][a]for a in LABELS}
    recalls={a:tp[a]/supports[a]for a in LABELS}
    f1={a:2*tp[a]/(supports[a]+sum(cm[b][a]for b in LABELS))if supports[a]+sum(cm[b][a]for b in LABELS)else 0 for a in LABELS}
    sufficient=[r for r in rows if r['ground_truth_label']!='Undetermined'];u=[r for r in rows if r['ground_truth_label']=='Undetermined']
    return dict(n=len(rows),families=len({r['family_id']for r in rows}),correct=sum(tp.values()),
        balanced_accuracy=sum(recalls.values())/3,macro_f1=sum(f1.values())/3,recall=recalls,tp=tp,support=supports,confusion_matrix=cm,
        semantic_invalid=sum(r['parsed_answer']not in LABELS for r in rows),
        exact_format=sum(r['raw_completion'].strip()in LABELS for r in rows),
        unsupported_definite=sum(r['parsed_answer']in ('Yes','No')for r in u),unsupported_denominator=len(u),
        incorrect_abstention=sum(r['parsed_answer']=='Undetermined'for r in sufficient),sufficient_denominator=len(sufficient))

def same_metrics(observed,expected):
    """Exact counts/identities; allow two ULPs for Python 3.11 vs 3.12+ sums."""
    if isinstance(observed,dict):
        return isinstance(expected,dict)and observed.keys()==expected.keys()and all(same_metrics(observed[k],expected[k])for k in observed)
    if isinstance(observed,float)and isinstance(expected,float):
        return math.isfinite(observed)and math.isfinite(expected)and abs(observed-expected)<=2*max(math.ulp(observed),math.ulp(expected))
    return observed==expected

def json_write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes((json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())

def tex(s):
    s=str(s)
    return ''.join({'&':r'\&','%':r'\%','_':r'\_','#':r'\#','{':r'\{','}':r'\}','\\':r'\textbackslash{}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}.get(c,c)for c in s)

def table(out,name,rows,caption):
    keys=list(rows[0]);path=out/'tables'/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.csv').open('w',encoding='utf-8',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    path.with_suffix('.md').write_bytes(('# '+caption+'\n\n| '+' | '.join(keys)+' |\n|'+'---|'*len(keys)+'\n'+''.join('| '+' | '.join(str(r[k]).replace('|','/')for k in keys)+' |\n'for r in rows)).encode())
    # Long, source-rich tables belong in supplementary material, not squeezed into a column.
    columns=''.join('p{'+str(round(0.94/len(keys),3))+r'\linewidth}'for _ in keys)
    text=r'\begin{table*}[t]'+'\n'+r'\centering\small'+'\n'+r'\caption{'+tex(caption)+'}\n'+r'\begin{tabular}{'+columns+'}\n'+r'\hline'+'\n'+' & '.join(tex(k)for k in keys)+r' \\ \hline'+'\n'
    text+='\n'.join(' & '.join(tex(r[k])for k in keys)+r' \\'for r in rows)+'\n'+r'\hline\end{tabular}'+'\n'+r'\end{table*}'+'\n'
    path.with_suffix('.tex').write_bytes(text.encode())

def build(out):
    sys.path.insert(0,str(ROOT))
    from dashboard.backend.artifacts import ArtifactStore
    from src.training.answer_parser import strict_parse_answer
    from src.symbolic.solver import SymbolicSolver
    from src.symbolic.relations import Premise,Query
    from src.benchmark.publication_factorial import shortcut_audit
    from scripts.build_neural_evidence import build as build_neural
    from scripts.run_phase3_cpu_audit import checkpoint_audit
    from src.training.differentiable_scoring import evidence_sufficiency_from_class_scores
    from src.config.paths import get_hf_home
    import torch
    assert not (out/'audit/recomputed.json').exists(),'previous audit preserved; choose a new output'
    sources={}
    def read(name):
        p=ROOT/name;sources[name]=sha(p);return json.loads(p.read_text(encoding='utf-8'))
    ce=read('artifacts/training/phase_2i_a/phase_2i_a_report_v1.json')
    run=read(RUN+'/run.json');base=read(RUN+'/baseline.json');pilot=read(RUN+'/evaluation_step_0005.json')
    store=ArtifactStore();cases=store.cases
    inputs={'frozen':ce['baseline_standard']['raw'],'prompt':base['abstention_prompt']['raw'],'ce5':ce['val_step_5']['raw'],'es5':pilot['standard']['raw'],'ce50':ce['val_final']['raw'],'es5prompt':pilot['abstention_prompt']['raw']}
    reference=set(cases);summaries={};raw=[]
    for condition,rows in inputs.items():
        assert {r['sample_id']for r in rows}==reference
        for r in rows:
            case=cases[r['sample_id']]
            assert r['parsed_answer']==strict_parse_answer(r['raw_completion'])['parsed_label']
            for key in ['family_id','ground_truth_label','prompt_sha256','image_sha256']:assert r[key]==case[key]
            assert sha(store.image(r['sample_id']))==r['image_sha256']
            assert hashlib.sha256(case['prompt'].encode()).hexdigest()==r['prompt_sha256']
            effective=case['prompt']
            if condition in ('prompt','es5prompt'):effective=effective.rstrip()+'\nYou may answer Undetermined when evidence is insufficient.\nAnswer directly with: Yes, No, or Undetermined.'
            raw.append({**r,'condition':condition,'stored_prompt_hash_scope':'BASE PROMPT, including for augmented conditions','effective_text_sha256_reconstructed':hashlib.sha256(effective.encode()).hexdigest(),'effective_prompt_reconstructed_from_evaluation_code':effective})
        summaries[condition]=recompute(rows)
    assert summaries['ce50']['semantic_invalid']==0 and summaries['ce50']['exact_format']==0
    assert summaries['ce50']['tp']=={'Yes':6,'No':6,'Undetermined':0}
    train_path=ROOT/RUN/'train_metrics_append.jsonl';sources[train_path.relative_to(ROOT).as_posix()]=sha(train_path)
    training=[json.loads(l)for l in train_path.read_text().splitlines()];assert [r['step']for r in training]==list(range(1,6))
    identity=run['identity'];assert [r['sample_id']for r in training]==[r['sample_id']for r in identity['schedule'][:5]]
    # Recompute the exact executed hinge for all five stored score triplets.
    for r in training:
        scores=r['class_ll'];gold=LABELS.index(r['label'])
        expected=max(0,1+max(scores[:2])-scores[2])if gold==2 else max(0,1+scores[2]-scores[gold])
        assert math.isclose(expected,r['es_loss'],abs_tol=3e-6)
        assert math.isclose(r['ce_loss']+.25*r['es_loss'],r['total_loss'],abs_tol=3e-6)
        observed=evidence_sufficiency_from_class_scores(torch.tensor([scores]),torch.tensor([gold]),sufficiency=torch.tensor([int(gold!=2)]))
        assert math.isclose(observed.item(),expected,abs_tol=3e-6)
        assert r['grad_l2']>0 and math.isfinite(r['grad_l2']) and r['parameter_acceptance']['passed']
    checkpoint=checkpoint_audit(ROOT/RUN,identity)
    # Header-only adapter shape accounting; no adapter loaded into a real model.
    from safetensors import safe_open
    with safe_open(str(ROOT/RUN/'checkpoints/step_0005/adapter_model.safetensors'),framework='pt',device='cpu')as file:
        adapter_shapes={k:file.get_slice(k).get_shape()for k in file.keys()}
    trainable=sum(math.prod(s)for s in adapter_shapes.values());assert trainable==802816
    env=read('paper/overleaf_evidence_20261025/audit/environment.json')
    from transformers import AutoTokenizer
    rev=identity['revision'];cache=get_hf_home();name='models--Qwen--Qwen3-VL-2B-Instruct'
    snapshot=next(p for p in [cache/name/'snapshots'/rev,cache/'hub'/name/'snapshots'/rev]if p.exists())
    tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
    candidate_ids=[tokenizer.encode(s,add_special_tokens=False)for s in (' Yes',' No',' Undetermined')]
    assert list(map(len,candidate_ids))==[1,1,2]
    benchmark_manifest=read('artifacts/publication/20261025/benchmark/manifest.json')
    bp=ROOT/'artifacts/publication/20261025/benchmark/records.jsonl';sources[bp.relative_to(ROOT).as_posix()]=sha(bp)
    assert sha(bp)==benchmark_manifest['records_sha256']
    benchmark=[json.loads(l)for l in bp.read_text().splitlines()]
    assert len(benchmark)==9216 and len({r['family_id']for r in benchmark})==6 and len({r['image_sha256']for r in benchmark})==72
    for r in {r['image_file']:r for r in benchmark}.values():assert sha(ROOT/r['image_file'])==r['image_sha256']
    shortcut=shortcut_audit(benchmark)
    neural=build_neural(ROOT) # Fresh raw-vector norm/shape/hash and probe-confusion audit.
    sources.update({r['path']:r['sha256']for r in neural['sources']})
    probe=[r for r in neural['probe']['points']if r['selector']=='last']
    assert [r['balanced_accuracy']for r in probe]==[.333333,.518519,.777778,.740741]
    # Separate descriptive family bootstrap from fixed score identities.
    import importlib.util
    spec=importlib.util.spec_from_file_location('dev_tables',ROOT/'paper/tables/build_development_tables.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    families=sorted({r['family_id']for r in inputs['es5']});deltas={f:sum(r['parsed_answer']==r['ground_truth_label']for r in inputs['es5']if r['family_id']==f)/3-sum(r['parsed_answer']==r['ground_truth_label']for r in inputs['ce5']if r['family_id']==f)/3 for f in families}
    interval=mod.bootstrap(deltas);assert math.isclose(interval['ci_low'],0) and math.isclose(interval['ci_high'],2/9)
    execution_path=next((ROOT/RUN).glob('execution_*.json'));execution=read(execution_path.relative_to(ROOT).as_posix())
    for step in range(1,6):
        audit=read(RUN+f'/parameter_audit_step_{step:04d}.json')
        assert audit['summary']=={'EXACT_NUMERICALLY_UNCHANGED':625,'NUMERICALLY_CHANGED':224}
    result={'status':'VERIFIED_CPU_RECOMPUTATION','conditions':summaries,'trainable_parameters':trainable,'adapter_tensors':len(adapter_shapes),'candidate_strings':[' Yes',' No',' Undetermined'],'candidate_token_ids':candidate_ids,'paired_step5_family_interval':interval,'new_model_forwards':0,'new_optimizer_steps':0,'replication_status':'BLOCKED_RAM','checkpoint_audit':checkpoint,'training_identity':identity,'training':training,'execution':execution,'probe':probe,'neural_validation':neural['validation'],'causal_audit':neural['causal_audit'],'benchmark':benchmark_manifest,'shortcut_audit':shortcut,'limits':['Historical probe intervals retained with B/seed/family metadata; not freshly resampled without saved per-example probe predictions.','Historical base-model invariance checked from saved per-tensor comparisons, not by reloading a real model.','No exact Overleaf source found; repository draft is audited.','Stored prompt hashes in augmented conditions identify the base prompt; effective-text hashes are reconstructed from code.']}
    json_write(out/'audit/recomputed.json',result);json_write(out/'audit/raw_predictions.json',raw)
    json_write(out/'audit/development_cases.json',list(cases.values()));json_write(out/'audit/neural_evidence_checked.json',neural)
    for c in cases.values():
        target=out/'figures/source_images'/str(c['sample_id'].replace('::','__')+'.png');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(store.image(c['sample_id']),target)
    for name in ['paper/manuscript/draft.md','paper/claims_registry.md','paper/evidence_registry.md','paper/prior_art_evidence_matrix.md','artifacts/publication/20261025/prior_art_extraction.json','artifacts/phase3/20261024_v2/audit.json','src/training/differentiable_scoring.py','src/training/phase_2i_b_protocol.py','src/training/sampling_and_masking.py','src/training/answer_parser.py','configs/phase_2i_a/ce_control_protocol_v1.json','configs/phase_2i_b/es_ablation_config_v1.json']:
        source=ROOT/name;sources[name]=sha(source);target=out/'audit/source_records'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    json_write(out/'manifest/source_hashes.json',sources)
    cfg=identity['optimizer'];lora=identity['lora']
    condition_sources={'frozen':'artifacts/training/phase_2i_a/phase_2i_a_report_v1.json#baseline_standard.raw','ce5':'artifacts/training/phase_2i_a/phase_2i_a_report_v1.json#val_step_5.raw','ce50':'artifacts/training/phase_2i_a/phase_2i_a_report_v1.json#val_final.raw','prompt':RUN+'/baseline.json#abstention_prompt.raw','es5':RUN+'/evaluation_step_0005.json#standard.raw','es5prompt':RUN+'/evaluation_step_0005.json#abstention_prompt.raw'}
    table(out,'a_model_conditions',[dict(condition=k,model='Qwen3-VL-2B',revision=rev,steps={'ce5':5,'es5':5,'ce50':50,'es5prompt':5}.get(k,0),cases=18,families=6,prompt='augmented'if 'prompt'in k else 'standard',source=condition_sources[k],status='VERIFIED historical')for k in summaries],'A. Pinned Qwen conditions; greedy generation, max 16 new tokens; not candidate argmax.')
    table(out,'b_dataset_units',[dict(dataset='Displayed legacy validation',records=18,families=6,images=18,unit='family, not variant',source='audit/development_cases.json',status='VERIFIED'),dict(dataset='Full-factorial development',records=9216,families=6,images=72,unit='family; 3 fit / 3 evaluation, all exposed',source='audit/recomputed.json#benchmark',status='VERIFIED')],'B. Dataset construction and dependence; no final holdout evaluated.')
    table(out,'c_behavioral_results',[dict(condition=k,correct=f"{s['correct']}/18",BA=f"{s['balanced_accuracy']:.6f}",macro_F1=f"{s['macro_f1']:.6f}",recall_Y_N_U='/'.join(str(s['tp'][l])+'/6'for l in LABELS),unsupported=f"{s['unsupported_definite']}/6",wrong_abstention=f"{s['incorrect_abstention']}/12",source='audit/raw_predictions.json#'+k,status='VERIFIED')for k,s in summaries.items()],'C. Recomputed historical behavior on identical development cases; no superiority claim.')
    table(out,'d_format_semantics',[dict(condition=k,semantic_invalid=f"{s['semantic_invalid']}/18",semantic_correct=f"{s['correct']}/18",exact_label=f"{s['exact_format']}/18",malformed=f"{18-s['exact_format']}/18",source='audit/raw_predictions.json#'+k,status='VERIFIED')for k,s in summaries.items()],'D. Prefix-parser correctness and exact-label compliance are separate endpoints.')
    table(out,'e_training_config',[dict(setting=k,value=v,scope='historical CE / CE+ES',source='audit/recomputed.json#training_identity',status='VERIFIED')for k,v in {'optimizer':'AdamW','lr':'0.0001','weight_decay':'0.01','betas':'0.9, 0.999','epsilon':'1e-8','LoRA rank':2,'LoRA alpha':4,'target modules':', '.join(lora['target_modules']),'trainable parameters':trainable,'quantization':'NF4 + double quantization','compute':'BF16; adapters/header dtype separately recorded','gradient accumulation':1,'batch size':1,'steps':'CE=50; CE+ES=5 paused','ES margin / coefficient':'1.0 / 0.25; CE+ES only'}.items()],'E. Recorded optimizer/adaptation configuration; updates are not epochs.')
    table(out,'f_five_step_telemetry',[dict(step=r['step'],sample=r['sample_id'],CE=r['ce_loss'],ES=r['es_loss'],total=r['total_loss'],grad_L2=r['grad_l2'],seconds=r['duration_sec'],peak_allocated_MiB=r['memory']['torch_max_allocated_mib'],peak_reserved_MiB=r['memory']['torch_max_reserved_mib'],source=RUN+'/train_metrics_append.jsonl',status='VERIFIED historical')for r in training],'F. Five observed updates on different examples; allocator peaks are not physical-device peaks.')
    table(out,'g_smol_probes',[dict(layer=r['layer'],selector=r['selector'],BA=r['balanced_accuracy'],CI_low=r['ci']['ci_low'],CI_high=r['ci']['ci_high'],validation='27 cases / 9 families',source=neural['probe']['source'],status='VERIFIED confusion BA; HISTORICAL CI')for r in probe],'G. Historical SmolVLM2 representation probes; development-selected, correlational, not causal.')
    table(out,'h_evidence_limits',[dict(evidence=k,cases=n,meaning=v,source=s,status=st)for k,n,v,s,st in [('Qwen generation',18,'Behavior only; development reuse','audit/raw_predictions.json','VERIFIED'),('CE+ES feasibility',5,'Five updates, not benefit or epochs','audit/recomputed.json#training','VERIFIED historical'),('Smol probes',27,'Accessibility, not causal use','audit/recomputed.json#probe','CORRELATIONAL'),('Intervention recovery',16,'10 interventions; 26 missing planned rows; control exceeds target','audit/recomputed.json#causal_audit','UNSUPPORTED mechanism'),('New replication',0,'RAM admission failed; zero attempts','audit/replication_admission.json','BLOCKED')]],'H. Evidence classification and claim boundaries.')
    print(json.dumps({'status':'PASS','conditions':len(summaries),'records':len(raw),'new_forwards':0,'trainable_parameters':trainable,'checkpoint_steps':6,'probe_points':len(probe)}))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'paper/overleaf_evidence_20261025');p.add_argument('--verify-packaged-metrics',action='store_true');args=p.parse_args();out=args.output.resolve()
    if args.verify_packaged_metrics:
        rows=json.loads((out/'audit/raw_predictions.json').read_text(encoding='utf-8'));expected=json.loads((out/'audit/recomputed.json').read_text(encoding='utf-8'))['conditions']
        assert same_metrics({k:recompute([r for r in rows if r['condition']==k])for k in expected},expected);print('PASS: packaged raw prediction metrics');return
    if not out.is_relative_to(ROOT):raise ValueError('output must remain in repository')
    denied=[];blocked={ROOT/p for p in ('configs/phase_1c/balanced_final_holdout_v1.jsonl','configs/phase_1c/final_holdout_family_ids.txt','datasets/cross_modal_contrast_v2/cross_modal_contrast_v2.jsonl','datasets/cross_modal_dependency_v1/cross_modal_dependency_v1.jsonl')}
    def guard(event,args):
        if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
            q=Path(os.fsdecode(args[0])).resolve()
            if q in blocked or q.is_relative_to(ROOT/'datasets/cross_modal_dependency_v1/images'):denied.append(str(q));raise PermissionError('Closed-data guard')
    sys.addaudithook(guard)
    try:build(out)
    finally:json_write(out/'audit/restricted_access_receipt.json',{'denied_opens':denied,'scope':'per-process Python open guard','model_forwards':0,'optimizer_steps':0})

if __name__=='__main__':main()
