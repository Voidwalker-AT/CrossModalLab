"""One development-only pilot: 2 models x 2 cases x 3 candidates = 12 forwards.

No generation, optimizer, gradient, causal patch or automatic retry. Candidate
argmax answers are explicitly separate from historical generated completions.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.config.paths import REPO_ROOT, get_hf_home, get_smolvlm2_snapshot_dir
from src.evaluation.forward_budget import ForwardBudget
from src.evaluation.research_audit import verified_label
from src.training.run_state import file_hash, write_once, append_bytes, canonical


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();manifest=args.manifest.resolve()
    if not out.is_relative_to(ROOT) or out.exists() or not manifest.is_relative_to(ROOT):parser.error('fresh checkout-local pilot output and development manifest required')
    declaration=json.loads(manifest.read_text())
    if declaration.get('status')!='NEW_DEVELOPMENT_NOT_CONFIRMATORY':raise ValueError('pilot requires explicit development identity')
    records_path=manifest.parent/'records.jsonl'
    if file_hash(records_path)!=declaration['records_sha256']:raise ValueError('development dataset hash mismatch')
    records=[json.loads(line)for line in records_path.read_text().splitlines()]
    family=sorted({r['family_id']for r in records})[0]
    selected=[next(r for r in records if r['family_id']==family and r['sample_id'].endswith(suffix))for suffix in ('v1t1q0','v1t0q0')]
    if [verified_label(r)for r in selected]!=['No','Undetermined'] or len({r['image_sha256']for r in selected})!=1 or selected[0]['query']!=selected[1]['query']:
        raise ValueError('matched contradicted/fork pair identity failed')
    for row in selected:
        image=(ROOT/row['image_file']).resolve()
        if not image.is_relative_to(ROOT) or file_hash(image)!=row['image_sha256'] or hashlib.sha256(row['prompt'].encode()).hexdigest()!=row['prompt_sha256']:
            raise ValueError('selected image/prompt identity failed')
    out.mkdir(parents=True);budget=ForwardBudget(out/'forward_ledger');unsafe=threading.Event()
    import torch
    import psutil
    from PIL import Image
    from src.models.qwen3vl2_loader import load_qwen3vl2_nf4, QWEN_MODEL_ID, QWEN_REVISION
    from src.models.model_loader import load_pinned_smolvlm2
    from src.training.differentiable_scoring import candidate_class_loglikelihoods, encode_candidates, CANDIDATES
    from src.training.scorer_verification import ParameterAudit, run_observed, sample_resources
    def sample():
        snapshot=sample_resources()
        if snapshot.get('host_available_mib') is None or snapshot['host_available_mib']<4096 or snapshot.get('nvidia_smi_used_mib') is None or snapshot['nvidia_smi_used_mib']>3900:unsafe.set()
        return snapshot
    deadline=time.perf_counter()+600
    def gate(*,loading=False):
        if time.perf_counter()>deadline:raise RuntimeError('ten-minute wall budget exhausted; no automatic extension')
        snap=sample()
        if unsafe.is_set():raise RuntimeError('resource gate: RAM below 4 GiB or physical GPU use above 3900 MiB; stop without retry')
        if loading and (not torch.cuda.is_available() or snap['nvidia_smi_total_mib']-snap['nvidia_smi_used_mib']<3500):raise RuntimeError('CUDA unavailable or insufficient free dedicated VRAM before loading')
        if shutil.disk_usage(ROOT).free<4*1024**3:raise RuntimeError('insufficient disk reserve for exact parameter audit')
        return snap
    revisions=[(QWEN_MODEL_ID,QWEN_REVISION),('HuggingFaceTB/SmolVLM2-2.2B-Instruct','482adb537c021c86670beed01cd58990d01e72e4')]
    hf=get_hf_home();snapshots=[]
    for model_id,revision in revisions:
        candidates=[hf/f"models--{model_id.replace('/','--')}"/'snapshots'/revision,hf/'hub'/f"models--{model_id.replace('/','--')}"/'snapshots'/revision]
        snapshot=next((p for p in candidates if p.is_dir()),None)
        if snapshot is None:raise RuntimeError('pinned local snapshot absent; downloads forbidden')
        snapshots.append(str(snapshot))
    report={'schema':'phase3_bounded_candidate_pilot_v1','created_utc':datetime.now(timezone.utc).isoformat(),'new_optimizer_steps':0,'causal_patching':False,'holdout_accessed':False,
            'source_sha256':{name:file_hash(ROOT/name)for name in ('scripts/run_phase3_bounded_pilot.py','src/evaluation/forward_budget.py','src/training/differentiable_scoring.py','src/models/model_loader.py','src/models/qwen3vl2_loader.py')},
            'protocol':'2 fixed development cases x 2 pinned frozen models x 3 full-sequence candidate forwards; batch one; no generation',
            'selected_cases':selected,'dataset_manifest_sha256':file_hash(manifest),'local_snapshots':snapshots,
            'seed':2026102403,'candidates':list(CANDIDATES),'inference_mode':True,'models':[],
            'versions':{name:importlib.metadata.version(name)for name in ('torch','transformers','bitsandbytes','psutil')},
            'runtime_estimate':'Historical Smol forward mean 16.20s, range 4.65–76.94s: six forwards ~97s central/~462s pessimistic; Qwen historical generation ~0.43s/case is not a teacher-forced estimate. Allow ten minutes plus loads; no automatic extension.'}
    write_once(out/'preregistered_plan.json',report)
    torch.manual_seed(report['seed']);loaded={};context={}
    def action():
        for tag,loader in [('qwen',load_qwen3vl2_nf4),('smol',load_pinned_smolvlm2)]:
            report['models'].append(entry:={'tag':tag,'preload':gate(loading=True)})
            torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            processor,model,meta=loader();loaded.update(processor=processor,model=model)
            entry['load_sec']=time.perf_counter()-start;entry['identity']=meta
            model.eval();model.requires_grad_(False)
            def before_forward(_module,_inputs):
                gate();budget.consume(model=tag,candidate_index=budget.used%3,**context)
                print(f'FORWARD {budget.used}/12 {tag} {context["sample_id"]}',flush=True)
            handle=model.register_forward_pre_hook(before_forward)
            try:
                with ParameterAudit(ROOT/'.cache/recovery-temp') as audit:
                    audit.capture(model)
                    token_ids=encode_candidates(processor.tokenizer)
                    for row in selected:
                        gate();context.clear();context.update(sample_id=row['sample_id'])
                        image=Image.open(ROOT/row['image_file']).convert('RGB')
                        messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':row['prompt']}]}]
                        inputs=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt').to('cuda:0')
                        started=time.perf_counter()
                        with torch.inference_mode():scores=candidate_class_loglikelihoods(model,inputs,token_ids)
                        torch.cuda.synchronize();values=scores[0].cpu().tolist()
                        if not all(__import__('math').isfinite(v)for v in values):raise ValueError('non-finite candidate scores')
                        result={'model':tag,'model_id':meta['model_id'],'revision':meta['revision'],'sample_id':row['sample_id'],'family_id':family,
                                'ground_truth_label':row['ground_truth_label'],'image_sha256':row['image_sha256'],'prompt_sha256':row['prompt_sha256'],
                                'candidate_strings':list(CANDIDATES),'candidate_token_ids':token_ids,'summed_loglikelihoods':values,
                                'argmax_answer':CANDIDATES[max(range(3),key=lambda i:values[i])].strip(),
                                'length_normalized_diagnostic':[v/len(ids)for v,ids in zip(values,token_ids)],
                                'is_generated_completion':False,'prompt_token_ids':inputs['input_ids'][0].cpu().tolist(),
                                'input_shapes':{key:list(value.shape)for key,value in inputs.items()if hasattr(value,'shape')},
                                'latency_sec':time.perf_counter()-started,'memory':sample()}
                        append_bytes(out/'candidate_scores.jsonl',canonical(result)+b'\n')
                        print(json.dumps({k:result[k]for k in ('model','sample_id','summed_loglikelihoods','argmax_answer','latency_sec')}),flush=True)
                        del inputs,scores;image.close();gate()
                    entry['parameter_audit']=audit.compare(model)
                    if not all(r['passed']for r in entry['parameter_audit']['parameters']):raise ValueError('parameter invariance audit failed')
            finally:handle.remove()
            del model,processor;loaded.clear();gc.collect();torch.cuda.empty_cache()
        if budget.used!=12:raise ValueError('unexpected forward accounting')
    def cleanup():
        loaded.clear();gc.collect();torch.cuda.empty_cache()
    code=run_observed(report,action,cleanup,sampler=sample,interval=.5)
    report['forward_attempts']=budget.used
    if unsafe.is_set():report['status']='FAILED';report['safety_gate_triggered']=True;code=1
    write_once(out/'execution.json',report)
    print(json.dumps({'status':report['status'],'forward_attempts':budget.used,'report':str(out/'execution.json')}),flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
