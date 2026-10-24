"""One explicitly approved 108-forward development screen, in two sequential processes.

Fixed followup_108_plan.json development inputs; no training, generation or patches.
An existing output is never resumed or reset. Failed attempts spend the budget.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.run_state import write_once, append_bytes, canonical, file_hash
from src.evaluation.research_audit import verified_label
from src.evaluation.screen_budget import ScreenBudget

OUTPUT = ROOT / 'artifacts/publication/20261025/screen108'
SOURCE = ROOT / 'artifacts/publication/20261025/followup_108_plan.json'
FROZEN_PLAN_SHA256 = 'f0377f7932b42439003d0fa646db982a2da75f0ad46e05039f9d05b3781d6bb7'


def resources():
    import psutil
    values = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
                                     '--format=csv,noheader,nounits'], text=True, timeout=10)
    if len(values.strip().splitlines()) != 1:
        raise RuntimeError('pilot requires one identified GPU')
    total, used, free = map(int, values.strip().split(','))
    return {'host_available_mib': psutil.virtual_memory().available / 1024**2,
            'nvidia_total_mib': total, 'nvidia_used_mib': used, 'nvidia_free_mib':free}


def gate(snapshot, *, loading=False):
    if snapshot['host_available_mib'] < 4096 or snapshot['nvidia_used_mib'] > 3900:
        raise RuntimeError('physical RAM/VRAM safety gate failed')
    if loading and snapshot['nvidia_free_mib'] < 3500:
        raise RuntimeError('insufficient free dedicated VRAM before worker')
    if shutil.disk_usage(ROOT).free < 4 * 1024**3:
        raise RuntimeError('disk safety reserve failed')


def selected_cases():
    if file_hash(SOURCE)!=FROZEN_PLAN_SHA256:raise ValueError('approved frozen plan byte identity changed')
    declaration = json.loads(SOURCE.read_text(encoding='utf-8'))
    rows = declaration['cases']
    from collections import Counter
    if len(rows)!=18 or len({r['family_id'] for r in rows})!=6 or Counter(verified_label(r) for r in rows)!={'Yes':6,'No':6,'Undetermined':6}:
        raise ValueError('approved 18-case development identity mismatch')
    for row in rows:
        image = (ROOT / row['image_file']).resolve()
        if not image.is_relative_to(ROOT) or file_hash(image) != row['image_sha256']:
            raise ValueError('image identity mismatch')
        if hashlib.sha256(row['prompt'].encode()).hexdigest() != row['prompt_sha256']:
            raise ValueError('prompt identity mismatch')
    return rows


def validate_worker_output(out, tag, rows):
    execution=json.loads((out/'execution.json').read_text())
    attempts=[json.loads(p.read_text()) for p in sorted((out/'forward_ledger').glob('forward_*.json'))]
    scored=[json.loads(line) for line in (out/'candidate_scores.jsonl').read_text().splitlines()]
    if execution['status']!='PASSED' or execution['forward_attempts']!=54 or len(attempts)!=54 or len(scored)!=18:
        raise ValueError('incomplete worker evidence')
    if [a['attempt'] for a in attempts]!=list(range(1,55)):
        raise ValueError('invalid ledger sequence')
    if execution['optimizer_steps'] or execution['backward_passes'] or execution['causal_patching'] or execution['holdout_accessed']:
        raise ValueError('worker exceeded authorized scope')
    parameters=execution['parameter_audit']['parameters']
    if not parameters or not all(p['passed'] for p in parameters):raise ValueError('parameter invariance failure')
    telemetry=execution['telemetry']
    if telemetry['sampling_errors'] or telemetry['minimum_sampled_host_available_mib']<4096 or telemetry['maximum_sampled_nvidia_used_mib']>3900:
        raise ValueError('unsafe resource evidence')
    for index,(actual,expected) in enumerate(zip(scored,rows)):
        if any(actual[k]!=expected[k] for k in ('sample_id','family_id','ground_truth_label','image_sha256','prompt_sha256')):
            raise ValueError('worker row identity mismatch')
        scores=actual['summed_loglikelihoods']
        if len(scores)!=3 or not all(math.isfinite(v) for v in scores) or actual['model']!=tag or actual['is_generated_completion'] is not False:
            raise ValueError('invalid candidate evidence')
        if actual['argmax_answer']!=['Yes','No','Undetermined'][max(range(3),key=lambda i:scores[i])]:
            raise ValueError('invalid candidate argmax')
        selected=attempts[index*3:index*3+3]
        if any(a['model']!=tag or a['sample_id']!=expected['sample_id'] for a in selected) or [a['candidate_index'] for a in selected]!=[0,1,2]:
            raise ValueError('unaccounted candidate attempt')
    return {'valid_scored_rows':18,'parameter_comparisons':len(parameters),'status':'VERIFIED_EXECUTION'}


def worker(tag):
    # Only the parent may create the mission directory and admission record.
    if not (OUTPUT / 'plan.json').is_file():
        raise RuntimeError('parent admission required')
    plan = json.loads((OUTPUT / 'plan.json').read_text())
    if plan['source_sha256'] != file_hash(Path(__file__)) or plan['input_plan_sha256'] != file_hash(SOURCE) or any(file_hash(ROOT/n)!=h for n,h in plan['scientific_source_sha256'].items()):
        raise RuntimeError('execution source/input changed after freeze')
    rows = selected_cases()
    out = OUTPUT / tag
    out.mkdir()
    budget = ScreenBudget(out / 'forward_ledger', limit=54)
    import torch
    from PIL import Image
    from src.models.model_loader import load_pinned_smolvlm2
    from src.models.qwen3vl2_loader import load_qwen3vl2_nf4
    from src.training.differentiable_scoring import CANDIDATES, encode_candidates, candidate_class_loglikelihoods
    from src.training.scorer_verification import ParameterAudit, run_observed, sample_resources
    report = {'model': tag, 'optimizer_steps': 0, 'backward_passes': 0, 'causal_patching': False,
              'holdout_accessed': False, 'seed': 2026102503, 'pid': os.getpid(), 'source_sha256': plan['source_sha256']}
    loaded = {}
    context = {}
    def observe():
        snap = sample_resources()
        if snap.get('host_available_mib', 0) < 4096 or snap.get('nvidia_smi_used_mib', 4096) > 3900:
            unsafe[0] = True
        return snap
    unsafe = [False]
    def check():
        gate(resources())
        if unsafe[0]:raise RuntimeError('worker telemetry crossed resource gate')
    def action():
        gate(resources(), loading=True)
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        torch.manual_seed(report['seed'])
        start = time.perf_counter()
        processor, model, metadata = (load_pinned_smolvlm2 if tag == 'smol' else load_qwen3vl2_nf4)()
        loaded.update(model=model, processor=processor)
        report.update(identity=metadata, load_sec=time.perf_counter()-start)
        model.eval();model.requires_grad_(False)
        def before_forward(_module, _inputs):
            check()
            budget.consume(model=tag, sample_id=context['sample_id'], candidate_index=budget.used % 3)
            print(f'FORWARD {budget.used}/54 {tag} {context["sample_id"]}', flush=True)
        handle = model.register_forward_pre_hook(before_forward)
        try:
            with ParameterAudit(ROOT / '.cache/recovery-temp') as audit:
                audit.capture(model)
                token_ids = encode_candidates(processor.tokenizer)
                for row in rows:
                    context['sample_id'] = row['sample_id'];check()
                    with Image.open(ROOT / row['image_file']) as raw:
                        image = raw.convert('RGB')
                    content={'image':{'type':'image','image':image},'text':{'type':'text','text':row['prompt']}}
                    messages = [{'role':'user','content':[content[k] for k in row['content_order']]}]
                    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                          return_dict=True, return_tensors='pt').to('cuda:0')
                    started = time.perf_counter()
                    with torch.inference_mode():scores = candidate_class_loglikelihoods(model, inputs, token_ids)
                    torch.cuda.synchronize()
                    values = scores[0].cpu().tolist()
                    if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite candidate scores')
                    result = {k:row[k] for k in ('sample_id','family_id','ground_truth_label','image_sha256','prompt_sha256')}
                    result.update(model=tag, revision=metadata['revision'], summed_loglikelihoods=values,
                                  candidate_strings=list(CANDIDATES), candidate_token_ids=token_ids,
                                  argmax_answer=CANDIDATES[max(range(3),key=lambda i:values[i])].strip(),
                                  length_normalized_diagnostic=[v/len(ids) for v,ids in zip(values,token_ids)],
                                  is_generated_completion=False, latency_sec=time.perf_counter()-started,
                                  prompt_token_ids=inputs['input_ids'][0].cpu().tolist(),
                                  input_shapes={k:list(v.shape) for k,v in inputs.items() if hasattr(v,'shape')})
                    append_bytes(out / 'candidate_scores.jsonl', canonical(result)+b'\n')
                    print(json.dumps({k:result[k] for k in ('model','sample_id','argmax_answer','summed_loglikelihoods')}),flush=True)
                    del inputs,scores;image.close();check()
                report['parameter_audit'] = audit.compare(model)
                if not all(p['passed'] for p in report['parameter_audit']['parameters']):
                    raise RuntimeError('exact parameter comparison failed')
        finally:handle.remove()
        if budget.used != 54:raise RuntimeError('worker forward accounting mismatch')
    def cleanup():
        loaded.clear()
        import gc
        gc.collect();torch.cuda.empty_cache()
    code = run_observed(report, action, cleanup, sampler=observe, interval=.5)
    report['forward_attempts'] = budget.used
    if unsafe[0]:report['status']='FAILED';code=1
    write_once(out / 'execution.json', report)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('smol','qwen'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:return worker(args.worker)
    if OUTPUT.exists():raise RuntimeError('mission already admitted; no automatic retry or budget reset')
    rows = selected_cases()
    before = resources();gate(before, loading=True)
    from src.config.paths import get_hf_home
    hf=get_hf_home()
    for model,revision in [('HuggingFaceTB/SmolVLM2-2.2B-Instruct','482adb537c021c86670beed01cd58990d01e72e4'),('Qwen/Qwen3-VL-2B-Instruct','89644892e4d85e24eaac8bacfd4f463576704203')]:
        if not any((base/f"models--{model.replace('/','--')}"/'snapshots'/revision).is_dir() for base in (hf,hf/'hub')):
            raise RuntimeError('pinned local snapshot missing; no download allowed')
    OUTPUT.mkdir(parents=True)
    plan = {'schema':'publication_approved_screen108_v1','authorization_date':'2026-10-25',
            'additional_forward_limit':108,'historical_forward_attempts':18,'worker_order':['smol','qwen'],
            'cases':rows,'source_sha256':file_hash(Path(__file__)),'input_plan_sha256':file_hash(SOURCE),
            'scientific_source_sha256':{name:file_hash(ROOT/name) for name in ('src/models/model_loader.py','src/models/qwen3vl2_loader.py','src/training/differentiable_scoring.py','src/evaluation/screen_budget.py','src/training/scorer_verification.py')},
            'preflight':before,'optimizer_steps':0,'causal_patching':False,'holdout_accessed':False,
            'estimate':json.loads(SOURCE.read_text())['runtime_estimate_sec'],
            'approval':'User explicitly approved this frozen 108-forward plan on 2026-10-25; no training/patching/holdout',
            'release_policy':'Child exit required; free VRAM >=3500MiB and GPU use <=pre-worker baseline+256MiB within 30s. Log host recovery; require >=4GiB, not byte equality during unrelated OS activity.'}
    write_once(OUTPUT / 'plan.json', plan)
    report = {'status':'FAILED','workers':[],'additional_forward_attempts':0}
    try:
        for tag in plan['worker_order']:
            baseline = resources();gate(baseline, loading=True)
            with (OUTPUT / f'{tag}_stdout.txt').open('xb') as stdout, (OUTPUT / f'{tag}_stderr.txt').open('xb') as stderr:
                child = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',tag],cwd=ROOT,stdout=stdout,stderr=stderr)
                started = time.monotonic()
                try:
                    while child.poll() is None:
                        gate(resources())
                        if time.monotonic()-started > 600:raise RuntimeError('worker wall budget exhausted')
                        time.sleep(.5)
                finally:
                    if child.poll() is None:child.terminate()
                    code = child.wait(timeout=30)
            release_started = time.monotonic()
            while True:
                after = resources()
                if after['nvidia_free_mib'] >= 3500 and after['nvidia_used_mib'] <= baseline['nvidia_used_mib']+256:break
                if time.monotonic()-release_started > 30:raise RuntimeError('GPU did not return to admissible baseline after worker exit')
                time.sleep(.5)
            gate(after, loading=True)
            entry={'model':tag,'pid':child.pid,'exit_code':code,'process_exited':child.poll() is not None,
                   'before':baseline,'after_process_exit':after,'elapsed_sec':time.monotonic()-started}
            report['workers'].append(entry)
            attempts=list((OUTPUT / tag / 'forward_ledger').glob('forward_*.json'))
            report['additional_forward_attempts'] += len(attempts)
            if code != 0:raise RuntimeError(f'{tag} worker failed; no subsequent model or automatic retry')
            entry['validation']=validate_worker_output(OUTPUT/tag,tag,rows)
        if report['additional_forward_attempts']!=108:raise RuntimeError('global forward accounting mismatch')
        report['status']='PASSED'
    except Exception as exc:
        report.update(error_type=type(exc).__name__,error_message=str(exc))
    finally:
        report['additional_forward_attempts']=len(list(OUTPUT.glob('*/forward_ledger/forward_*.json')))
        report['final_resources']=resources()
        write_once(OUTPUT / 'execution.json', report)
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='PASSED' else 1


if __name__=='__main__':raise SystemExit(main())
