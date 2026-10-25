"""Execute the frozen three-case protocol once, with no retry or generation.

Uses the existing loader, candidate scorer, forward ledger and telemetry.
Only the parent launches a model worker; existing output is never resumed.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_publication_isolated_pilot import resources, gate
from src.training.run_state import write_once, append_bytes, canonical, file_hash
from src.evaluation.forward_budget import ForwardBudget

PROTOCOL = ROOT / 'paper/overleaf_evidence_20261025/audit/replication_protocol.json'
PROTOCOL_HASH = '247c6aef4b06d6d2282d87e983fcefd6662e00a8a8fb21a2ff15f5dc50fa3ff3'
SOURCES = ('scripts/run_qwen_ram_recovery.py', 'scripts/run_publication_isolated_pilot.py',
           'src/models/qwen3vl2_loader.py', 'src/training/differentiable_scoring.py',
           'src/training/scorer_verification.py', 'src/evaluation/forward_budget.py')


def selected_cases():
    if file_hash(PROTOCOL) != PROTOCOL_HASH:
        raise ValueError('frozen protocol identity mismatch')
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    available = json.loads((ROOT / 'dashboard/assets/development_cases.json').read_text(encoding='utf-8'))
    by_id = {r['sample_id']: r for r in available}
    rows = []
    for frozen in protocol['cases']:
        row = by_id[frozen['sample_id']]
        if any(row[k] != v for k, v in frozen.items()):
            raise ValueError('development input differs from frozen protocol')
        image = (ROOT / row['image_file']).resolve()
        if not image.is_relative_to(ROOT) or file_hash(image) != frozen['image_sha256']:
            raise ValueError('image hash mismatch')
        if hashlib.sha256(row['prompt'].encode('utf-8')).hexdigest() != frozen['prompt_sha256']:
            raise ValueError('prompt hash mismatch')
        if row['split'] not in ('validation', 'development') or not row['oracle_agreement']:
            raise ValueError('not an oracle-verified development case')
        rows.append(row)
    if [r['ground_truth_label'] for r in rows] != ['Yes', 'No', 'Undetermined']:
        raise ValueError('class order mismatch')
    return protocol, rows


def worker(out):
    plan = json.loads((out / 'plan.json').read_text(encoding='utf-8'))
    if any(file_hash(ROOT / n) != h for n, h in plan['source_sha256'].items()):
        raise ValueError('source changed after admission')
    if file_hash(ROOT / 'dashboard/assets/development_cases.json') != plan['development_allowlist_sha256']:
        raise ValueError('development allowlist changed after admission')
    protocol, rows = selected_cases()
    budget = ForwardBudget(out / 'forward_ledger', limit=protocol['planned_forwards'])
    import torch
    from PIL import Image
    from src.models.qwen3vl2_loader import load_qwen3vl2_nf4
    from src.training.differentiable_scoring import CANDIDATES, encode_candidates, candidate_class_loglikelihoods
    from src.training.scorer_verification import ParameterAudit, run_observed, sample_resources
    report = dict(model=protocol['model'], revision=protocol['revision'], adapter='none; frozen base',
                  optimizer_steps=0, backward_passes=0, causal_patching=False, holdout_accessed=False,
                  is_generated_completion=False, pid=os.getpid(), seed=2026102501)
    loaded, context, unsafe = {}, {}, [False]

    def observe():
        snap = sample_resources()
        if snap.get('host_available_mib', 0) < 4096 or snap.get('nvidia_smi_used_mib', 4096) > 3900:
            unsafe[0] = True
        return snap

    def check():
        gate(resources())
        if unsafe[0]:
            raise RuntimeError('sampled resource gate crossed; no retry')

    def action():
        gate(resources(), loading=True)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.manual_seed(report['seed'])
        processor, model, metadata = load_qwen3vl2_nf4(local_files_only=True)
        loaded.update(model=model, processor=processor)
        report['identity'] = metadata
        if metadata['model_id'] != protocol['model'] or metadata['revision'] != protocol['revision']:
            raise ValueError('model identity mismatch')
        model.eval()
        model.requires_grad_(False)
        check()

        def before_forward(module, args, kwargs):
            check()
            # Runs after the safety check, immediately before entering the model.
            budget.consume(model=protocol['model'], sample_id=context['sample_id'],
                           candidate_index=budget.used % 3)
            ids = kwargs['input_ids']
            for name in ('attention_mask', 'mm_token_type_ids'):
                if kwargs[name].shape != ids.shape:
                    raise ValueError('multimodal sequence alignment mismatch')
            print(f'FORWARD {budget.used}/9 {context["sample_id"]}', flush=True)

        handle = model.register_forward_pre_hook(before_forward, with_kwargs=True)
        try:
            with ParameterAudit(ROOT / '.cache/recovery-temp') as audit:
                audit.capture(model)
                tokens = encode_candidates(processor.tokenizer)
                report['candidate_token_ids'] = tokens
                if list(CANDIDATES) != protocol['candidates'] or tokens != [[7414], [2308], [16887, 46637]]:
                    raise ValueError('frozen candidate token identity mismatch')
                for row in rows:
                    context['sample_id'] = row['sample_id']
                    check()
                    with Image.open(ROOT / row['image_file']) as raw:
                        image = raw.convert('RGB')
                    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                                                             {'type': 'text', 'text': row['prompt']}]}]
                    rendered = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                           return_dict=True, return_tensors='pt').to('cuda:0')
                    started = time.perf_counter()
                    with torch.inference_mode():
                        scores = candidate_class_loglikelihoods(model, inputs, tokens)
                    torch.cuda.synchronize()
                    values = scores[0].cpu().tolist()
                    if not all(math.isfinite(v) for v in values):
                        raise ValueError('nonfinite candidate likelihood')
                    answer = CANDIDATES[max(range(3), key=lambda i: values[i])].strip()
                    result = {k: row[k] for k in ('sample_id', 'family_id', 'ground_truth_label',
                                                  'image_file', 'image_sha256', 'prompt', 'prompt_sha256')}
                    result.update(model=protocol['model'], revision=protocol['revision'], adapter='none',
                                  scoring=protocol['score'], candidate_strings=list(CANDIDATES),
                                  candidate_token_ids=tokens, summed_loglikelihoods=values, argmax_answer=answer,
                                  candidate_argmax_correct=answer == row['ground_truth_label'],
                                  is_generated_completion=False, parsed_generated_answer=None,
                                  exact_format_compliance=None, generated_answer_metrics='NOT_APPLICABLE',
                                  effective_chat_template=rendered,
                                  effective_chat_template_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                                  prompt_token_ids=inputs['input_ids'][0].cpu().tolist(),
                                  input_shapes={k: list(v.shape) for k, v in inputs.items() if hasattr(v, 'shape')},
                                  latency_sec=time.perf_counter() - started, attempted_forwards=budget.used)
                    append_bytes(out / 'candidate_scores.jsonl', canonical(result) + b'\n')
                    print(json.dumps({k: result[k] for k in ('sample_id', 'argmax_answer', 'summed_loglikelihoods')}), flush=True)
                    del inputs, scores
                    image.close()
                    check()
                report['parameter_audit'] = audit.compare(model)
                if not report['parameter_audit']['passed']:
                    raise RuntimeError('parameter invariance failed')
        finally:
            handle.remove()
        if budget.used != 9:
            raise RuntimeError('forward count mismatch')

    def cleanup():
        loaded.clear()
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    code = run_observed(report, action, cleanup, sampler=observe, interval=.5)
    report['forward_attempts'] = budget.used
    if unsafe[0]:
        report['status'] = 'FAILED'
        code = 1
    write_once(out / 'worker_execution.json', report)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / 'artifacts/publication/20261025'):
        raise ValueError('output must be a new publication evidence directory')
    if args.worker:
        return worker(out)
    if out.exists():
        raise FileExistsError('existing mission cannot be retried or reset')
    protocol, rows = selected_cases()
    out.mkdir()
    report = dict(status='BLOCKED', attempted_forwards=0, optimizer_steps=0, holdout_accessed=False,
                  parent_resource_samples=[])
    child = None
    try:
        before = resources()
        report['before_worker'] = before
        gate(before, loading=True)
        # Check local snapshot without loading or downloading any model.
        from src.config.paths import get_hf_home
        hf = get_hf_home()
        repo = 'models--' + protocol['model'].replace('/', '--')
        if not any((base / repo / 'snapshots' / protocol['revision'] / 'model.safetensors').is_file()
                   for base in (hf, hf / 'hub')):
            raise FileNotFoundError('existing pinned monolithic model missing')
        write_once(out / 'plan.json', dict(protocol=protocol, protocol_sha256=PROTOCOL_HASH,
                   source_sha256={n: file_hash(ROOT / n) for n in SOURCES},
                   development_allowlist_sha256=file_hash(ROOT / 'dashboard/assets/development_cases.json'),
                   planned_forwards=9, hard_authorized_cap=12, previous_attempts=0,
                   wall_limit_sec=600, minimum_host_mib=4096, maximum_physical_gpu_mib=3900,
                   minimum_loading_free_vram_mib=3500, minimum_disk_reserve_gib=4,
                   failure_policy='stop without retry', generation=False))
        # Terminal output remains local; scientific evidence is written separately.
        logs = ROOT / '.cache' / out.name
        logs.mkdir(parents=True, exist_ok=True)
        with (logs / 'stdout.txt').open('xb') as stdout, (logs / 'stderr.txt').open('xb') as stderr:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--output', str(out), '--worker'],
                                     cwd=ROOT, stdout=stdout, stderr=stderr)
            report.update(status='FAILED', worker_pid=child.pid)
            started = time.monotonic()
            try:
                while child.poll() is None:
                    snap = resources()
                    report['parent_resource_samples'].append(dict(elapsed_sec=time.monotonic() - started, **snap))
                    gate(snap)
                    if time.monotonic() - started > 600:
                        raise RuntimeError('worker wall limit exceeded')
                    time.sleep(.5)
            finally:
                if child.poll() is None:
                    child.terminate()  # Only this owned worker, to enforce resource/wall gates.
                report['worker_exit_code'] = child.wait(timeout=30)
                report['worker_process_exited'] = child.poll() is not None
            execution = json.loads((out / 'worker_execution.json').read_text(encoding='utf-8'))
            report['status'] = 'PASSED' if report['worker_exit_code'] == 0 and execution['status'] == 'PASSED' else 'FAILED'
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error_message=str(exc))
    finally:
        report['attempted_forwards'] = len(list((out / 'forward_ledger').glob('forward_*.json')))
        report['after_worker'] = resources()
        report['gpu_returned_to_baseline'] = (report['after_worker']['nvidia_used_mib'] <=
                                              report.get('before_worker', report['after_worker'])['nvidia_used_mib'] + 256)
        if report['status'] == 'PASSED' and (report['attempted_forwards'] != 9 or not report['gpu_returned_to_baseline']):
            report['status'] = 'FAILED'
        write_once(out / 'execution.json', report)
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
