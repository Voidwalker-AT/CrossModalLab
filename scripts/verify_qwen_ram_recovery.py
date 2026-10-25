"""Read-only semantic validation of a completed bounded candidate-scoring run."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_qwen_ram_recovery import selected_cases, PROTOCOL_HASH
from src.training.run_state import file_hash


def verify(out):
    out = Path(out)
    read = lambda name: json.loads((out / name).read_text(encoding='utf-8'))
    plan, parent, worker = read('plan.json'), read('execution.json'), read('worker_execution.json')
    protocol, expected = selected_cases()
    if plan['protocol_sha256'] != PROTOCOL_HASH or plan['protocol'] != protocol:
        raise ValueError('protocol mismatch')
    if any(file_hash(out / 'executed_sources' / n) != h for n, h in plan['source_sha256'].items()):
        raise ValueError('executed source mismatch')
    if file_hash(ROOT / 'dashboard/assets/development_cases.json') != plan['development_allowlist_sha256']:
        raise ValueError('allowlist mismatch')
    if parent['status'] != 'PASSED' or worker['status'] != 'PASSED' or parent['worker_exit_code'] != 0:
        raise ValueError('incomplete or failed execution')
    ledger = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((out / 'forward_ledger').glob('forward_*.json'))]
    if len(ledger) != 9 or parent['attempted_forwards'] != 9 or worker['forward_attempts'] != 9:
        raise ValueError('forward accounting mismatch')
    if [r['attempt'] for r in ledger] != list(range(1, 10)):
        raise ValueError('ledger sequence mismatch')
    if any(worker[k] != v for k, v in dict(optimizer_steps=0, backward_passes=0,
            causal_patching=False, holdout_accessed=False, is_generated_completion=False).items()):
        raise ValueError('scope mismatch')
    if worker['identity']['model_id'] != protocol['model'] or worker['identity']['revision'] != protocol['revision']:
        raise ValueError('model mismatch')
    audit = worker['parameter_audit']
    if not audit['passed'] or not audit['parameters'] or not all(
            r['passed'] and r['raw_hash_before'] == r['raw_hash_after'] and
            r['requires_grad_before'] is False and r['requires_grad_after'] is False
            for r in audit['parameters']):
        raise ValueError('parameter invariance mismatch')
    telemetry = worker['telemetry']
    if telemetry['sampling_errors'] or telemetry['minimum_sampled_host_available_mib'] < 4096 or telemetry['maximum_sampled_nvidia_used_mib'] > 3900:
        raise ValueError('worker resource violation')
    samples = parent['parent_resource_samples']
    if not samples or min(r['host_available_mib'] for r in samples) < 4096 or max(r['nvidia_used_mib'] for r in samples) > 3900:
        raise ValueError('parent resource violation')
    if not parent['worker_process_exited'] or not parent['gpu_returned_to_baseline']:
        raise ValueError('worker release incomplete')
    scored = [json.loads(line) for line in (out / 'candidate_scores.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(scored) != 3:
        raise ValueError('incomplete scored cases')
    for index, (actual, target) in enumerate(zip(scored, expected)):
        keys = ('sample_id', 'family_id', 'ground_truth_label', 'image_sha256', 'prompt_sha256', 'prompt')
        if any(actual[k] != target[k] for k in keys):
            raise ValueError('scored input identity mismatch')
        ll = actual['summed_loglikelihoods']
        if len(ll) != 3 or not all(math.isfinite(x) for x in ll):
            raise ValueError('nonfinite likelihood')
        answer = protocol['candidates'][max(range(3), key=lambda i: ll[i])].strip()
        if actual['argmax_answer'] != answer or actual['candidate_argmax_correct'] != (answer == target['ground_truth_label']):
            raise ValueError('argmax mismatch')
        if actual['is_generated_completion'] is not False or actual['exact_format_compliance'] is not None or actual['parsed_generated_answer'] is not None:
            raise ValueError('generation and candidate metrics conflated')
        if actual['candidate_token_ids'] != [[7414], [2308], [16887, 46637]]:
            raise ValueError('candidate token mismatch')
        if hashlib.sha256(actual['effective_chat_template'].encode()).hexdigest() != actual['effective_chat_template_sha256']:
            raise ValueError('effective template hash mismatch')
        attempts = ledger[index * 3:index * 3 + 3]
        if any(r['sample_id'] != target['sample_id'] or r['model'] != protocol['model'] for r in attempts) or [r['candidate_index'] for r in attempts] != [0, 1, 2]:
            raise ValueError('candidate ledger mismatch')
    return dict(status='VERIFIED', actual_forwards=9, cases=3, independent_families=1,
                finite_candidate_scores=9, candidate_argmax_correct=sum(r['candidate_argmax_correct'] for r in scored),
                parameter_tensors_unchanged=len(audit['parameters']), generation_metrics='NOT_APPLICABLE',
                historical_generation_reproduction='NOT_ESTABLISHED: different scoring procedure')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    print(json.dumps(verify(parser.parse_args().output), indent=2))
