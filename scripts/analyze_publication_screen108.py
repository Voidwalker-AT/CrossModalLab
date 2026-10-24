"""Validate preserved screen failure on CPU without retrying any forward."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.matched_screen import complete_matched_summary
from src.training.run_state import file_hash, write_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('fresh checkout-local output required')
    directory = ROOT / 'artifacts/publication/20261025/screen108'
    execution = json.loads((directory / 'execution.json').read_text())
    plan = json.loads((directory / 'plan.json').read_text())
    if file_hash(directory / 'executed_source.py') != plan['source_sha256']:
        raise ValueError('executed source identity changed')
    rows = []; attempts = []
    for model in ('smol', 'qwen'):
        ledger = sorted((directory / model / 'forward_ledger').glob('forward_*.json'))
        entries = [json.loads(p.read_text()) for p in ledger]
        if [r['attempt'] for r in entries] != list(range(1, len(entries) + 1)):
            raise ValueError('non-contiguous attempt ledger')
        attempts.extend(entries)
        path = directory / model / 'candidate_scores.jsonl'
        if path.exists(): rows.extend(json.loads(line) for line in path.read_text().splitlines())
    if len(attempts) != execution['additional_forward_attempts'] or len(attempts) > 108:
        raise ValueError('forward accounting mismatch')
    try:
        complete_matched_summary(plan['cases'], rows, ['smol', 'qwen'],
                                execution_status=execution['status'])
    except ValueError as exc:
        reason = str(exc)
    else:
        raise ValueError('expected failed screen; investigate without overwriting evidence')
    result = {'schema': 'publication_screen_failure_analysis_v1',
              'execution_status': execution['status'], 'attempted_forwards': len(attempts),
              'completed_scored_rows': len(rows), 'complete_metrics_rejected': reason,
              'effect_size': None, 'confidence_interval': None,
              'triggering_resource': 'UNVERIFIED: original parent did not preserve its triggering sample',
              'parameter_invariance': 'UNVERIFIED: worker terminated before final comparison',
              'retry_performed': False, 'qwen_worker_started': (directory / 'qwen').exists(),
              'executed_source_sha256': plan['source_sha256'],
              'source_hashes': {name: file_hash(directory / name) for name in
                                ('plan.json', 'execution.json', 'failure_audit.json', 'executed_source.py')}}
    write_once(output, result)
    print(json.dumps(result))


if __name__ == '__main__': main()
