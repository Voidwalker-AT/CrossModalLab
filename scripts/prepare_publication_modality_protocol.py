"""Freeze a CPU-validated, NOT AUTHORIZED Qwen modality-control proposal."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.symbolic.relations import Premise, Query
from src.symbolic.solver import SymbolicSolver
from src.training.run_state import file_hash, write_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('fresh checkout-local proposal path required')
    source = ROOT / 'artifacts/publication/20261025/followup_108_plan.json'
    plan = json.loads(source.read_text()); conditions = []
    for row in plan['cases']:
        text = next(p for p in row['premises'] if p['modality'] == 'textual')
        sentence = 'The ' + text['subject'].replace('_', ' ') + ' is to the left of the ' + text['object'].replace('_', ' ') + '.'
        if row['prompt'].count(sentence) != 1: raise ValueError('text premise anchor ambiguous')
        for mode in ('joint', 'text_only', 'image_only'):
            kept = [p for p in row['premises'] if mode == 'joint' or
                    p['modality'] == ('textual' if mode == 'text_only' else 'visual')]
            solved = SymbolicSolver.solve([Premise(**p) for p in kept], Query(**row['query']), verify_with_oracle=True)
            label = {'entailed': 'Yes', 'contradicted': 'No', 'underdetermined': 'Undetermined'}[solved.epistemic_status.value]
            expected = row['ground_truth_label'] if mode == 'joint' else 'Undetermined'
            if not solved.algorithm_agreement or label != expected: raise ValueError('condition label failed dual oracle')
            prompt = row['prompt'] if mode != 'image_only' else row['prompt'].replace(sentence, '').strip()
            conditions.append({'sample_id': row['sample_id'], 'family_id': row['family_id'], 'condition': mode,
                               'prompt': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                               'image_file': None if mode == 'text_only' else row['image_file'],
                               'image_sha256': None if mode == 'text_only' else row['image_sha256'],
                               'content_order': ['text'] if mode == 'text_only' else row['content_order'],
                               'full_context_diagnostic_target': row['ground_truth_label'],
                               'available_evidence_gold': label})
    if len(conditions) != 54 or len({(r['sample_id'], r['condition']) for r in conditions}) != 54:
        raise ValueError('exact 18 x three conditions required')
    declaration = {'schema': 'publication_qwen_modality_proposal_v1', 'authorized': False,
                   'status': 'PROPOSED_NOT_EXECUTED_REQUIRES_EXPLICIT_APPROVAL',
                   'selection_source_sha256': file_hash(source), 'generator_sha256': file_hash(Path(__file__)),
                   'model': 'Qwen/Qwen3-VL-2B-Instruct', 'revision': '89644892e4d85e24eaac8bacfd4f463576704203',
                   'quantization': '4-bit NF4, double quantization, BF16 compute; existing local snapshot only',
                   'conditions': conditions, 'candidates': [' Yes', ' No', ' Undetermined'],
                   'additional_forward_cap': 162, 'generation': False, 'optimizer_steps': 0, 'causal_patching': False,
                   'scope': '18 exposed development cases / six families; no held-out confirmation; no Smol worker.',
                   'scoring': 'Same repaired complete-answer summed log-likelihood; LL/token secondary diagnostic only. Exact candidate tokens recorded per run.',
                   'primary_endpoint': 'Paired joint-minus-stronger-unimodal balanced accuracy against FULL-CONTEXT diagnostic targets; this is label predictability, not evidence-valid correctness of withheld-input answers.',
                   'secondary_endpoint': 'Available-evidence correctness and unsupported-answer/over-refusal counts separately; every unimodal query is logically Undetermined. No three-class BA on all-U condition gold.',
                   'statistics': 'Six-family paired descriptive bootstrap B=2000 only if every planned row succeeds; no confirmatory p-values. Complete joint three-class counts and missing outputs required.',
                   'resource_gates': {'minimum_available_host_mib_throughout': 4096, 'minimum_free_dedicated_vram_mib_before_load': 3500,
                                      'maximum_physical_gpu_used_mib': 3900, 'minimum_disk_reserve_gib': 4,
                                      'hard_worker_seconds': 600, 'hard_total_seconds': 660, 'worker_processes': 1,
                                      'no_shared_vram': True, 'sampling_seconds': 0.5},
                   'runtime_estimate': 'About three minutes: 162 x observed 0.56s/candidate plus approximately 80s load/audit; UNVERIFIED workload transfer. Hard limits above apply.',
                   'execution_admission': 'Not runnable by this generator. A separately reviewed worker must consume this exact plan, validate no-image/text-only handling on CPU, retain telemetry before each gate, exact parameter before/after comparisons, no conflicting CUDA model, PID/creation-time ownership and baseline restoration. Approval is required before admission.',
                   'failure_policy': 'Count every attempt; preserve outputs and triggering sample; stop without retry or quota reuse; no complete-screen result if any planned row is missing.',
                   'rationale': 'Qwen-only safety/construct diagnostic after failed Smol screen, not a completed cross-model replication or a replacement for independent evidence.'}
    write_once(output, declaration)
    print(json.dumps({'status': declaration['status'], 'conditions': 54, 'forwards_if_approved': 162,
                      'cpu_dual_oracle_checks': 54, 'actual_model_forwards': 0}))


if __name__ == '__main__': main()
