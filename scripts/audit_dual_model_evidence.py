"""Read-only CPU audit of historical model evidence; never loads either VLM."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_overleaf_evidence import recompute, same_metrics
from scripts.build_neural_evidence import build as check_neural
from scripts.verify_qwen_ram_recovery import verify as check_qwen
from src.training.run_state import file_hash

OLD = ROOT / 'paper/overleaf_evidence_20261025'
REPLICATION = ROOT / 'artifacts/publication/20261025/qwen_replication_20261025_464531de'
MODELS = {'qwen': ('Qwen/Qwen3-VL-2B-Instruct', '89644892e4d85e24eaac8bacfd4f463576704203'),
          'smol': ('HuggingFaceTB/SmolVLM2-2.2B-Instruct', '482adb537c021c86670beed01cd58990d01e72e4')}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s]


def compare_inputs(left, right, *, left_method, right_method):
    """Common task strings/images, not model-native token IDs, define matching."""
    keys = ('sample_id', 'image_sha256', 'prompt_sha256', 'ground_truth_label')
    if any(left.get(k) != right.get(k) for k in keys):
        return 'NOT COMPARABLE'
    return 'DIRECTLY COMPARABLE' if left_method == right_method else 'PARTIALLY COMPARABLE'


def audit():
    sources = {}
    def track(path):
        path = Path(path).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('audit source outside workspace')
        sources[path.relative_to(ROOT).as_posix()] = file_hash(path)
        return path
    def read(name):
        return load(track(ROOT / name))
    def rows(name):
        return lines(track(ROOT / name))
    from src.config.paths import get_hf_home
    architectures = {}
    for tag, (model, revision) in MODELS.items():
        repo = 'models--' + model.replace('/', '--')
        bases = (get_hf_home(), get_hf_home() / 'hub')
        config = next(base / repo / 'snapshots' / revision / 'config.json' for base in bases
                      if (base / repo / 'snapshots' / revision / 'config.json').is_file())
        cfg = load(config)
        architectures[tag] = dict(model_id=model, revision=revision, config_sha256=file_hash(config),
            architectures=cfg['architectures'], model_type=cfg['model_type'],
            text={k: v for k, v in cfg['text_config'].items() if k in
                  ('model_type', 'hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads')},
            vision={k: v for k, v in cfg['vision_config'].items() if k in
                    ('model_type', 'hidden_size', 'num_hidden_layers', 'patch_size', 'depth', 'out_hidden_size')})

    smol_path = 'artifacts/evaluations/phase_0d2/behavioral_predictions_v1.jsonl'
    smol = rows(smol_path)
    manifest_path = 'configs/phase_0d2/behavioral_manifest_v1.json'
    manifest = read(manifest_path)
    by_id = {r['sample_id']: r for r in manifest['samples']}
    if {r['sample_id'] for r in smol} != set(by_id) or len(smol) != 36:
        raise ValueError('Smol manifest membership mismatch')
    inputs = []
    for row in smol:
        target = by_id[row['sample_id']]
        if row['prediction_status'] != 'ok' or row['is_synthetic_demo'] is not False:
            raise ValueError('non-empirical or failed Smol row')
        for key in ('prompt', 'image_sha256', 'ground_truth_label', 'matched_group_id'):
            if row[key] != target[key]:
                raise ValueError('Smol input identity mismatch')
        if row['manifest_sha256'] != file_hash(ROOT / manifest_path):
            raise ValueError('Smol manifest hash mismatch')
        if (row['model']['model_id'], row['model']['revision']) != MODELS['smol']:
            raise ValueError('Smol model identity mismatch')
        if row['is_correct'] != (row['parsed_answer'] == row['ground_truth_label']):
            raise ValueError('Smol correctness mismatch')
        image = ROOT / row['image_file'].replace('\\', '/')
        if file_hash(track(image)) != row['image_sha256']:
            raise ValueError('original Smol image changed')
        inputs.append({k: row[k] for k in ('sample_id', 'matched_group_id', 'condition', 'modality',
                       'ground_truth_label', 'prompt', 'image_file', 'image_sha256', 'raw_completion',
                       'parsed_answer', 'generation_config', 'run_id', 'timestamp_utc', 'candidate_scores')})
        inputs[-1]['prompt_sha256'] = hashlib.sha256(row['prompt'].encode()).hexdigest()
    smol_for_metrics = [dict(r, family_id=r['matched_group_id']) for r in smol]
    metrics = recompute(smol_for_metrics)
    stored = read('artifacts/evaluations/phase_0d2/behavioral_metrics_v1.json')
    if abs(metrics['balanced_accuracy'] - stored['balanced_accuracy']) > .5e-6 or metrics['correct'] != 18:
        raise ValueError('Smol historical metric discrepancy')
    conditions = {c: dict(n=len(group), correct=sum(r['is_correct'] for r in group))
                  for c in sorted({r['condition'] for r in smol})
                  for group in [[r for r in smol if r['condition'] == c]]}

    qraw = read('paper/overleaf_evidence_20261025/audit/raw_predictions.json')
    qsummary = read('paper/overleaf_evidence_20261025/audit/recomputed.json')
    qmetrics = {c: recompute([r for r in qraw if r['condition'] == c]) for c in qsummary['conditions']}
    if not same_metrics(qmetrics, qsummary['conditions']):
        raise ValueError('Qwen historical metric discrepancy')
    neural = check_neural(ROOT)  # Pure CPU validation, does not write the dashboard.
    prior_neural = read('dashboard/assets/neural_evidence_v1.json')
    for source in prior_neural['sources']:
        if file_hash(track(ROOT / source['path'])) != source['sha256']:
            raise ValueError('historical neural source changed')
    probe = read('artifacts/probes/phase_1f/probe_reliability_summary_v1.json')
    feature_rows = rows('artifacts/probes/phase_1f/features_meta_v1.jsonl')
    train, val = set(probe['train_families']), set(probe['val_families'])
    if train & val or len(train) != 27 or len(val) != 9:
        raise ValueError('probe group split mismatch')
    if any(r['family_id'] not in train | val for r in feature_rows):
        raise ValueError('feature family absent from authoritative evaluation split')
    split_metadata = dict(acquisition_roles=dict(Counter(r['split_role'] for r in feature_rows)),
        evaluation_rows=dict(train=sum(r['family_id'] in train for r in feature_rows),
                             validation=sum(r['family_id'] in val for r in feature_rows)),
        role_membership_disagreements=sum((r['family_id'] in train) != (r['split_role'] == 'train') for r in feature_rows),
        interpretation='Acquisition split_role differs from Phase 1F family-based evaluation; use stored family lists and runner source, not role strings')
    causal = rows('artifacts/causal/phase_2b/intervention_records_v1.jsonl')
    controls = defaultdict(list)
    for r in causal:
        if r['role'] == 'intervention':
            controls[r['direction']].append(r['delta_m']['delta_m'])
    causal_audit = read('artifacts/causal/phase_2b/recovery_audit_v1.json')
    for direction, effects in controls.items():
        expected = causal_audit['effect_audit']['mean_abs_delta_m_by_direction'][direction]
        if abs(sum(abs(v) for v in effects) / len(effects) - expected) > 1e-12:
            raise ValueError('historical causal effect recomputation mismatch')

    pilot_root = 'artifacts/publication/20261025/pilot'
    pilot_plan = read(pilot_root + '/plan.json')
    pilot_parent = read(pilot_root + '/execution.json')
    common_rows, pilot_reports = {}, {}
    for tag in ('smol', 'qwen'):
        report = read(f'{pilot_root}/{tag}/execution.json')
        scores = rows(f'{pilot_root}/{tag}/candidate_scores.jsonl')
        ledger = sorted((ROOT / pilot_root / tag / 'forward_ledger').glob('forward_*.json'))
        for p in ledger:
            track(p)
        if report['status'] != 'PASSED' or report['forward_attempts'] != 6 or len(ledger) != 6:
            raise ValueError('matched pilot not complete')
        if not report['parameter_audit']['passed']:
            raise ValueError('matched pilot invariance not established')
        if (report['identity']['model_id'], report['identity']['revision']) != MODELS[tag]:
            raise ValueError('matched pilot model mismatch')
        for r, frozen in zip(scores, pilot_plan['cases']):
            if compare_inputs(r, frozen, left_method='sum_LL', right_method='sum_LL') != 'DIRECTLY COMPARABLE':
                raise ValueError('matched pilot input mismatch')
            if file_hash(track(ROOT / frozen['image_file'])) != frozen['image_sha256']:
                raise ValueError('matched pilot image changed')
            if not all(math.isfinite(v) for v in r['summed_loglikelihoods']):
                raise ValueError('matched pilot nonfinite score')
        common_rows[tag] = [{k: r[k] for k in ('sample_id', 'family_id', 'ground_truth_label', 'image_sha256',
                              'prompt_sha256', 'summed_loglikelihoods', 'candidate_strings', 'argmax_answer', 'revision')}
                            for r in scores]
        pilot_reports[tag] = dict(status=report['status'], forwards=6, identity=report['identity'])
    if pilot_parent['status'] != 'PASSED' or pilot_parent['additional_forward_attempts'] != 12:
        raise ValueError('matched pilot parent accounting mismatch')

    qrep = check_qwen(REPLICATION)
    rep_scores = rows('artifacts/publication/20261025/qwen_replication_20261025_464531de/candidate_scores.jsonl')
    extra = {}
    for name in ('artifacts/evaluations/phase_0d2_smoke/run_metadata_v1.json',
                 'artifacts/evaluations/phase_1b/phase_1b_pilot_summary_v1.json',
                 'artifacts/evaluations/smolvlm2_feasibility_audit_results.json',
                 'artifacts/evaluations/smolvlm2_extended_validation_results.json',
                 'artifacts/probes/phase_1d/probe_pilot_summary_v1.json',
                 'artifacts/probes/phase_1e/contrast_probe_summary_v1.json',
                 'artifacts/training/phase_2f/qlora_feasibility_report_v1.json',
                 'artifacts/training/phase_2g/phase_2g_report_v1.json',
                 'artifacts/training/phase_2i_a/phase_2i_a_report_v1.json',
                 'artifacts/publication/20261025/screen108/execution.json'):
        data = read(name)
        extra[name] = dict(keys=list(data), sha256=sources[name])
    source_inventory = []
    # Inventory all stored research JSON/JSONL, without opening restricted datasets.
    for folder in ('evaluations', 'instrumentation', 'probes', 'causal', 'training'):
        for p in sorted((ROOT / 'artifacts' / folder).rglob('*')):
            if p.suffix not in ('.json', '.jsonl') or not p.is_file():
                continue
            track(p)
            source_inventory.append(dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                                         sha256=sources[p.relative_to(ROOT).as_posix()]))
    for name in ('scripts/audit_dual_model_evidence.py', 'scripts/run_phase_0d2_behavioral_benchmark.py',
                 'scripts/run_phase_1f_probe_eval.py', 'src/probes/eval_protocol.py',
                 'src/models/model_loader.py', 'src/models/qwen3vl2_loader.py',
                 'configs/phase_2b/causal_pilot_exec_spec_v1.json'):
        track(ROOT / name)
    return dict(status='CPU_ONLY_RECOMPUTATION', new_model_forwards=0, new_optimizer_steps=0,
                models=MODELS, architectures=architectures, smol=dict(source=smol_path, metrics=metrics, by_condition=conditions,
                row_run_ids=dict(Counter(r['run_id'] for r in smol)), rows=inputs,
                generation_config=smol[0]['generation_config'],
                historical_case_bootstrap=stored['overall_accuracy'],
                unit='30 matched groups; not 36 independent trials'),
                qwen=dict(metrics=qmetrics, raw=qraw, training_identity=qsummary['training_identity'],
                paired_step5_family_interval=qsummary['paired_step5_family_interval']),
                neural=dict(validation=neural['validation'], captures=neural['captures'], split_metadata=split_metadata,
                            probes=probe, causal=causal_audit, recomputed_effects=dict(controls)),
                matched_pilot=dict(classification='DIRECTLY COMPARABLE', cases=2, families=1,
                    procedure='full summed teacher-forced candidate LL; no generated answers',
                    rows=common_rows, workers=pilot_reports),
                qwen_replication=dict(validation=qrep, rows=rep_scores,
                    near_tie_gap=rep_scores[0]['summed_loglikelihoods'][0]-rep_scores[0]['summed_loglikelihoods'][2]),
                additional_evidence=extra, research_file_inventory=source_inventory, source_hashes=sources)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    target = parser.parse_args().output.resolve()
    if not target.is_relative_to(ROOT) or target.exists():
        raise ValueError('fresh workspace output required')
    result = audit()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
    print(json.dumps(dict(status=result['status'], smol_cases=result['smol']['metrics']['n'],
                        smol_correct=result['smol']['metrics']['correct'],
                        neural_validation=result['neural']['validation'],
                        matched_pilot=result['matched_pilot']['classification'])))
