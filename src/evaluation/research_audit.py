"""Development-only integrity and shortcut checks. No model or holdout access."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from src.symbolic.relations import Premise, Query
from src.symbolic.solver import SymbolicSolver

LABELS = ("Yes", "No", "Undetermined")


def verified_label(row):
    if row.get("split") not in {"train", "validation", "development"}:
        raise ValueError("restricted or unknown split; development audit refuses record")
    if row.get("is_synthetic_demo") is not False:
        raise ValueError("demo or unspecified evidence cannot enter research audit")
    result = SymbolicSolver.solve([Premise(**p) for p in row["premises"]],
                                  Query(**row["query"]), verify_with_oracle=True)
    label = {"entailed": "Yes", "contradicted": "No", "underdetermined": "Undetermined"}.get(result.epistemic_status.value)
    if not result.algorithm_agreement or label != row["ground_truth_label"]:
        raise ValueError("symbolic label/oracle mismatch")
    return label


def role_key(row, modality):
    """An oracle-parsed single-modality feature; never a model capability claim."""
    q = row["query"]
    def role(entity):
        return "query_subject" if entity == q["subject"] else "query_object" if entity == q["object"] else "bridge"
    return tuple((role(p["subject"]), role(p["object"])) for p in row["premises"] if p["modality"] == modality)


def balanced_accuracy(rows, predictions):
    support = Counter(r["ground_truth_label"] for r in rows)
    if set(support) != set(LABELS) or len(predictions) != len(rows):
        raise ValueError("three supported classes and one prediction per record required")
    return sum(sum(r["ground_truth_label"] == label == p for r,p in zip(rows,predictions)) / support[label] for label in LABELS) / 3


def lookup_audit(train, evaluation, modality):
    if {r["family_id"] for r in train} & {r["family_id"] for r in evaluation}:
        raise ValueError("family leakage in shortcut fit/evaluation")
    counts = defaultdict(Counter)
    for row in train:
        verified_label(row)
        counts[role_key(row, modality)][row["ground_truth_label"]] += 1
    majority = lambda c: max(LABELS, key=lambda label: (c[label], -LABELS.index(label)))
    fallback = majority(Counter(r["ground_truth_label"] for r in train))
    predictions = [majority(counts[role_key(r,modality)]) if role_key(r,modality) in counts else fallback for r in evaluation]
    return {"balanced_accuracy": balanced_accuracy(evaluation,predictions), "predictions": predictions,
            "train_cases":len(train),"evaluation_cases":len(evaluation),
            "feature":"oracle-parsed single-modality entity roles; no gold label in feature",
            "interpretation":"Dataset predictability diagnostic, not a VLM result or proof of model shortcut use"}


def audit_records(rows):
    ids=[r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):raise ValueError("duplicate sample identity")
    split_by_family=defaultdict(set)
    for row in rows:
        verified_label(row)
        split_by_family[row["family_id"]].add(row["split"])
    if any(len(splits)>1 for splits in split_by_family.values()):raise ValueError("family crosses splits")
    single_modality_definite={modality:sum(SymbolicSolver.solve(
        [Premise(**p)for p in r['premises']if p['modality']==modality],Query(**r['query']),
        verify_with_oracle=True).epistemic_status.value in {'entailed','contradicted'} for r in rows)
        for modality in ('visual','textual')}
    return {"cases":len(rows),"families":len(split_by_family),"labels":dict(Counter(r["ground_truth_label"] for r in rows)),
            "dual_oracle_verified":True,"family_split_disjoint":True,
            "single_modality_logically_definite_counts":single_modality_definite,
            "unique_prompt_hashes":len({r["prompt_sha256"] for r in rows}),
            "unique_image_hashes":len({r["image_sha256"] for r in rows})}


def role_ambiguity(rows,modality):
    """A counterbalanced input must not deterministically reveal a gold label.

    Above-chance BA is not itself leakage: one relational premise can rule out
    one class without determining the other two. Report both predictability
    and deterministic strata rather than promising chance-level performance.
    """
    groups=defaultdict(Counter)
    for row in rows:groups[role_key(row,modality)][row['ground_truth_label']]+=1
    return {'groups':len(groups),'deterministic_groups':sum(len(c)==1 for c in groups.values()),
            'maximum_label_purity':max(max(c.values())/sum(c.values())for c in groups.values()),
            'label_sets':sorted([sorted(c)for c in groups.values()])}


def verify_partial_pilot(path):
    """A failed cross-model run may contain valid rows, never a completed study."""
    path=Path(path)
    execution=json.loads((path/'execution.json').read_text())
    plan=json.loads((path/'preregistered_plan.json').read_text())
    attempts=[json.loads(p.read_text())for p in sorted((path/'forward_ledger').glob('forward_*.json'))]
    records=[json.loads(line)for line in (path/'candidate_scores.jsonl').read_text().splitlines()]
    if [a['attempt']for a in attempts]!=list(range(1,len(attempts)+1)) or len(attempts)>12 or len(attempts)!=execution['forward_attempts']:
        raise ValueError('invalid forward ledger accounting')
    cases={r['sample_id']:r for r in plan['selected_cases']}
    seen=set()
    for row in records:
        key=(row['model'],row['sample_id'])
        if key in seen:raise ValueError('duplicate scored case')
        seen.add(key)
        case=cases[row['sample_id']]
        verified_label(case)
        if any(row[k]!=case[k]for k in ('image_sha256','prompt_sha256','ground_truth_label')) or row['is_generated_completion'] is not False:
            raise ValueError('scored identity or generation attribution mismatch')
        scores=row['summed_loglikelihoods']
        if len(scores)!=3 or not all(math.isfinite(v)for v in scores):raise ValueError('nonfinite or incomplete candidate vector')
        if row['argmax_answer']!=row['candidate_strings'][max(range(3),key=lambda i:scores[i])].strip():raise ValueError('incorrect candidate argmax')
        matching=[a for a in attempts if a['model']==row['model'] and a['sample_id']==row['sample_id']]
        if [a['candidate_index']for a in matching]!=[0,1,2]:raise ValueError('unaccounted candidate forward')
    if execution['new_optimizer_steps']!=0 or execution['holdout_accessed'] or execution['causal_patching']:raise ValueError('pilot exceeded scientific scope')
    telemetry=execution['telemetry']
    if telemetry['sampling_errors'] or telemetry['minimum_sampled_host_available_mib']<4096 or telemetry['maximum_sampled_nvidia_used_mib']>3900:
        raise ValueError('unsafe or unverifiable resource samples')
    if any(not all(r['passed']for r in model['parameter_audit']['parameters'])for model in execution['models']):raise ValueError('parameter invariance failed')
    return {'status':'VERIFIED_PARTIAL_EVIDENCE' if execution['status']!='PASSED' else 'VERIFIED_EXECUTION',
            'execution_status':execution['status'],'forward_attempts':len(attempts),'valid_scored_rows':len(records),
            'complete_cross_model':len(seen)==4 and {r['model']for r in records}=={'qwen','smol'} and execution['status']=='PASSED',
            'limit':'This validates recorded rows and failure accounting, not successful cross-model replication.'}
