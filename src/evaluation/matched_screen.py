"""CPU admission checks for complete matched candidate-scoring analyses.

This module never loads a model or treats attempted forwards as scored examples.
"""
from collections import Counter
import math

from src.evaluation.research_audit import LABELS, balanced_accuracy


def complete_matched_summary(cases, rows, models, *, execution_status):
    """Reject missing/duplicate/misaligned evidence before computing any metric.

    Family counts describe exposed development, not independent confirmation.
    No parameter-invariance or causal statement is inferred from these rows.
    """
    if execution_status != 'PASSED':
        raise ValueError('incomplete execution cannot enter complete-screen metrics')
    expected = {r['sample_id']: r for r in cases}
    if not cases or len(expected) != len(cases):
        raise ValueError('nonempty unique planned cases required')
    if not models or len(set(models)) != len(models):
        raise ValueError('nonempty unique model identities required')
    wanted = {(model, sample) for model in models for sample in expected}
    observed = {}
    for row in rows:
        key = (row['model'], row['sample_id'])
        if key in observed or key not in wanted:
            raise ValueError('duplicate or unplanned model/case identity')
        case = expected[row['sample_id']]
        for name in ('family_id', 'ground_truth_label', 'image_sha256', 'prompt_sha256'):
            if row[name] != case[name]:
                raise ValueError('matched input identity mismatch: ' + name)
        if row.get('is_generated_completion') is not False:
            raise ValueError('candidate scoring cannot be labeled generation')
        if row['candidate_strings'] != [' Yes', ' No', ' Undetermined']:
            raise ValueError('candidate contract changed')
        scores = row['summed_loglikelihoods']
        if len(scores) != 3 or not all(math.isfinite(v) for v in scores):
            raise ValueError('three finite complete-answer scores required')
        if row['argmax_answer'] != LABELS[max(range(3), key=lambda i: scores[i])]:
            raise ValueError('summed-score argmax mismatch')
        observed[key] = row
    if set(observed) != wanted:
        raise ValueError('missing planned model/case scores')
    support = Counter(r['ground_truth_label'] for r in cases)
    if set(support) != set(LABELS):
        raise ValueError('all three ground-truth classes required')
    results = {}
    for model in models:
        ordered = [observed[model, r['sample_id']] for r in cases]
        results[model] = {
            'balanced_accuracy': balanced_accuracy(cases, [r['argmax_answer'] for r in ordered]),
            'per_class_recall': {label: sum(r['argmax_answer'] == label == r['ground_truth_label']
                                          for r in ordered) / support[label] for label in LABELS},
            'correct': sum(r['argmax_answer'] == r['ground_truth_label'] for r in ordered),
        }
    return {'cases_per_model': len(cases), 'families': len({r['family_id'] for r in cases}),
            'models': results, 'scope': 'exposed development; no population or causal inference'}
