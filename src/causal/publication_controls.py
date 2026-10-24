"""CPU-only alignment and intervention admission; no GPU execution interface."""
from src.causal.interventions import apply_residual_replacement, validate_intervention_spec


def aligned_span(source_ids, target_ids, anchor):
    """Locate one exact declared anchor in each stream; never guess an index.

    Tokenizer/model identities must be equal before using this helper. A match
    validates token identity only, not equivalent hidden-state semantics.
    """
    anchor=tuple(anchor)
    if not anchor:raise ValueError('explicit nonempty anchor required')
    positions=[]
    for sequence in (source_ids,target_ids):
        matches=[i for i in range(len(sequence)-len(anchor)+1) if tuple(sequence[i:i+len(anchor)])==anchor]
        if len(matches)!=1:raise ValueError('anchor missing or ambiguous')
        positions.append((matches[0],matches[0]+len(anchor)))
    return tuple(positions)


def controlled_replacement(target, source, *, source_ids, target_ids, anchor, spec):
    """Replace only the last coordinate of the exact aligned token span.

    Safe mock infrastructure, not an authorized live VLM hook or causal result.
    """
    issues=validate_intervention_spec(spec)
    if issues:raise ValueError('; '.join(issues))
    source_span,target_span=aligned_span(source_ids,target_ids,anchor)
    source_position=source_span[1]-1;target_position=target_span[1]-1
    if spec.token_position!=target_position:raise ValueError('declared patch position disagrees with aligned anchor')
    if target.ndim!=2 or source.ndim!=2 or len(target_ids)!=target.shape[0] or len(source_ids)!=source.shape[0] or target.shape[1]!=source.shape[1]:
        raise ValueError('activation/token/hidden shape mismatch')
    return apply_residual_replacement(target,None if spec.direction=='no_op' else source[source_position],token_position=target_position,spec=spec)


def eligible_pair(source_scores, target_scores):
    """Predefined opposed No-minus-Undetermined margins, without unstable ratios."""
    import math
    if any(len(values)!=3 or not all(math.isfinite(v) for v in values) for values in (source_scores,target_scores)):
        raise ValueError('three finite class scores required')
    source=source_scores[1]-source_scores[2]
    target=target_scores[1]-target_scores[2]
    return {'source_no_minus_undetermined':source,'target_no_minus_undetermined':target,
            'eligible':source>0 and target<0,'interpretation':'Screening criterion only; no intervention executed.'}
