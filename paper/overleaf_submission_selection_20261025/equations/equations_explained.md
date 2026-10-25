# Exact implemented mathematics

VERIFIED by source inspection, CPU recomputation of all five historical score triplets, and existing unit tests. This mission performed no model forward or optimizer step.

The relevant executed code is `src/training/differentiable_scoring.py`, called by `phase_2i_b_protocol.compose_objective` and `scripts/run_phase_2i_b_es_ablation.py`. Older `losses.py` and `evidence_losses.py` expose different design helpers; their formulas must not replace the executed objective. Copies of the actual source are under `audit/source_records/`.

## Logic

Strict left-of is transitive and asymmetric. Each compatible world is a strict total ordering; the supplied evidence is a partial constraint, not itself a complete world. Reachability checks a forward or reverse query path; exhaustive linear extensions check the query across every compatible ordering. The first yields Yes, reverse yields No, and both possible orderings yield Undetermined. Inconsistent evidence is excluded. The two solvers agree for the selected cases and all 9,216 audited development variants.

## Scoring and supervision

Candidate order is leading-space ` Yes`, ` No`, ` Undetermined`; exact local-tokenizer IDs are in `audit/recomputed.json`. Lengths are 1, 1 and 2. Scores sum causal next-token log-probabilities for every continuation token, using float32 log-softmax. The second Undetermined token is conditioned on the first. The scorer appends matching attention and multimodal type metadata, marks continuation tokens as text, and recomputes multimodal position IDs from the extended sequence. These are teacher-forced complete-answer likelihoods, not probabilities calibrated across classes or spontaneous generations.

CE supervises only the exact gold-answer token span, with all other labels -100; it averages those supervised token losses. Thus CE and a negative summed candidate score differ by candidate token count for a two-token answer, even with identical context. Assistant-prefix/template masking is implemented in `sampling_and_masking.build_supervised_example`; EOS and unrelated prompt/image positions are not supervised. Historical generation evaluations use greedy decoding with up to 16 new tokens and a separately versioned parser.

## Evidence-sufficiency loss

- Insufficient evidence (gold Undetermined): penalize `max(0, 1 + max(score_Yes, score_No) - score_Undetermined)`.
- Sufficient evidence (gold Yes or No): penalize `max(0, 1 + score_Undetermined - score_gold)`.
- The inconsistent sufficient/gold-Undetermined defensive branch returns zero; this is not used by the valid label mapping.

The coefficient is 0.25 and margin is 1.0. The loss is class/sufficiency dependent and explicitly abstention-specific, including an anti-over-abstention branch. It is **not** a generic gold-versus-best-other-class hinge. For insufficient cases the best-definite competitor excludes the gold Undetermined; for sufficient cases the sole competitor is Undetermined, excluding gold but also excluding the other definite class. CE supplies the remaining gold supervision.

No detach occurs between the three candidate model calls and ES. Backward therefore reaches the LoRA tensors through these scores and CE. Historical five-step records contain finite nonzero gradient norms and frozen-parameter comparisons; a fresh real-model gradient replication was not executed here. No optimizer step was performed while checking the saved losses or states.

## Metrics and uncertainty

The equations file defines BA, macro-F1, unsupported definite-answer rate (denominator: six insufficient cases) and incorrect-abstention rate (denominator: twelve sufficient cases). Rates are not silently redefined over all 18 cases. Invalid outputs contribute errors in class support. Exact-format compliance trims outer whitespace only for the check; the raw strings in `audit/raw_predictions.json` remain unchanged, including U+FFFD suffixes.

CE+ES5 minus CE5 is +0.111111 BA. Re-running the existing paired family bootstrap with B=2000 and seed 20261023 gives [0, 0.222222]. All three variants of a sampled family remain together; six families, not eighteen rows, supply clusters. This descriptive interval does not correct development reuse or imply a significant benefit. Historical Smol probe intervals retain their saved seed/B/group metadata; without saved per-example probe predictions they were not freshly resampled. No new interval or p-value is invented.
