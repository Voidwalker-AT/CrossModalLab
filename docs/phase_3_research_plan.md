# Phase 3 roadmap and GO/NO-GO decisions

2026-10-24. This is a prioritized research plan, not approval for extra GPU forwards, final-holdout access, installations or training. The first allowance stopped at six Qwen forwards after a model-switch VRAM gate failure; do not silently spend a reset allowance.

| Experiment / task | Decision now | Cost and admission condition |
|---|---|---|
| Allowed-data oracle, split and shortcut audit | GO; completed scoped CPU pass | 68 exact selected records; no mixed/final record file opened. New tests preserve evidence. |
| Counterbalanced development preparation | GO; completed CPU generation | 12 families/96 records/24 images; literal 2D names and recorded seed. New data are exposed development. |
| Initial two-model candidate pilot | PARTIAL / STOPPED | Six Qwen forwards complete, finite and weight-invariant; model-switch free-VRAM gate failed before Smol. No automatic retry. |
| Process-isolated completion/replication | CONDITIONAL GO, approval required after this failure | Fresh protocol must reconcile the existing six attempts, isolate model workers and wait for full process exit/free VRAM; pre-register whether remaining six or a new allocation is requested. No result-dependent case changes. |
| Full A–H behavioral battery | NO-GO under current allowance | Full factorial alone is 576 candidate forwards before controls and generation. Calculate observed per-model costs after isolated smoke and obtain approval. |
| Passive Qwen neural capture | CONDITIONAL GO for planning only | No per-case Qwen capture exists; count any passive forwards in a new approved budget and validate hooks/alignment/no output perturbation. |
| Narrow causal No/U test | NO-GO until separate approval and baseline eligibility | Proposed one-layer/one-token/one-family 20-forward design in the intervention protocol. Current Qwen pair has positive No-minus-U margins in both states, so it fails the proposed opposed-margin screening criterion. |
| Resume CE+ES steps 6–50 | NO-GO | No optimizer authorization; benchmark and comparison validity first. Preserve existing checkpoints/identity. |
| Final-holdout evaluation | NO-GO | Separate one-time approval after protocol/data-policy freeze and exposure audit; never use its labels to tune. |
| Literature, paper, figures and CPU tests | GO | Continue from verified artifacts; preserve registries and old releases. |

## Milestones

- **October 24–28:** freeze the working university dashboard/package; rehearse with a human presenter and partner. Show preliminary evidence and scientific caveats. This mission does not replace the verified gallery.
- **October 29–November 5:** manually audit rendering semantics, implement separated model-worker lifecycle on CPU, obtain bounded-pilot approval, then collect premise/format/counterfactual diagnostics with explicit compute accounting.
- **November 6–12:** lock the behavioral endpoints and novel-data protocol. Screen mechanistic pairs on dedicated development data. Compare full methods of the closest prior art. Decide whether a benchmark/control contribution is defensible.
- **November 13–20:** if approved and feasible, execute the narrow causal control experiment and then a family-disjoint extension. Failed controls are results to report. Do not launch an unrestricted layer/head search.
- **November 21–26:** only if behavioral validity holds and resources/approval permit, run a matched CE/ES continuation or replication. Otherwise prioritize independent reproduction and a coherent negative-result paper.
- **November 27–December 1:** freeze tables/claims, verify all references and licenses, obtain author agreement and choose a legitimate venue whose actual requirements fit. Run a separately approved final evaluation only after all tuning decisions are locked. Submission is a target; acceptance is unpromised.

## Research value per GPU-hour

The first priority is factorial/modality controls: they distinguish dataset predictability, perception and joint composition with existing loaders. The second is one valid targeted causal comparison with strong negative controls. Extra loss training on the same six-family slice is lower value because it does not resolve those confounds. Use the new measured Qwen pair timings plus load/parameter-audit overhead in the next estimate; do not extrapolate from fast generation to slow Smol scoring. No cloud expense, model download or dependency change is proposed.

See [dataset audit](phase_3_dataset_audit.md), [evaluation protocol](phase_3_evaluation_protocol.md), [intervention protocol](phase_3_intervention_protocol.md), [training assessment](phase_3_training_feasibility.md), [gap analysis](publication_gap_analysis.md) and [novelty review](phase_3_novelty_audit.md).
