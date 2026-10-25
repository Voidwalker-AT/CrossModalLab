# Claim-specific scientific audit

Sources below are paths in the repository at audit base commit 532e2276e585881a7d2a7f032bc32e09a31d0ec1, or packaged `dual_model_audit.json` sections. CPU recomputation verifies stored arithmetic/identity; it does not reenact model behavior. Existing `paper/claims_registry.md` and `paper/evidence_registry.md` are preserved, not superseded silently.

| Claim | Classification | Source artifact | Allowed wording / missing evidence |
|---|---|---|---|
| Chain/reverse/fork distinguish Yes/No/U under consistent strict-order premises | VERIFIED formal task | equations/equations_verified.tex; src/symbolic/solver.py; dashboard/assets/development_cases.json | Task semantics, independently checked ground truth; not neural reasoning |
| Qwen generations differ across the recorded conditions | VERIFIED descriptive historical counts | artifacts/training/phase_2i_a/phase_2i_a_report_v1.json; phase_2i_b run baseline/evaluation_step_0005; packaged qwen.metrics | Report exact six-condition counts on 18 exposed cases/six families |
| CE50 has 12/18 semantic correctness but zero exact-label compliance | VERIFIED CPU recomputation | original audit/raw_predictions.json; qwen.metrics.ce50 | Parser-dependent semantic labels; not 12 compliant answers |
| CE+ES pilot executed five updates | VERIFIED historical feasibility | phase_2i_b run train_metrics_append.jsonl, state/checkpoint metadata; original audit/recomputed.json | Paused at step five, not an executed 50-step CE+ES experiment |
| CE+ES improves Qwen performance | PRELIMINARY descriptive difference only | qwen.paired_step5_family_interval; ce5 7/18 versus es5 9/18 | Stored descriptive six-family interval [0, .222222] includes zero; selection uncorrected. No superiority claim |
| CE50 is a fair equal-compute comparison to ES5 | UNSUPPORTED | selected conditions / training budgets | 50 versus five updates differ; retain only as separate reference |
| Smol produces definite answers on all six historical fork questions | VERIFIED bounded behavior | artifacts/evaluations/phase_0d2/behavioral_predictions_v1.jsonl; smol_raw_behavior.json | Five Yes, one No; none correct. Do not claim a universal negation-only failure |
| Smol behavior is equivalent to Qwen behavior on the same test | UNSUPPORTED except tiny matched pilot | audit/cross_model_comparability.md | Different larger batteries; no pooled leaderboard |
| Two candidate pilot cases use common task input/scoring across models | VERIFIED historical protocol match | artifacts/publication/20261025/pilot/{smol,qwen}/; matched_pilot | Preferences on two dependent cases, not broad model accuracy |
| Recent Qwen frozen candidate argmax matches three gold labels | NEWLY VERIFIED prior execution, current CPU check | qwen_replication_20261025_464531de/; qwen_replication | One family/nine calls; near-tie, no generated-answer replication |
| Candidate softmax or score gap is calibrated confidence | UNSUPPORTED | raw candidate-score records | Ranking scores only; no calibration test or uncertainty calibration study |
| Smol layer18 last-token features permit selected probe decoding | CORRELATIONAL FINDING | artifacts/probes/phase_1f/probe_reliability_summary_v1.json | BA .777778, stored nine-family CI; selected layer/selector/L2, no corrected significance |
| Probe labels reveal the representation used by the VLM | UNSUPPORTED | same probe source; Phase2B controls | External accessibility does not establish causal usage; failure does not establish absence |
| Passive capture artifacts have finite 2048-D rows at four layers | VERIFIED CPU integrity + historical passivity receipt | instrumentation/phase_1a/; neural.validation | Ten captures/200 rows; not individual neurons, attention heads or Qwen measurements |
| A relation-specific causal circuit was found | UNSUPPORTED / incomplete experiment | causal/phase_2b/intervention_records_v1.jsonl; recovery_audit_v1.json | Controls exceed targets; missing/deleted records and protocol/cap ambiguity. No circuit or restored reasoning |
| 9216 variants constitute independent benchmark examples | UNSUPPORTED independence claim | artifacts/publication/20261025/benchmark/manifest.json; original tables/b_dataset_units.csv | Six dependent families, 72 images; design diagnostics only |
| CrossModalLab invents QLoRA / sufficiency / abstention / the representation-action gap | UNSUPPORTED novelty claim | LoRA, W8, W4/W5/W14/W15; implemented ES source | Existing methods/concepts; scoped implementation/integration contribution only |
| Frozen final holdout or real-world transfer validates the findings | MISSING EVIDENCE | no authorized evaluation in this mission | No independent generalization, real-world transfer or broad robustness claim |
| Dashboard provides causal neuron-level explanations | UNSUPPORTED | dashboard/assets/neural_evidence_v1.json missing list | Evidence inspection and symbolic explanations only; no causal neuron measurement |

## Essential disclosure points

Smol acquisition roles (105/3) differ from the Phase1F evaluation split (81/27): 30 row-role disagreements, disclosed without modifying historical metadata. The evaluation code uses disjoint family lists. Phase0D2 rows span three resumed run IDs. The passive-capture summary is an earlier invocation, while the append-only metadata contains ten rows. The causal recovery audit's 26 missing items include baselines; they are not 26 known failed patches. The nominal patch grid exceeds the execution specification's stated patch cap. These are provenance/design limits, not evidence that all stored results are invalid or that a causal mechanism is established.

Stored probe CIs and the historical Smol case-bootstrap interval are not newly re-estimated in this audit. Report them as historical descriptive intervals with their actual units. No probability calibration, multiple-selection correction, independent validation cohort, cross-model neural comparison or causal uniqueness test is supplied.

## Strongest defensible contribution

A reproducible research system that makes the distinction between false and unresolved cross-modal relations explicit, checks logical ground truth, links original stimuli to recorded behavior and separates output quality, adaptation feasibility and exploratory representation evidence. This is a defensible preliminary university contribution. Claims of broad novelty, generalization, causal discovery or an effective corrective method require additional evidence and are excluded.
