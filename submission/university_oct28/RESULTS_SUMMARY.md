# Verified results for the university demonstration

**PRELIMINARY development findings.** This summary reorganizes existing evidence; no new inference, training, patching or final-holdout evaluation was performed. Different slices and procedures are kept separate.

## Main dashboard: Qwen checkpoint comparison

`Qwen/Qwen3-VL-2B-Instruct` @ `89644892e4d85e24eaac8bacfd4f463576704203`, four-bit NF4. Every row uses the same **18 development validation cases, six families, six cases per class**, one training seed and repeated inspection. CE is answer cross-entropy; ES adds the frozen evidence-sufficiency margin. Steps are optimizer updates, not epochs.

| Standard-prompt condition | Parsed correct | BA | Yes / No / U recall | Exact-format outputs |
|---|---:|---:|---|---:|
| Frozen | 7/18 | 38.9% | 1/6, 4/6, 2/6 | 18/18 |
| CE step five | 7/18 | 38.9% | 3/6, 4/6, 0/6 | 0/18 |
| CE+ES step five | 9/18 | 50.0% | 3/6, 4/6, 2/6 | 18/18 |
| CE step 50 | 12/18 | 66.7% | 6/6, 6/6, 0/6 | 0/18 |

Balanced accuracy (BA) averages class recalls and equals accuracy here because class sizes are equal. CE+ES5 has two correct abstentions and four incorrect abstentions on twelve sufficient cases. The paired CE+ES5-minus-CE5 development difference is 0.1111; descriptive family-bootstrap 95% interval **[0, 0.2222]**, B=2,000, six clusters. It includes zero and does not establish superiority or repair selection bias. CE50 has a different budget and misses all six forks. No CE+ES50 result exists.

The additional abstention-prompt conditions (frozen and CE+ES5) each have BA 44.4%, U recall 2/6 and exact compliance 18/18. They reuse the same cases, not new independent samples.

Sources: [dashboard CSV](../../paper/tables/submission_metrics.csv), [development analysis](../../paper/tables/development_summary.json), [CE report](../../artifacts/training/phase_2i_a/phase_2i_a_report_v1.json), [CE+ES evaluation](../../artifacts/training/phase_2i_b/runs/56e277e421e442b08d33b107fc934494/evaluation_step_0005.json). CE checkpoints belong to Phase 2I-A. CE+ES run `56e277e421e442b08d33b107fc934494` is paused after five updates; original config/schedule/source identities remain in place. Dashboard metrics are recomputed from stored raw answers.

## Parser correctness versus exact formatting

Despite its name, historical `strict_answer_v1` accepts some valid label prefixes followed by malformed suffixes. Parsed correctness is therefore not clean generation. Exact compliance requires stripped output to equal Yes, No or Undetermined, with nothing else. On the featured fork, CE5 is `No` followed by fifteen U+FFFD replacement characters; CE50 has a leading space, `No` and fifteen U+FFFD characters. Both parse as No but are malformed and logically wrong here. Original strings remain unchanged. On a contradicted case, a correct extracted No can still fail the format rule.

## Separate evidence

- **Historical Smol:** `HuggingFaceTB/SmolVLM2-2.2B-Instruct` @ `482adb537c021c86670beed01cd58990d01e72e4`, NF4. [Neural inventory](../../dashboard/NEURAL_EVIDENCE.md) distinguishes measured captures/development-selected probes from Qwen. Decodability is correlational, not causal usage.
- **Successful October 25 likelihood pilot:** two fixed No/U cases, one exposed family, 12 additional candidate forwards across both pinned models. Smol selects Yes/Yes; Qwen No/Yes. These are likelihood rankings, not generated completions or calibrated confidence. No three-class BA/CI is justified. [Stored analysis](../../artifacts/publication/20261025/analysis.json).
- **Failed 108-forward screen:** one attempted Smol forward, zero completed score rows; Qwen never started, no retry. Exact RAM/VRAM trigger and post-termination parameter invariance are unverified. No performance estimate follows. [Failure audit](../../artifacts/publication/20261025/screen108/failure_audit.json).
- **Controlled development generator:** 9,216 variants, six dependent families, 72 distinct images. Oracle/role diagnostics are design properties, not 9,216 independent empirical successes or completed VLM evaluations. [Manifest](../../artifacts/publication/20261025/benchmark/manifest.json).

Do not pool these slices, budgets or scoring definitions. The [evidence registry](../../paper/evidence_registry.md), [claims registry](../../paper/claims_registry.md) and [paper](../../paper/manuscript/draft.md) retain the scientific record. Generalization, real-world transfer, causal explanation and corrective-method superiority remain unverified.
