# Neural evidence inventory and interpretation

Prepared from existing files on 2026-10-24, with zero new model forwards or optimizer steps. `dashboard/assets/neural_evidence_v1.json` is a small curated bundle, not an activation simulator. Revalidation command: `ulvo run . -- python scripts/build_neural_evidence.py`. The builder refuses to overwrite an existing bundle; a subsequent audit must explicitly version it.

## What was validated

All measurements below use historical frozen `HuggingFaceTB/SmolVLM2-2.2B-Instruct@482adb537c021c86670beed01cd58990d01e72e4`, NF4. **They are not the current Qwen checkpoint's activations.** Phase 1A and Phase 1F also use different selections; matching text in IDs does not establish matched experiments.

| Artifact | Measurement / shape / position | CPU validation | Scientific status |
|---|---|---|---|
| Phase 1A `activations_meta_v1.jsonl` and ten referenced `vectors/*.json` | Decoder residual-output rows at layers 3,12,18,23; float32 [2048]; last, first_image, last_image, mean_image, last_text_non_image | 200 vectors finite; exact shape/selector/index checks; recomputed float64 L2 agrees with six-decimal stored float32 norms within explicit floating-point rounding tolerance; stored zero logit drift/passivity fields checked | OBSERVED measurements; interpretation CORRELATED — NOT CAUSAL |
| Phase 1F `features_meta_v1.jsonl`, 108 referenced Phase 1E NPZs | Float32 [24576] concatenating 4 layers × 3 selectors × 2048 | Existing capture-identity utility recomputes all hashes; 108 arrays have expected shape/dtype and finite values; only train/validation split roles accepted | CORRELATED; no model rerun |
| Phase 1F `probe_reliability_summary_v1.json` | Twelve non-PCA layer/selector probe results; 2048 input dimensions; BA and stored family-bootstrap CIs | Stored confusion matrices reproduce BA within six-decimal rounding; 81 train / 27 validation cases, 27 / 9 families; 2,000 bootstrap resamples. Thirteenth PCA result is not plotted as another measurement of the same non-PCA curve | CORRELATED; development-selected layer/selector/L2, optimistic selection risk |
| Phase 2B `intervention_records_v1.jsonl`, `recovery_audit_v1.json` | Sixteen surviving rows; ten interventions at layer 18, last-token [2048] capture; output margin changes | Row count agrees with audit; historical control limitations preserved | UNVERIFIED relation-specific causal explanation; 26 planned rows missing; unrelated-token controls exceed targeted changes |

The complete per-capture shapes, token indices, conditions, method, model identity, source paths and SHA-256 values are in the curated bundle and exported inventory. Source tensor files are kept in place and excluded from the sharing archive. A source-file hash records which file was inspected; it cannot by itself authenticate the original experiment or reconstruct a missing model run.

Phase 1A correctness annotations refer to earlier Phase 0D2 behavior, not a new answer generated during the passive capture. This distinction is shown in the dashboard. Matched-variant plots require the recorded genuinely-paired group, same model/condition and identical image hash. They compare the same selector at its recorded position; text and sequence lengths can differ. They do not isolate correctness or a causal failure component.

## What the interface shows

Architecture Explorer retains its conceptual Qwen pipeline and adds a separately labeled historical Smol Neural Diagnostics section. It displays selectable measured activation norms, stored probe curves with descriptive CIs, source metadata and the failed-control audit. Failure Investigator retains symbolic ground truth and recorded behavior, and explicitly says neural measurements for the selected Qwen case are **NOT MEASURED**. No cross-model attribution is made.

Norm growth does not demonstrate information transfer, reasoning quality, confidence or a failure layer. Probe decodability does not establish model usage. A residual coordinate is not automatically an individual neuron or sparse semantic feature. Existing controls do not support a confirmatory causal heatmap, neuron explanation or unique circuit claim.

No verified per-head attention tensors, separately captured MLP outputs or validated intermediate candidate-answer readouts are provided for these Qwen dashboard cases. They remain NOT MEASURED. The symbolic graph is an oracle, not the Transformer's internal thought process. Synthetic fixtures under tests are excluded from the empirical bundle.

## Optional acquisition plan — NOT EXECUTED

A separately approved follow-up could capture one underdetermined and one matched sufficient **development** case with the existing pinned Qwen NF4 model. Freeze IDs, image/prompt hashes, checkpoint identity, runtime, selector indices and deterministic run identity before acquisition. Use batch size one, read-only decoder hooks at layers 0,7,14,27, CPU-copy only last-text and mean-image statistics, and verify unchanged logits with/without hooks. No attention tensors, training, optimizer, downloads or holdout would be involved.

For hidden size 2048, four float32 layers × two sparse vectors occupy 65,536 bytes (64 KiB) per case, excluding metadata. A full 1500×2048 float32 layer output would be about 11.7 MiB and must not be retained/copied in full. Extra device peak and base-model loading cost are **UNVERIFIED estimates**; prior training resource figures are not capture guarantees. Require at least 4 GiB available host RAM, measure host/device memory during an approved run and abort safely before exceeding the hardware budget. The value is directly matched Qwen representation observations; causation would still require separately designed and approved interventions. This optional work is not needed for October 28 and is not authorized by preparing this plan.
