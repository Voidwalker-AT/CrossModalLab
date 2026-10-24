# Phase 3 evaluation protocol v1

Status: development protocol prepared before the bounded candidate pilot; further inference requires budget approval. Partition identities are in `artifacts/phase3/20261024_v2/evaluation_manifest.json`. Preserve October 28 artifacts and the closed final-holdout record file.

## Endpoints and matched inputs

Use exact pinned SmolVLM2 and Qwen3-VL loaders, offline NF4/double quantization/BF16, one model at a time, batch one, eval/inference mode. Preserve image bytes, text/query semantics and families across models; save each processor's serialized token IDs and multimodal shapes. The generic feasibility `ModelSpec` still carries an unpinned Qwen marker; it is not the experiment identity. Use the loader's exact revision and returned metadata.

The first pilot fixes family `p3dev2_2026102403_000`, cases `v1t1q0` (No) and `v1t0q0` (Undetermined). They have identical image SHA-256 `6802f3851ceae26a66b4c14930158ffbe02867fe86ad8ecd5808626460e4bf17`, identical query and opposite textual premise directions. The corrected prompt names the visible violet diamond and orange ring; the pink hexagon is supplied by text. Source evidence is the factorial manifest/records, never a final-holdout record.

Score the three exact candidates ` Yes`, ` No`, ` Undetermined` under teacher forcing, summing every answer token. The repaired scorer extends modality/mask metadata and recomputes Qwen multimodal positions. Record token IDs, summed LLs, LL/token as a prespecified sensitivity diagnostic, and the argmax label. No candidate softmax is called calibrated confidence. No free generation is performed, so raw-generation format compliance, generation-vs-scoring agreement and spontaneous abstention are **NOT MEASURED** in this pilot.

## Hard execution limits

Exactly two models × two cases × three candidates = at most **12 top-level model forwards**, including failed forward attempts. This is not 12 generations: autoregressive generation could exceed the allowance. A persistent pre-forward ledger reserves each attempt; it refuses a thirteenth attempt or a silent reset. A ten-minute total wall budget is an operational cap, not an empirical performance criterion. Failed results remain; no altered-parameter retry follows a failure.

Preflight: ULVO doctor, CUDA, pinned local snapshots, available physical RAM ≥4 GiB, at least 3,500 MiB free dedicated VRAM before loading, at least 4 GiB disk reserve and a competing-process inventory. These VRAM/disk reserves are operational gates derived from prior ~3.33 GiB allocator use and exact parameter snapshots. They are not accuracy thresholds. Stop at sampled host RAM <4 GiB, sampled physical GPU use >3,900 MiB, nonfinite scores, changed dataset identity or unexpected parameter changes. Telemetry samples every 0.5 s plus forward/case boundaries; sampling can miss transients. Preserve load failures, logs and the attempt ledger. Never use shared system RAM as extra dedicated VRAM.

A before/after exact parameter audit verifies observed frozen weights. No optimizer is created, no backward occurs, no adapter is loaded, and no causal replacement is applied. Model-specific inputs-merger repair is the historical approved Smol implementation, not an experimental intervention.

## Runtime estimate made before execution

Historical Phase 1F Smol capture forwards have mean 16.20 s and observed range 4.65–76.94 s (`artifacts/probes/phase_1f/runtime_latency_v1.json`). Six such forwards suggest ~97 s central and ~462 s using the observed maximum, before load/audit overhead; this is a projection from another workload, not a guarantee. Historical Qwen paired generation averages ~0.43 s/case and cannot directly predict full teacher-forced latency. The script enforces the wall cap rather than launching an unpriced campaign. Save observed pilot timings for the next estimate.

## Larger development battery: not executed under this allowance

| Stratum | Controlled question | Primary endpoint |
|---|---|---|
| A visual-only | Query the visible pair, forward/reverse; no textual relation | Binary premise recall, color/shape confusion |
| B text-only | Query the stated pair using text only and a blank-image control | Binary relation comprehension |
| C forward | Full chain and both query directions | Three-class BA/recall, strict format |
| D reverse | Reverse both premise directions with matched layout | Directional consistency |
| E forks | Both fork orientations and both queries | Unsupported definite rate and correct logical abstention |
| F image swap | Same text/query, reverse visual relation using fixed geometry | Paired label-consistent responses |
| G text swap | Exact same image/query, reverse bridge premise | Paired label-consistent responses |
| H prompt/choices | Reverse presentation order; rotate label order; retain semantics | Within-family robustness and exact-format sensitivity |

For a full factorial paired evaluation, even 96 cases × two models × three candidates costs 576 forwards before controls/generation. **NO-GO without a new measured estimate and approval.** Do not silently expand the current pilot to this battery.

## Statistics for an approved expanded evaluation

Report macro-F1, all class recalls, invalid rate, exact-format compliance, unsupported definite rate on insufficient cases, over-abstention on sufficient cases, and all-required-member counterfactual consistency. Missing/failed predictions count as failures in the preregistered primary analysis, with completed-only sensitivity separately labeled. Use family-clustered descriptive bootstrap intervals with B=2,000 and a fixed analysis seed; compare models on identical resampled families. Repeated development inspection and narrow generation are not corrected by those intervals.

One family and two labels do not support three-class BA, population CIs, a superiority test or a calibrated rejection curve. For future confirmatory work, freeze a practically meaningful effect target and simulation-based family count before data access, then register the primary contrast and Holm correction for prespecified secondary comparisons. Do not optimize thresholds/prompts against final labels. Report likelihood-length rank flips descriptively; never change the frozen ES objective retroactively.
