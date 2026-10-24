# Independent research progress audit — Phase 3

2026-10-24. Initial Git HEAD and origin/main independently matched `38ead1e38da34e5b4529e814e1de2c0d6c0a7581`. Historical untracked files were preserved. This audit reads source code and raw development artifacts; prior agent narrative is not empirical proof.

## Questions and evidence

| Question | Current support | Status / next decision |
|---|---|---|
| Do compact VLMs make unsupported definite answers? | Earlier matched 18-case/six-family Smol/Qwen raw records have identical image/prompt hashes; both miss all six forks. Fresh CPU parsing/paired intervals are in `paper/tables/phase3_20261024/analysis.json`. | PRELIMINARY behavioral evidence; GO for improved development controls, no population/generalization claim. |
| Does the principal displayed benchmark require both modalities statistically? | All 68 allowed selected records match the dual oracle, but role lookups from either modality reach BA 2/3 on validation and reveal deterministic label strata. | Valid logic does not ensure a shortcut-free generator. NO-GO for treating displayed BA as proof of cross-modal composition. |
| Are train/validation/final partitions independently validated? | Scheduled train/validation families and selected image/prompt hashes are disjoint; eleven entity names overlap. Existing final metadata and contamination history are documented. | Selected-scope family integrity VERIFIED; complete final independence UNVERIFIED because its records stay closed. |
| Do internal representations expose evidence-state information? | Ten historical Smol captures/200 vectors and 108 probe feature files have prior CPU validation; all 123 stored source hashes can be rechecked without new inference. Selected layer-18 probe BA 0.777778 is development-selected. | CORRELATIONAL. No causal use, feature absence or Qwen-specific explanation follows. |
| Has a relation-specific causal computation been isolated? | Phase 2B has 16 surviving rows, ten interventions, 26 planned records missing; unrelated-token effects exceed targeted effects in the completed subset. | NO-GO for a causal mechanism claim; exact replacement/control protocol prepared for separate approval. |
| Does ES improve independent performance? | Five CE+ES steps versus CE-five on six reused families; interval includes zero; CE-50 has an unequal budget and malformed outputs. | UNVERIFIED. No optimizer steps authorized; continuation is lower priority than benchmark validity. |
| Is the proposed representation/action gap or margin training novel? | Fresh primary-source review finds close overlap with Rep2Act/VAD-R, SpatialUncertain and MM-AQA. | No novelty claim. Narrow the contribution and perform full-method comparisons. |

## New authorized pilot: partial, preserved

The user-cleared RAM recheck showed 7.296 GiB available and 3,781 MiB dedicated VRAM free. ULVO doctor/CUDA/local pinned caches passed. The new factorial development set has 12 mixed-pool families, 96 dual-oracle-verified rows, 24 images and literal 2D shape names. It is development, not an independent final test.

The bounded scorer executed **six of twelve allowed forwards**, all Qwen. For `p3dev2_2026102403_000::v1t1q0` (No), summed-LL argmax was No. For `::v1t0q0` (Undetermined), it was Yes. Exact LLs, token IDs, inputs and ledger are preserved under `artifacts/phase3/20261024_v2/pilot/`. All 625 Qwen parameter comparisons passed. LL/token selects Undetermined on both, demonstrating ranking sensitivity, not a validated fix. No free generation, optimizer or causal intervention occurred.

Smol was not loaded: the next pre-load gate found insufficient free dedicated VRAM. End telemetry showed 8.126 MiB allocated but 1,468 MiB reserved; process exit returned physical GPU use to baseline. The residual owner is UNVERIFIED. The run exited 1 and was not automatically retried. Peak allocator use was 1,575.535 MiB, maximum sampled physical GPU use 1,961 MiB, minimum sampled RAM 5,422.375 MiB; no 4-GiB host-RAM breach was sampled. New matched cross-model replication is therefore INCOMPLETE. Existing paired artifacts were reanalyzed separately rather than relabeled as fresh results.

## Engineering versus scientific maturity

The five-view dashboard, original October 28 ZIP and gallery remain unchanged. The renderer-name mismatch (star→diamond), rotating nuisance variables, query-role predictability, parser acceptance of suffixes, and answer-length ranking flips are now explicit competing confounds. GPU scorers' numerical correctness does not settle semantic reasoning. The legacy Qwen feasibility spec and hybrid causal scorer are not validated new experiment paths; the pilot uses pinned loaders and the repaired full-candidate scorer.

Fresh checkpoint inspection verifies step-zero through step-five component hashes, AdamW CPU reloads, RNG payloads, config/source/schedule identities and append-only metrics with no source-run changes. It does not certify a current full mixed-dataset hash, new CUDA RNG restoration or resumed optimization equality. Those qualifications prevent overclaiming verification.

## Highest-value next work

First validate literal perceptual names and modality controls across the counterbalanced development design. Then approve a process-isolated paired inference run with explicit global forward accounting and measured runtime. Next screen a narrowly specified source/recipient pair for causal controls; authorize that experiment separately. Only then decide whether matched multi-seed CE/ES training is worthwhile. The [gap analysis](publication_gap_analysis.md), [research plan](phase_3_research_plan.md), [dataset audit](phase_3_dataset_audit.md) and [novelty audit](phase_3_novelty_audit.md) govern claims; no result here establishes publication acceptance or novelty.
