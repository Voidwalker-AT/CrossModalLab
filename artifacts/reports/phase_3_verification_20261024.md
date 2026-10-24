# Phase 3 verification and decision record — October 24, 2026

Scope: publication-oriented research audit and one bounded development-only inference pilot. This is a preliminary research update, not evidence of a completed causal study, a successful CE+ES method, independent generalization, or publication acceptance. The October 28 dashboard and immutable evidence archive were preserved.

## Required deliverables

| Deliverable | Observed status / evidence |
|---|---|
| 1. Independent state audit | [Progress audit](../../docs/research_progress_audit.md); initial local main, origin/main and recovery branch all at `38ead1e38da34e5b4529e814e1de2c0d6c0a7581`. Fresh hashes verify 133 historical files, 40 pilot files and 123 neural sources unchanged. |
| 2. Scientific gap analysis | [Publication gaps](../../docs/publication_gap_analysis.md): conditional dataset shortcuts, perceptual naming, parser/answer-length sensitivity, selection reuse, failed causal controls and close prior art. |
| 3. Dataset/split audit | [Dataset audit](../../docs/phase_3_dataset_audit.md); 68 selected legacy cases reconstructed and matched to selected saved identities without opening mixed/final record files. Zero selected cross-split families, exact prompts/images or unordered triples; 11 shared entity names. New 96-case/12-family factorial development set dual-oracle verified. |
| 4. Evaluation protocol | [Protocol](../../docs/phase_3_evaluation_protocol.md), exact IDs, token strings, endpoints, ledger, costs, stop gates and future A–H controls. Final holdout remains closed. |
| 5. Matched model pilot | **INCOMPLETE / STOPPED.** Six Qwen forwards, two cases, finite scores; no new Smol inference. Second-model free-VRAM gate failed. Historical paired data reanalysis is explicitly separate. |
| 6. Intervention plan | [Protocol](../../docs/phase_3_intervention_protocol.md), proposed one-layer/one-token controlled design. No intervention executed. Current pair fails the proposed opposed-margin eligibility screen. |
| 7. CE+ES feasibility | [Assessment](../../docs/phase_3_training_feasibility.md); six historical checkpoint boundaries validate on CPU including optimizer and RNG requirements. No new GPU state restore or optimizer step. Five historical training steps remain five. |
| 8. Novelty collisions | [Audit](../../docs/phase_3_novelty_audit.md), primary-source metadata and disclosed inspection depths. Recent spatial uncertainty, multimodal abstention and representation/action work substantially overlap; novelty remains UNVERIFIED. |
| 9. Manuscript | [Source](../../paper/manuscript/draft.md), [offline HTML](../../paper/manuscript/draft.html); scoped dataset diagnostic, exact partial pilot, historical uncertainty, limitations and 17 references. Eight figures decode offline; desktop table and mobile overflow checks pass. |
| 10. Evidence/claims | [Evidence registry](../../paper/evidence_registry.md) E13–E16 and [claims registry](../../paper/claims_registry.md) C10–C15 appended; historical entries preserved. |
| 11. Reproducible figures | [Three new PNG/SVG figures](../../paper/figures/phase3_20261024/manifest.json) and [tables](../../paper/tables/phase3_20261024/analysis.json). Fresh output directory support prevents overwriting earlier evidence. Tables rebuilt byte-identically and PNGs pixel-identically. |
| 12. Tests | Full ULVO: 316 run, 313 passed, 3 explicit restricted-data skips. Ordinary Python: 316 run, 221 passed, 95 skips (92 optional Torch, 3 restricted-data). Guarded discovery in both environments: zero restricted open attempts. Doctor, portability, compilation and browser checks pass. |
| 13. Git | Focused source/docs/small evidence integration on a dedicated `codex/` branch; recovery branch preserved. Commit and remote identities are recorded in Git and the final handoff, not a self-referential hash in this committed document. No broad staging, shared-history rewrite or force push. |
| 14. December roadmap | [Prioritized research plan](../../docs/phase_3_research_plan.md): protect Oct28 demo, validate controls, obtain isolated inference approval, freeze independent evaluation, screen causal pairs, decide training value, then manuscript/venue freeze. |
| 15. GO decisions | GO for preserved demo and CPU research deliverables. CONDITIONAL GO for a newly approved process-isolated paired inference protocol. NO-GO for current causal/novelty/CE+ES superiority claims, further GPU inference under the stopped run, training and final-holdout access. |

## Hardware gate and partial GPU evidence

After the user freed memory, available physical RAM was **7.296 GiB** and dedicated free VRAM **3,781 MiB** (182 MiB used). CUDA/ULVO doctor and pinned local snapshots passed. No package installation, model download or unrelated process termination was performed.

The fixed new-development family is `p3dev2_2026102403_000`. Both cases query whether the pink hexagon is left of the violet diamond and share image SHA-256 `6802f3851ceae26a66b4c14930158ffbe02867fe86ad8ecd5808626460e4bf17`. The image shows violet diamond left of orange ring; text direction distinguishes the cases. Candidates are the exact strings ` Yes`, ` No`, ` Undetermined` (one, one and two tokens respectively).

| Case suffix | Gold | Complete-answer summed LL: Yes / No / Undetermined | Argmax |
|---|---|---|---|
| `::v1t1q0` | No | -15.4401731491 / -14.6901731491 / -20.4401874542 | No |
| `::v1t0q0` | Undetermined | -14.2001104355 / -15.3251104355 / -20.3251209259 | Yes |

All **625 exact parameter comparisons passed**. No optimizer, backward pass, adapter update, generation, causal replacement or new neural capture occurred. LL/token selects Undetermined in both cases; this flips both rankings and makes the first case wrong. It demonstrates score-rule sensitivity, not a validated alternative or calibrated confidence. Two cases in one family do not support three-class accuracy, uncertainty estimates or mechanism claims.

Peak allocated GPU memory was **1,575.535 MiB**, peak reserved **1,670 MiB**, maximum sampled physical GPU use **1,961 MiB**, and minimum sampled available RAM **5,422.375 MiB**. Sampling was every 0.5 seconds plus event boundaries and can miss transients. Case scoring took 2.358 and 1.020 seconds; Qwen loading 12.884 seconds; full failed run 65.728 seconds including parameter audits.

After Qwen cleanup, 8.126 MiB remained allocated and 1,468 MiB reserved; the next load required 3,500 MiB free dedicated VRAM and failed before loading Smol. CUDA was available: the failing predicate was free VRAM. Residual memory ownership is UNVERIFIED. Process exit returned physical use to baseline. The run exited **1**, with six ledger reservations and two score rows; it was not retried. Remaining allowance was not silently reset or spent. A future separate-process design requires a new approved execution protocol.

Exact evidence: [preregistered plan](../phase3/20261024_v2/pilot/preregistered_plan.json), [scores](../phase3/20261024_v2/pilot/candidate_scores.jsonl), [execution/telemetry/error](../phase3/20261024_v2/pilot/execution.json), [raw output](../phase3/20261024_v2/gpu_execution_output.txt). CPU validation labels these `VERIFIED_PARTIAL_EVIDENCE`, never a complete matched experiment.

## Scientific limits and unchanged artifacts

Selected legacy data admit a role-only lookup balanced accuracy of 2/3 from either modality and deterministic label strata. The new factorial design removes deterministic strata but retains the same 2/3 role-lookup BA because partial premises can exclude a class. Neither is a measured VLM shortcut mechanism. New data are exposed development, with shared renderer/grammar and changed literal shape names; they are not an independent final test. Full mixed-dataset hashes and custodian-level holdout independence were not freshly verified.

Historical paired Smol/Qwen results use the same 18 cases/six families: balanced accuracies 0.444444 and 0.555556. Paired difference 0.111111 has descriptive family-bootstrap 95% interval [-0.111111, 0.277778], 2,000 draws. Both models miss all six underdetermined cases. Repeated development inspection and few families prevent a superiority claim. No historical model output was rewritten.

The r3 Oct28 ZIP SHA-256 remains `24cfd58d51640ed1baa6fb3696dbd0d6f22f4cec13e18c973b1a07f0f2ae1283`. Dashboard/dependency manifests, frozen training code/config and historical records remain unchanged. No TypeScript/build rerun was necessary for this mission because dashboard code and lockfiles have no diff. Partner-machine reproduction, human rehearsal, author approval and venue requirements remain pending; prior local browser verification does not establish these.

## Fresh verification record

Exact logs and their hashes are in [verification](../phase3/20261024_v2/verification/manifest.json). Test commands use checkout-local temporary/cache paths and offline model flags. Literal unittest discovery uses `CROSSMODAL_EXCLUDE_MIXED_DATASETS=1`; guarded discovery also blocks restricted opens and fails unexpected attempts. Three legacy mixed-dataset tests retain their default behavior outside this explicit mode. The guard is per Python process, not OS-wide tracing.

Staging reproduced a byte-hash mismatch caused by Windows Git line-ending normalization of the raw logs. Scoped `.gitattributes` rules preserve the new Phase 3 evidence/tables/figures as exact bytes; staged hashes were rechecked after applying them. Matplotlib SVG path lines retain generated trailing spaces to preserve their recorded identities; the whitespace check excludes only those three SVGs.

- `ulvo project doctor .`: PASS.
- `ulvo run . -- python -m unittest discover -s tests -v`: PASS with 3 explicit skips.
- `python -m unittest discover -s tests -v`: PASS with 95 explicit skips.
- `ulvo run . -- python scripts/verify_phase3_integrity.py` and ordinary Python equivalent: PASS, zero denied-open attempts.
- `ulvo run . -- python scripts/check_portability.py`: PASS.
- `ulvo run . -- python -m compileall -q src scripts tests dashboard paper`: PASS; bytecode redirected to checkout cache.
- Fresh table/figure rebuilds, historical hash checks, partial-pilot identity validation, JSON/Markdown/link checks and manuscript offline browser interaction: PASS.

No new training, final-holdout evaluation, causal patching, dependencies, model weights or optimizer binaries form part of this change.
