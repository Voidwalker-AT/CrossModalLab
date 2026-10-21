# Phase 0D.2 Behavioral Benchmark Report

**CrossModalLab — Academic Research Report**  
**Phase:** 0D.2 (Reproducible Behavioral Benchmark & Failure Evidence Collection)  
**Date:** 2026-10-21  
**Target Model:** `HuggingFaceTB/SmolVLM2-2.2B-Instruct` @ `482adb537c021c86670beed01cd58990d01e72e4`  
**Quantization:** 4-bit NF4 (bitsandbytes)  
**Host:** NVIDIA GeForce RTX 2050 (4 GB), Windows 11  
**Manifest:** `configs/phase_0d2/behavioral_manifest_v1.json`  
**Manifest SHA-256:** `b13c4bccb560d830c822409703ba272e1cded9665dd826810da710453a8a46da`

---

## 1. Scope and non-claims

This is a **fixed 36-case controlled pilot**, not a representative real-world benchmark.  
Behavioral errors are **not** evidence of specific internal neural mechanisms.  
Candidate-normalized softmax scores are **uncalibrated ranking scores**, not confidence.

## 2. Execution integrity

| Check | Result |
|---|---|
| Process still running at recovery | No (stopped; GPU idle) |
| Prior valid records recovered | 17 |
| Newly completed during recovery | 19 |
| Final valid `ok` records | **36 / 36** |
| Duplicate ok sample IDs | 0 |
| Truncated JSONL lines | 0 |
| Manifest hash consistency | All records match frozen SHA |
| Model revision consistency | All `482adb537c021c86670beed01cd58990d01e72e4` |
| Quantization consistency | All NF4 4-bit |
| Image hash mismatches vs manifest | 0 |
| Ground-truth mismatches vs manifest | 0 |
| Non-finite candidate scores | 0 |

**Interruption cause:** Unknown. Agent session ended during the first long GPU run after model load and partial evaluation. No confirmed OOM or Python exception in recovered artifacts. Last recovered sample was mid-`reverse_chain` block (`p0d2_ctrl_reverse_chain_0065`).

**Processor warning:**  
`Kwargs passed to processor.__call__ have to be in processor_kwargs dict, not in **kwargs`  
appeared during successful runs. Diagnosis: transformers 5.x compatibility warning related to processor/generate kwarg plumbing. Empirical checks: image inputs still produce finite logits, merger audit (Phase 0D.1) remains PASS, all 36 records completed with valid candidate scores. **No preprocessing change was made mid-run.** Warning was reduced via `TRANSFORMERS_VERBOSITY=error` for resume only (does not alter model inputs).

## 3. Behavioral results (independently recomputed from JSONL)

| Metric | Value |
|---|---|
| Overall accuracy (95% bootstrap CI, B=2000, seed=42) | **0.500** [0.333, 0.667] |
| Balanced accuracy | 0.426 |
| Forward chain accuracy | 0.667 (4/6) |
| Reverse chain accuracy | 0.500 (3/6) |
| Underdetermined abstention accuracy | **0.000** (0/6) |
| Underdetermined definite-answer rate | **1.000** (6/6 definite Yes/No) |
| Directional-reversal control accuracy | 0.333 (2/6) |
| Matched counterfactual pair joint correctness | **0.000** (0/6 groups) |
| Visual-premise control accuracy | 1.000 (6/6) |
| Text-premise control accuracy | 0.500 (3/6) |
| Mean latency | 14.92 s (min 0.18 s text-only; max 22.43 s) |
| Peak VRAM | 3330.35 MiB (visual path); ~1841 MiB text-only |

### Confusion matrix (true × predicted)

|  | Yes | No | Undetermined |
|---|---:|---:|---:|
| **Yes** (entailed) | 10 | 2 | 0 |
| **No** (contradicted) | 10 | 8 | 0 |
| **Underdetermined** | 5 | 1 | 0 |

**Observed answer bias (behavioral only):** Greedy completions never produced “Undetermined” on the 6 fork cases; definite answers dominated. This is an abstention failure on this controlled subset, not a calibrated overconfidence claim.

## 4. Representative failures (full distribution preserved)

Raw evidence: `artifacts/evaluations/phase_0d2/behavioral_predictions_v1.jsonl`  
Incorrect count: **18 / 36**

Representative examples (not cherry-picked exclusions):

1. **Fork abstention failure:** `p0d2_ctrl_underdetermined_fork_0073` GT=Undetermined → raw `" Yes."`
2. **Matched reversal failure:** parent forward-chain correct Yes, paired reversal control still `" Yes."` (e.g. `mg_ctrl_forward_chain_0019`)
3. **Text-only definite bias:** `p0d2_ctrl_unimodal_text_0164` GT=No → `" Yes.\n\nThe statement..."`

## 5. Comparison with earlier pilots (cautious)

- Consistent with Phase 0D.1 qualitative findings: weak abstention on underdetermined forks; language-prior / affirmative bias on degraded evidence.  
- Visual-premise-only controls were **strong** (6/6) in this pilot, so failure is **not** a blanket “cannot see left-of.”  
- Composition and epistemic calibration remain the weaker surface on this 36-case subset.  
- Sample size is small; CIs are wide; matched pairs are dependent.

## 6. Artifacts

| Path | Role |
|---|---|
| `configs/phase_0d2/behavioral_manifest_v1.json` | Frozen 36-case manifest |
| `artifacts/evaluations/phase_0d2/behavioral_predictions_v1.jsonl` | Append-only raw predictions |
| `artifacts/evaluations/phase_0d2/behavioral_metrics_v1.json` | Metrics summary |
| `artifacts/evaluations/phase_0d2/run_metadata_v1.json` | Run metadata |
| `artifacts/evaluations/phase_0d2/progress_v1.log` | Resume/progress log |
| `artifacts/evaluations/phase_0d2_smoke/` | 3-case smoke (separate) |

## 7. Recommendation for mechanistic probing

**CONDITIONAL GO** for carefully scoped mechanistic follow-up on:

1. **Fork abstention circuit** (underdetermined vs entailed matched entities)  
2. **Cross-modal composition failure** (forward-chain errors + reversal inconsistency)

with strict caveats: behavioral evidence alone does not localize circuits; Phase 1 activation patching requires separate authorization and causal protocol.

## 8. Limitations

- N=36 controlled synthetic geometric scenes only  
- Single prompt template; no broad prompt-order sweep in this phase  
- Greedy decode primary; candidate scores uncalibrated  
- One model, one quantization, one GPU  
- Directional-reversal GT is constructed for inverted queries on the same canvas  
- Text-only controls use blank canvas images (not pure no-image prompts)

---

**Verdict:** Phase 0D.2 harness + full 36-case run **COMPLETE and VALIDATED**.
