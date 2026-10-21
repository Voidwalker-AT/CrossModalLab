# Phase 2B Recovery Report (Exploratory / Post-Hoc Modified)

**Date:** 2026-10-21  
**Git base:** `d756d10` (Phase 2A design)  
**Mode:** Recovery audit only — **no new GPU experiment**

---

## A. Current process state
- **No Phase 2B Python process running** (no project ULVO/python worker).
- GPU idle-ish (~294 MiB); unrelated compute PIDs present with N/A memory reporting.
- Evidence file preserved:  
  `artifacts/causal/phase_2b/intervention_records_v1.jsonl`  
  SHA-256 `50bb225b97e29b394b9a746a1c35134c02cf95d1203ad1126cecb0fcd512e337`  
  **16 valid JSONL records**, 0 truncated, 0 duplicate IDs.

## B. Why execution is slow (confirmed vs speculation)
| Cause | Status |
|---|---|
| Model load ~18–20s once per process | **Confirmed** (prior runs) |
| Candidate scoring = 3 extra full forwards per scored condition | **Confirmed** (scorer design) |
| Per-intervention multi-forward (baseline + patch + 3 candidates) | **Confirmed** |
| Heavy latency tail 30–77s in feature collection | **Confirmed empirically** |
| Shared-GPU / host contention as tail cause | **Speculation** (no clocks/temp logs) |
| Windows WDDM shared memory accounting | **Inferred** for VRAM >4096 |

## C. Completed interventions
- **10** interventions + **6** baselines = **16** ok records  
- Families: `xc_ho_0013` (complete both pair types), `xc_ho_0016` (baselines only for entailed pair)  
- Missing for full 3-family × 2-pair × 5-dir matrix: **26** IDs  
- Last completed: `xc_ho_0016:entailed_vs_underdetermined::baseline_target`

### Exploratory Δm (not confirmatory)
| Direction | n | mean Δm |
|---|---:|---:|
| source_to_target | 2 | −0.42 |
| target_to_source | 2 | +0.46 |
| same_label | 2 | +0.03 |
| unrelated_token | 2 | **−3.73** |
| random_source | 2 | **−1.92** |

**Flag: GENERIC_INTERVENTION_SENSITIVITY** — negative controls larger than targeted patches.

## D. GPU memory risk
- Recorded `peak_vram_mib` ≈ **5076–5142** on a **4096 MiB** card.
- Field intended as `torch.cuda.max_memory_allocated` but **peak stats were not reset per intervention** → process-lifetime peaks.
- Values >4096 are **possible** under WDDM shared GPU memory; **not treated as fabricated**.
- Risk: **moderate** for long sessions; engineering gate used ~3330 MiB in earlier sparse captures; patch+scoring stacks higher.
- Candidate scoring retains graph-free inference tensors; no evidence of unbounded growth in 16 records, but repeated forwards increase peak.

## E. Scientific validity issues
1. **Pair image hashes differ** even for “same visual structure” (canvas jitter) → not identical-image interventions.  
2. **Scoring hook lifetime**: first discarded run scored without active patch (Δm=0). Fixed with `residual_patch_context` during scoring. **POST-HOC MODIFIED**.  
3. **Baseline bookkeeping**: early resume bug; fixed by appending baselines to `existing`.  
4. **Controls dominate targeted effects** on N=2 pairs.  
5. **L18 selected after multi-comparison** (from 1F).  
6. Historical `Remove-Item` discarded earlier partial JSONL — **unrecoverable**; current file is post-fix survivor.

## F. Code fixes & CPU tests
- `src/causal/recovery.py` + `scripts/recover_phase_2b_audit.py`  
- Hook-lifetime context already in `patch_runner.py`  
- Tests: recovery validation, resume matrix, control flag, hook lifetime, margins, config version  
- **ULVO 155 OK** · **System Python 155 OK (9 skips)** · portability/compileall OK  

## G. Safe reduced-cost strategy (if ever re-authorized)
1. Freeze pairs with **identical image_sha256** or rename condition.  
2. Only `source_to_target` + unmodified target baseline on **3 families × 1 pair type** (≤6 scored runs).  
3. Reset CUDA peak stats each forward; log allocated/reserved + nvidia-smi dedicated.  
4. Do **not** expand matrix automatically.

## H. Recommendation
### **REDESIGN / STOP** before any further GPU causal runs.

Rationale: incomplete matrix, control-dominated effects, image-hash mismatch vs “same image” claim, post-hoc scoring-hook fix, unreliable peak telemetry.

---

**Git:** feature branch with engineering recovery only. No confirmatory causal claims pushed as science.
