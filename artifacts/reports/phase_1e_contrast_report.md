# Phase 1E Report — Confound-Controlled Contrast Benchmark & Probe Revalidation

**Prior commit:** `23f5ddd`  
**No patching. No final 1C holdout evaluation. v1 datasets unchanged.**

---

## 1. Original confounding audit
Phase 1C balanced sample used condition→label bijection (lookup BA **1.0**). Verdict: **CONDITION-CONFOUNDED**.

## 2. Synthetic fixture correction
Initial BA **0.722** was due to weak separation (σ noise vs moderate class offsets), small eval groups, and early stopping.  
**Not justified** to permanently lower the bar to 0.7. Fixture restored with:
- Stronger class offsets (±4)
- More samples (90)
- More iterations / lower L2
- Threshold **BA ≥ 0.85**

`verify_probe_synthetic_fixtures.py` → **PASS**

## 3–4. New schema (v2)
- **One condition_id:** `CROSS_MODAL_TWO_PREMISE`
- Variants: `entailed` / `contradicted` / `underdetermined` (evaluator metadata only)
- Shared prompt format; paraphrases/query forms rotated
- Visual never shows entity **C**
- Labels from SymbolicSolver + oracle agreement; inconsistent excluded
- Single-modality insufficiency: **0** records sufficient from one modality alone

## 5. Size & splits
| | |
|---|---:|
| Families | **240** |
| Records | **720** |
| Labels | Yes/No/Undetermined = **240 each** |
| Splits | train 168 · val 36 · test 36 |
| Counterfactual-complete families | **True** |

v1 untouched. New ids `xc_*` do not reuse 1C holdout families.

## 6. Shortcut audit (family-disjoint train→eval)

| Baseline | Balanced accuracy |
|---|---:|
| Condition-only | **0.333** (chance) |
| Template / entity / color / lexical / layout | **≤ ~0.4** (acceptable) |
| Majority | ~0.333 |

Verdict: **SHORTCUTS_ACCEPTABLE**  
(Note: earlier full-set singleton layout lookup was methodological error; fixed with family-disjoint maps.)

## 7. Symbolic verification
All 720 records: solver status matches label; oracle agreement; C not on canvas; cf families complete.

## 8. Bounded probe pilot (development only)
**45** forwards (12 train + 3 val families × 3). Layers 3/12/18/23 sparse.

| Method | Balanced accuracy |
|---|---:|
| Multi-layer neural probe | **0.667** |
| Condition-only | **0.333** |
| Majority | **0.333** |
| L3/L12 last-token | 0.444 |
| L18/L23 last-token | 0.556 |

**Exploratory only** (N=45). Probe now **exceeds** condition baseline (unlike Phase 1D). Still **not** causal evidence.

## 9. GPU / RAM
Peak VRAM **~3331 MiB**; latencies elevated late in run (thermal/host load); no OOM.

## 10. Tests
Doctor / ULVO unittest / system Python / portability / compileall — see session verification (all green after 1E suite).

## 11. Git
Commit: `Add confound-controlled cross-modal contrast benchmark` (see SHA below).

## 12. Limitations
- N=45 probe pilot; wide uncertainty  
- Linear probes only  
- Paraphrase set finite  
- Mean-image loses spatial detail  
- Not publication metrics; not holdout  

## 13. Recommendation
**CONDITIONAL GO** for larger development probing on contrast v2 with pre-registered layer/probe plan, still **excluding** 1C final holdout until a dedicated evaluation protocol.  

**NO-GO** for causal claims or “publication novelty established.”

**Stopped after Phase 1E.**
