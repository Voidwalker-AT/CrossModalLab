# Phase 1F Report — Reliable Representation Probing & Shortcut-Resistant Evaluation

**Prior commit:** `7199700`  
**No causal patching. Final holdout untouched.**

---

## 1. Baseline audit
- Reproduced Phase 1E multi-layer probe BA **0.667** on stored features.
- Standardization/L2 fit on **train only**; eval on **development validation families**.
- Frozen shortcut baselines (train-fit majority maps, family-disjoint): condition/template/entity/layout/lexical/majority/random ≈ **0.33–0.4** chance band on full contrast audit; on 1F val families **max baseline BA = 0.333**.
- Premise information is legitimate signal when using lexical/text features; not labeled “shortcut” merely for predicting well.

## 2. Probe configurations
- Specs: layers {3,12,18,23} × tokens {last, last_text_non_image, mean_image}
- One-layer / one-token linear L2 logistic (no 24576-dim primary result)
- Optional PCA-64 control on L12 last (train-fit)
- Spec hash recorded; seeds fixed (20261101)

## 3. Family splits
- Feature meta: 36 development families (train+validation contrast v2 only)
- Frozen split: **27 train / 9 validation** families
- All three variants kept together per family
- **Final holdout not accessed**

## 4. GPU forwards
- Reused **45** Phase 1E captures (identity match)
- **New** forwards: **63** (cap 96)
- Total feature rows: **108**
- Peak VRAM ~**3331 MiB**

## 5. Probe results (development validation, n_val_families=9)

| Spec | BA (point) | Group-bootstrap 95% CI |
|---|---:|---|
| **L18 last** (best) | **0.778** | **[0.593, 0.963]** |
| Other layer×token specs | recorded in summary JSON | |
| Max shortcut baseline | 0.333 | |

Best status: **EXPLORATORY_POSITIVE_DECODE**  
CI low **0.593 > 1/3**; exceeds baselines by **>0.05**.

## 6. Shortcut comparisons
On the same 9 validation families, neural L18_last ≫ condition/entity/layout/majority/random baselines (0.333).

## 7. Statistical uncertainty
Group bootstrap (B=2000) over **9 families** — still wide; not a publication-grade power analysis.

## 8. Runtime findings
- Latency distribution heavy-tailed: min **4.7s**, median **14.3s**, p90 **34s**, max **77s**
- Slow cases: same image-token count (1377), similar VRAM (~3330 MiB)
- Late-run burst of ~4.7s samples after earlier 30–70s samples → consistent with **shared-GPU / host contention** (speculation flagged; no clocks/temp logging)
- No OOM

## 9. Tests
| Command | Exit |
|---|---|
| `ulvo project doctor .` | 0 |
| ULVO unittest discover | 0 |
| System Python unittest | 0 |
| `check_portability.py` / compileall | 0 |

## 10. Git
See push SHA below.

## 11. Limitations
- N=9 validation families for CI  
- Development only; not holdout  
- Linear probes; NF4 residuals  
- Best-spec selection on development val (pre-registration for holdout still required)  
- Decodability ≠ causal usage  

## 12. Recommendation for causal experiments
**CONDITIONAL GO** to **design** (not execute) causal patching experiments targeting late layers (18/23) at the final question token, contingent on:
1. Pre-registered holdout evaluation protocol  
2. Explicit user approval  
3. Token-aligned intervention design from Phase 0E  

**NO-GO** for executing patching or claiming causal circuits from probing alone.

**Stopped after Phase 1F.**
