# Phase 1C — Probe Study Design (No Training in This Phase)

**Status:** Design + split freeze only  
**Central RQ:** Cross-modal relational composition and evidence sufficiency  
**Model:** SmolVLM2-2.2B-Instruct @ `482adb537c021c86670beed01cd58990d01e72e4` (NF4)

> Probe **decodability ≠ causal model usage**. No probe is trained in Phase 1C.

---

## 1. Dataset integrity verdict (v1)

| Check | Result |
|---|---|
| Families / records | 240 / 1920 |
| Conditions per family | 8 |
| Solver labels | **0 mismatches** |
| Family-disjoint original splits | True |
| Cross-split logical duplicates | 0 |
| C hidden on both_premises canvases | Yes |
| Modality ablations | OK |
| Class imbalance | Yes 240 · No 240 · Undetermined 1440 (condition-driven) |

**Inconsistent vs contradicted:**  
- `both_modified` → solver **contradicted** → 3-way label **No**  
- `contradictory_evidence` → solver **inconsistent** → **not** forced into Yes/No/Undetermined  

---

## 2. Class-balanced sampling (v1 GT unchanged)

Per development/holdout family include exactly:

| Stratum | Condition | Label |
|---|---|---|
| Yes | `both_premises` | Yes |
| No | `both_modified` | No |
| Undetermined | `underdetermined_fork` | Undetermined |

Metrics: **balanced accuracy**, **macro-F1**, **per-class recall** (`compute_probe_metrics`).  
Related condition variants of the same family are **never** treated as independent samples for accuracy.

---

## 3. Contamination audit & frozen splits

Phase 1B pilot used **12 families, all from original `test` (12/36)** → original test split is **partially contaminated**; final evaluation must not use those 12 families.

Frozen protocol (`configs/phase_1c/frozen_split_manifest_v1.json`):
- Mark pilot families **development_exposed**
- **Final holdout** = unexposed former-test families + 12 families carved from unexposed train (seed `20261022`)
- **Development** = remaining unexposed train + validation
- Disjointness: pilot ∩ (dev ∪ holdout) = ∅; dev ∩ holdout = ∅
- Checksums in `configs/phase_1c/checksums_v1.json`

**Rule:** No prompt/model/probe tuning on final holdout labels.

---

## 4. Probe architecture (future Phase 1D+)

### 4.1 Tasks

| Probe | Labels | Features (preferred) | Layers |
|---|---|---|---|
| Visual direction | left-right order metadata | `mean_image`, `first_image`, `last_image` residuals | 3, 12 |
| Textual premise | text premise present / relation polarity on text-only diagnostics | `last`, `last_text_non_image` | 12, 18 |
| Cross-modal composition | Yes vs not-Yes on `both_premises` vs ablations | `last` residual | 12, 18, 23 |
| Evidence sufficiency | Undetermined vs definite (excluding inconsistent) | `last` residual | 12, 18, 23 |

### 4.2 Capacity & regularization
- Linear multinomial logistic regression (L2)
- C selected on **development** validation only (small grid)
- No deep MLPs in the first pass
- Standardize features with train-fold statistics only

### 4.3 Splits & leakage
- Group-disjoint by `family_id`
- Train on balanced development sample; evaluate on balanced final holdout
- Never put two conditions of one family across train/eval

### 4.4 Statistics
- Bootstrap CI on holdout balanced accuracy (B ≥ 2000)
- Permutation test vs chance (label shuffle within eval)
- Negative controls: Gaussian noise features; random-label probes
- Multiple comparisons: Benjamini–Hochberg across layer×probe families
- Power note: holdout families ≈ 36 balanced triplets — **low power** for small effects; report wide CIs

### 4.5 Interpretation limits
High probe accuracy shows **linear accessibility**, not that the model **uses** the feature.  
Output-readout mismatch hypotheses require **later causal tests** (not Phase 1C).

---

## 5. Resource plan (from Phase 1A/1B measurements)

| Item | Estimate |
|---|---|
| Time / forward (visual) | ~10–17 s |
| Time / forward (text-only) | ~1–2 s |
| Peak VRAM (NF4 + hooks) | ~3.3–3.6 GiB |
| Sparse rows / sample / 4 layers | ~0.16 MiB CPU |
| Full residual (if ever) | ~11.7 MiB/layer FP32 |
| Balanced dev forwards | 216 families × 3 = **648** multimodal forwards (≈ 2–3 h) |
| Balanced holdout forwards | 36 × 3 = **108** forwards (≈ 25–35 min) |
| Total sparse activation disk | ≈ 0.16 MiB × 756 ≈ **120 MiB** |

**Not executed in 1C.** Collection must use capture identity + resumable JSONL (Phase 1A/1B integrity).

Chunking: 12 families/run; abort on OOM; skip complete identities only.

---

## 6. GO / NO-GO criteria (for later probe training)

**Supports (still non-causal) representation claims if:**
1. Individual-premise probes beat chance on held-out families  
2. Composition probe above chance on `both_premises` vs ablations  
3. Sufficiency probe above chance for Undetermined vs definite  
4. Negative controls fail as expected  

**Alternative explanations to falsify:** entity/color shortcuts; prompt-template artifacts; quantization noise; position confounds at `last` token.

**NO-GO for causal language** until activation patching with pre-registered margins (separate phase).

---

## 7. Phase 1C deliverables checklist
- [x] Dataset audit  
- [x] Contamination audit  
- [x] Frozen split manifests + checksums  
- [x] Balanced sampling protocol (no GT rewrite)  
- [x] Capture integrity hardening + tests  
- [x] Probe design + resource plan  
- [ ] Probe training → **await approval (Phase 1D)**  
