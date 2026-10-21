# Phase 1B Report — Cross-Modal Dependency Validation & Dataset Expansion

**Label:** Behavioral pilot DESCRIPTIVE (non-causal)  
**Model:** SmolVLM2-2.2B-Instruct @ `482adb537c021c86670beed01cd58990d01e72e4` (NF4)  
**Date:** 2026-10-21  
**Prior commit:** `404c9c8` (Phase 1A)

---

## 1. Capture resume corrections

Implemented specification-aware identity in `src/instrumentation/capture_integrity.py`:

Identity binds: **sample_id + image_sha256 + prompt_sha256 + model_id/revision + quantization + layer set + token selectors + schema version**.

Resume **skips only** when:
1. A stored `ok` row matches the full identity, and  
2. Vector file exists with required layers/selectors.

Phase 1A runner now recaptures incomplete rows (e.g. smoke 2-layer vs full 4-layer spec).  
Historical Phase 1A artifacts were **not deleted**.

Tests cover: complete match, missing layer, missing/corrupted vectors, changed revision/selectors.

## 2. Dataset size and splits

Generator: `scripts/build_cross_modal_dependency_dataset.py`  
Output: `datasets/cross_modal_dependency_v1/`

| Item | Value |
|---|---|
| Families | **240** |
| Conditions / family | **8** |
| Records | **1920** |
| Label counts | Yes 240 · No 240 · Undetermined 1440 |
| Splits (families) | train 168 · validation 36 · test 36 |
| Group-disjoint splits | **True** |
| Solver mismatches | **0** |
| Held-out entity triples | 48 families (`xm_ho_*`) |

**Note:** Object-vocabulary disjointness does **not** alone eliminate shortcut leakage.  
Pilot preferred **test-split** families first.

## 3–4. Cross-modal design & matched counterfactuals

Entities A,B,C; visual shows only two (C hidden on canvas for chain conditions).  
Conditions: both_premises, visual_removed, text_removed, visual_reversed, text_reversed, both_modified, underdetermined_fork, contradictory_evidence.

Labels from **SymbolicSolver** independently.  
Example: both_premises → entailed/Yes; single-premise ablations → underdetermined; contradictory → inconsistent (not forced definite).

Matched family = all 8 conditions share A,B,C and group_id=`family_id`.

## 5. Pilot results (n=36 = 12 families × 3 conditions)

| Condition | N | Accuracy |
|---|---:|---:|
| both_premises (Yes) | 12 | **0.75** |
| visual_removed (Undetermined) | 12 | **0.00** |
| text_removed (Undetermined) | 12 | **0.00** |

Confusion (true→pred):  
Yes→Yes 9, Yes→No 3; Undetermined→Yes 21, Undetermined→No 3; **Undetermined→Undetermined 0**.

Paired changes when leaving `both_premises`:
- Prediction changed on visual_removed: **3/12**
- Prediction changed on text_removed: **4/12**
- visual_removed still “Yes”: **10/12**

**Interpretation (behavioral only):** On this pilot, single-premise ablations rarely induce abstention; the model often emits definite answers under insufficient evidence. Answer changes alone are **not** proof of internal cross-modal composition.

## 6. GPU / RAM
Peak VRAM **~3330 MiB**; mean latency **~11.0 s**; no OOM.

## 7. Tests

| Command | Exit |
|---|---|
| `ulvo project doctor .` | 0 |
| ULVO unittest discover | 0 (see session log; includes 1B tests) |
| System Python unittest | 0 |
| `check_portability.py` / compileall | 0 |
| Dataset build 240 | 0 |
| Pilot 36 | 0 |

## 8. Limitations
- Pilot N=12 families (not publication metrics)  
- Preferential test-split selection may differ from train distribution  
- Undetermined class dominates full pool (by condition design)  
- No probe/patching  
- Behavioral only  

## 9. Git
See push evidence in session report.

## 10. Recommendation for Phase 1C
**CONDITIONAL GO** for probe **design** and further group-disjoint data use, **only with**:
- larger evaluation slices from validation/test partitions  
- explicit probe train/test grouping  
- separate approval before probe training or causal patching  

**NO-GO** for claiming established cross-modal circuits from this pilot alone.

**Stopped after Phase 1B.**
