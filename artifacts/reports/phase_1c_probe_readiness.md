# Phase 1C Report — Probe Readiness, Validated Splits, Mechanistic Study Design

**Date:** 2026-10-21  
**Prior commit:** `792a475`  
**No probe training. No activation patching.**

---

## 1. Dataset integrity verdict (v1)

| Check | Result |
|---|---|
| Families / records | **240 / 1920** |
| Conditions per family | **8** |
| Independent solver re-audit | **0 mismatches** |
| Family-disjoint original splits | **True** |
| Cross-split logical duplicates | **0** |
| both_premises third entity hidden | **True** |
| Modality ablations | **OK** |
| Answer-label leakage in premise body | **0** |

Class distribution (condition-driven, unchanged): Yes **240** · No **240** · Undetermined **1440**.

**Contradicted vs inconsistent:**  
- `both_modified` → **contradicted** → 3-way **No**  
- `contradictory_evidence` → **inconsistent** → **excluded** from 3-way metrics (not forced)

## 2. Class-balanced sampling strategy

v1 ground truth **preserved**. Protocol: per family one record each of  
`both_premises`(Yes) + `both_modified`(No) + `underdetermined_fork`(Undetermined).  
Metrics defined: balanced accuracy, macro-F1, per-class recall (`compute_probe_metrics`).  
Related conditions never counted as independent accuracy samples.

## 3. Frozen train/validation/test manifests

`configs/phase_1c/frozen_split_manifest_v1.json` + checksums + family ID lists + balanced JSONL extracts.

| Partition | Families |
|---|---:|
| Development | **192** |
| Final holdout | **36** |
| Balanced holdout labels | Yes 36 · No 36 · Undetermined 36 |

## 4. Test contamination audit

- Pilot families: **12**, all originally in `test` (**12/36**)  
- Verdict: **PARTIAL_ORIGINAL_TEST_CONTAMINATION**  
- New final holdout **excludes all pilot families**  
- Development also excludes pilot families  
- Disjointness verified in manifest  

**Rule:** No tuning on final holdout labels.

## 5. Capture integrity corrections

- Identity hash from **canonical fields only** (never includes `identity_hash`)  
- Resume **rebuilds** identity from stored fields; forged hashes cannot bypass  
- Prompt/image/layer/selector/revision changes invalidate captures  
- Non-finite values anywhere in vectors **rejected**  
- `atomic_write_json` for safe artifact updates  
- Historical captures preserved  

## 6. Probe architecture and controls

See `docs/phase_1c_probe_study_design.md`: linear L2 probes for visual direction, textual premise, composition, sufficiency; group-disjoint by family; permutation/bootstrap; negative controls; BH multiple-comparison handling; explicit decodability≠causality.

## 7. Resource estimates (not executed)

- Balanced dev: **648** multimodal forwards (~2–3 h)  
- Balanced holdout: **108** forwards (~25–35 min)  
- Peak VRAM ~3.3–3.6 GiB  
- Sparse activation disk ~**120 MiB**  

## 8. Tests

| Command | Exit |
|---|---|
| `ulvo project doctor .` | 0 |
| ULVO unittest | **116 OK**, 0 |
| System Python unittest | **116 OK (4 skips)**, 0 |
| `check_portability.py` / compileall | 0 |

## 9. Git

Commit message: `Prepare CrossModalLab mechanistic probe study and validated splits`  
(see push SHA in session evidence)

## 10. Recommendation

**CONDITIONAL GO** for Phase 1D **probe training on development only**, using frozen manifests, with holdout evaluation reserved and untouched until pre-registration.  

**NO-GO** for causal circuit claims or holdout peeking.

**Stopped after Phase 1C preparation.**
