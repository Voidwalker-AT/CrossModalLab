# Phase 1D Stage 1 Report — Representation Probing Pipeline & Development Pilot

**Verdict:** Engineering pipeline **VERIFIED**. Scientific 3-way relational interpretation **BLOCKED (condition-confounded)**.  
**No activation patching. Final holdout untouched.**

---

## 1. Dataset confounding audit

Balanced Phase 1C protocol maps:

| Condition | Label |
|---|---|
| `both_premises` | Yes |
| `both_modified` | No |
| `underdetermined_fork` | Undetermined |

**Condition→label lookup accuracy = 1.0 (bijective).**  
Verdict: **CONDITION-CONFOUNDED**.

Artifact: `artifacts/probes/phase_1d/label_condition_confounding_v1.json`  
Smallest defensible extension: ≥2 surface templates per 3-way label (already partially available in v1 as visual_removed/text_removed/forks) evaluated as a **contrast set**, not collapsed into a single condition-determined label.

## 2. Probing task definitions

See design in Phase 1C + pilot implementation:
- Features: concat of `last`, `last_text_non_image`, `mean_image` at layers **3/12/18/23** (dim **24576**)
- Labels: 3-way Yes/No/Undetermined on balanced conditions
- Groups: `family_id`
- Baselines: majority, **condition lookup**, permutation

## 3–5. Implementation

- `src/probes/linear_probe.py` — NumPy L2 multinomial logistic (no sklearn)
- `src/probes/confounding.py` — confounding metrics
- `scripts/analyze_label_condition_confounding.py`
- `scripts/verify_probe_synthetic_fixtures.py` — **PASS**
- `scripts/run_phase_1d_probe_pilot.py` — resumable capture + probe fit

## 6. Unit tests
`tests/test_phase_1d_probes.py` + full suite (see verification section).

## 7. GPU forward count
**45** new forwards (12 train + 3 val families × 3 conditions). Cap 54. Holdout not used. Development-exposed pilot families excluded.

## 8. Probe metrics (development only; confounded)

| Method | Balanced accuracy |
|---|---:|
| Multi-layer linear probe (val families) | **0.667** |
| **Condition-label lookup** | **1.000** |
| Majority | 0.333 |

Layer-wise last-token probes recorded in `probe_pilot_summary_v1.json`.  
Permutation test on train groups recorded.

**Interpretation:** Probe does **not** beat the condition-lookup shortcut. Scientific claim of decodable *relational logic* is **BLOCKED** until the confound is broken by dataset extension. Engineering pipeline is valid.

## 9. VRAM / RAM
Peak VRAM **~3331 MiB**; mean latency ~**15.3 s**/forward; no OOM.

## 10. Limitations
- N=45 confounded samples; not holdout  
- Single feature layout  
- Linear probe only  
- Confounding prevents relational interpretation  
- Not publication metrics  

## 11. Git
See push SHA below.

## 12. Recommendation
**CONDITIONAL GO** for larger probing **only after** a non-confounded contrast-set extension (v1b) with ≥2 templates per label.  

**NO-GO** for presenting current 3-way accuracy as evidence of logical reasoning.  

**NO-GO** for causal claims / patching without separate approval.

**Stopped after Phase 1D Stage 1.**
