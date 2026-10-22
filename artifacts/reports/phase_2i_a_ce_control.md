# Phase 2I-A Report — Fair CE-QLoRA Control

**CE only · ≤50 steps · no custom loss · no holdout · exploratory**

---

## 1. Sampling repair
- Replaced curriculum-head schedule with **`maximize_family_coverage_schedule`**
- Result: **50 steps / 50 unique samples / 50 unique families** (vs Phase 2G’s 17 families)
- Labels: Yes 17 / No 17 / Undetermined 16 · seed `20261111` · deterministic

## 2. Token masking verification
- `verify_gold_answer_masking.py` → **PASS**
- `" Yes"` / `" No"` → **1** supervised token; `" Undetermined"` → **2**
- Prompt length **157**; exact gold span only; not last-N shortcut
- Processor-only check saved under `artifacts/training/phase_2i_a/masking_processor_check_v1.json`

## 3. Parser audit
- **`strict_answer_v1`**: Invalid/Missing/Conflicted never coerced to No/Undetermined
- Evaluation reports **invalid_output_rate**

## 4. Frozen control protocol
`configs/phase_2i_a/ce_control_protocol_v1.json` — CE-only, r=2/α=4, 50 steps, holdout forbidden, strict parser, raw completions on.

## 5. Actual training steps
**50 / 50** (sanity 5 + continue)

| Step | Loss |
|---:|---:|
| 1 | 14.89 (higher — correct answer-span supervision) |
| 5 | 17.15 |
| 20 | 10.47 |
| 50 | **0.76** |

## 6. Baseline / control metrics (6 val families / 18 cases)

| System | BA | Invalid rate |
|---|---:|---:|
| Frozen base standard | **0.389** | 0.0 |
| Base abstention prompt | 0.389 (prior run) | — |
| CE-QLoRA @5 | **0.389** | 0.333 |
| **CE-QLoRA @50** | **0.667** | **0.0** |

**Δ baseline→final = +0.278**  
Final recalls: Yes **1.0**, No **1.0**, Undetermined **0.0** (fork abstention still unsolved).

## 7. Family coverage
**50 train families** in schedule (full development pool head), balanced classes; val 6 families disjoint; holdout **false**.

## 8. Resource measurements
~**0.95–1.1 s/step**; RAM gate ~7 GiB; NF4 LoRA r=2; no OOM.

## 9. Tests
| | |
|---|---|
| Phase 2I-A tests | **11 OK** |
| Full ULVO suite | **226 OK** |
| System suite | **226 OK (41 skips)** |
| Gold-mask check | **PASS** |

## 10. Git
See push SHA below.

## 11. Remaining limitations
- Val N=18; Undetermined recall still 0 after CE  
- Invalid rate spiked at step 5 then recovered at 50 — monitor  
- Not a strictly matched comparison to Phase 2G (different sampling/masking/parser)  
- No custom losses used  

## 12. Phase 2I-B recommendation
**CONDITIONAL GO** for evidence-aware pilot **using this frozen CE control as baseline** (same schedule/seed/steps).  
Do **not** assume ES/CF will fix abstention — Undetermined still 0 recall.

**STOP after Phase 2I-A.**
