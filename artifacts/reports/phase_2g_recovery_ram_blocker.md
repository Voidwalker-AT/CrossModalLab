# Phase 2G Recovery Report — Shell Restored; RAM Gate Blocked Training

**Mode:** Recovery + CPU validation. **No training executed.**

---

## 1. Shell recovery
`Write-Output SHELL_READY` → **SHELL_READY**. Execution backend healthy.

## 2. Code audit & corrections (draft files from failed session)

| Issue | Fix |
|---|---|
| `configs/... dataset` pointed at nonexistent `cross_modal_v1.jsonl` | Corrected to `cross_modal_contrast_v2.jsonl` |
| `freeze_val_ids` raised on any `test` row in file | Skips holdout rows; only selects `validation` families; asserts selected ∉ test |
| Training script had no host-RAM gate | Added **≥4 GiB** free gate before model load (exit 4) |
| Tests imported torch unconditionally | Skipped when torch absent (Mode B) |
| Compiled | `py_compile` **exit 0** for all 2G Python files |

**Untrusted draft risks reviewed:** no unbounded loop (max_steps slices schedule); no holdout eval path; completion-only masking present; LoRA vision filter present; checkpoint dir under artifacts (not Git).

## 3. CPU validation
| Check | Result |
|---|---|
| `tests/test_phase_2g_training.py` (ULVO) | **8 OK** |
| Same (system Python) | **8 OK (3 skips)** |
| Full suite ULVO | **196 OK** |
| Portability | **PASS** |
| ULVO doctor | **PASSED** |
| CUDA stack | torch **2.6.0+cu124**, CUDA **True** |

## 4. Host RAM gate (BLOCKER)
| Metric | Value |
|---|---|
| Free host RAM | **3.50 GiB** (required **≥ 4.0 GiB**) |
| Total | ~16 GB |
| Top consumers | GoogleDriveFS ~2.4 GB WS, mimo ~1.0 GB, ChatGPT ~0.9 GB |
| Action | **Did not kill unrelated processes** |
| Training / baseline GPU eval | **NOT STARTED** |

## 5–7. Baseline, training, metrics, checkpoints
**Not executed** (RAM gate). **0 / 50 optimizer steps.**

## 8. Tests
See §3 — all CPU gates pass.

## 9. Git
This recovery commits **validated source/tests/config only** — no fabricated empirical training results.  
Commit message: `Add controlled Qwen QLoRA learning validation experiment` (engineering+tests; training still gated).

## 10. Recommendation
**NO-GO** for GPU training until free host RAM ≥ 4 GiB.  
**GO** for CPU pipeline readiness.  
Re-run Stage 4–7 when RAM gate passes:  
`ulvo run . -- python scripts/run_phase_2g_qlora_training.py`

**STOP after this recovery report.**
