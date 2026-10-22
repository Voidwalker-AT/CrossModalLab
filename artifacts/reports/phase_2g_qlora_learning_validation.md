# Phase 2G Report — Controlled QLoRA Learning Validation

**Exploratory, development-only. Holdout never used. Custom losses not used.**

---

## A. Environment & RAM preflight
| Item | Value |
|---|---|
| RAM available (psutil) | **6.22–6.86 GiB** ≥ 4.0 gate → **PASS** |
| torch | 2.6.0+cu124 · CUDA True · RTX 2050 |
| peft / transformers / bnb | 0.21.2 / 5.20.1 / 0.50.2 |
| Qwen snapshot | Present @ `89644892…` |
| E: free | ~36 GB |

## B. Training-code audit
- Fixed `hashlib_sha(dict)` checkpoint bug  
- Resume-aware step counter (skip completed steps only if checkpoint exists)  
- Completion-only masking (last 4 tokens supervised; prompt/image/pad = -100)  
- LoRA only on language q/k/v/o (vision path filtered)  
- RAM gate embedded before model load  

## C. Dataset integrity & splits
- Source: `cross_modal_contrast_v2` (sha256 `cda6379b…`)  
- Train: 168 families · Eval freeze: **6 validation families / 18 cases**  
- Holdout families excluded (`holdout_used: false`)  
- All three labels present in train schedule  

## D. Frozen baseline (standard prompt)
**18 cases / 6 families**

| Metric | Value |
|---|---:|
| Balanced accuracy | **0.389** |
| Macro-F1 | 0.363 |
| Recall Yes / No / Undet | 0.167 / 0.667 / 0.333 |
| Unsupported definite | 0.667 |
| Over-abstention | 0.333 |

Strong abstention prompt baseline: **identical BA 0.389** (no prompt gain on this set).

## E. LoRA configuration
r=**2**, α=**4**, targets `q_proj,k_proj,v_proj,o_proj`, vision/projector frozen, NF4 base, AdamW 1e-4, batch 1, grad ckpt, `use_cache=False`  
Trainable params: **802,816**

## F. Five-step sanity
| Step | Label | Loss | Grad L2 |
|---:|---|---:|---:|
| 1 | Yes | 7.03 | 19.1 |
| 2 | No | 6.32 | 18.7 |
| 3 | Undetermined | 5.52 | 20.1 |
| 4 | Yes | 5.28 | 21.8 |
| 5 | No | 4.33 | 23.8 |

All finite; all three classes in schedule; VAL@5 BA **0.333**.

## G. Actual total optimizer steps
**50 / 50**

## H. Training-loss trajectory
`7.03 → 4.33 (5) → 0.0001 (20) → 2.8e-5 (50)` — **near-total memorization** of the completion span.

## I. Validation at checkpoints
| Checkpoint | Balanced accuracy |
|---|---:|
| Step 0 (base) | **0.389** |
| Step 5 | **0.333** |
| Step 50 | **0.222** |

## J. Base vs prompt vs QLoRA

| System | BA |
|---|---:|
| Base standard | 0.389 |
| Base abstention prompt | 0.389 |
| QLoRA standard (step 50) | **0.222** |

**Δ = −0.167 vs baseline.** Underdetermined recall remained **0.0** after training.

## K. Memory & runtime
- ~0.95–1.1 s / step; RAM gate stayed ≥ ~6 GiB  
- Peak telemetry in per-step JSONL under `artifacts/training/phase_2g/`  

## L. Checkpoints
```
artifacts/training/phase_2g/checkpoints/step_005/  (adapter + optimizer + meta)
artifacts/training/phase_2g/checkpoints/step_050/
```
No base-model duplication; binaries **not** committed to Git.

## M. Tests
| | Exit |
|---|---|
| `py_compile` new files | 0 |
| `tests.test_phase_2g_training` | 8 OK |
| Full ULVO suite | **196 OK** (from pre-train; post-train same code) |

## N. Git
See push SHA below.

## O. Scientific limitations
- **Development validation only (6 families, 18 cases)** — tiny  
- Loss collapse ⇒ **memorization**, not generalization  
- Validation **degraded** after training  
- No custom loss; no holdout; no causal claims  
- Abstention prompt did not help this baseline set  

## P. Next-phase recommendation
**CONDITIONAL GO / prioritize redesign over more CE-QLoRA:**
1. Treat this as **negative evidence** for naive 50-step CE-QLoRA on this curriculum.  
2. Next justified experiment: **evidence-aware / class-balanced loss or longer data** — only after new design approval.  
3. Do **not** scale CE training steps further on this data without a generalization fix.

**VERDICT: PASS** (experiment completed as specified; scientific outcome = no improvement / overfit).

**STOP after Phase 2G.**
