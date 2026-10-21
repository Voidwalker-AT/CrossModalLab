# Phase 2C-C Report — Paired SmolVLM2 vs Qwen3-VL Behavioral Evaluation

**Mode:** Behavioral only (not mechanistic)  
**Holdout:** contrast-v2 `test` split **untouched**  
**Budget:** 18+18 = **36** evaluations, one model at a time

---

## 1. Frozen manifest
`configs/phase_2c_c/paired_eval_manifest_v1.json`  
**6 train-split families** (xc_ho_0017/0019/0020/0021/0022/0023) × 3 variants = **18 cases**  
All modality audits: single-modality **insufficient**; C **hidden**; labels solver-verified.

## 2. Input equivalence
Identical `image_sha256`, `prompt_sha256`, GT labels. Differences: official chat templates / processors only (recorded per record).

## 3. Processor differences
| | SmolVLM2 | Qwen3-VL |
|---|---|---|
| Image tokens (384) | ~1377 | **144** |
| Serialisation | SmolVLM chat template | Qwen3-VL chat template |
| Token IDs | model-specific | model-specific |

## 4–5. Counts & metrics (n=18 each)

| Metric | SmolVLM2 | Qwen3-VL-2B |
|---|---:|---:|
| Balanced accuracy | 0.444 | **0.556** |
| Macro-F1 | 0.357 | 0.453 |
| Recall Yes | 0.667 | 0.667 |
| Recall No | 0.667 | **1.000** |
| Recall Undetermined | **0.000** | **0.000** |
| Undetermined abstention | **0.0 (0/6)** | **0.0 (0/6)** |
| Unsupported definite rate | **1.0** | **1.0** |
| Mean latency | **20.8 s** | **0.43 s** |
| Peak nvidia-smi | 3873 MiB | 3092 MiB |

## 6. Paired comparison (18 families-cases)

| Outcome | N |
|---|---:|
| Both correct | 7 |
| Both incorrect | **7** |
| Only SmolVLM2 correct | 1 |
| Only Qwen correct | 3 |
| **Shared underdetermined failure** | **6** |
| Prediction disagreements | 7 |

## 7. Shared failure evidence
**All 6 underdetermined cases failed in both models** (definite Yes/No instead of Undetermined) — strongest shared behavioral pattern.  
Archive: `artifacts/paired/phase_2c_c/paired_evaluations_v1.jsonl` (full distribution preserved).

## 8. Memory / runtime
SmolVLM2 slower (~21 s/case); Qwen ~0.43 s/case. Both within 4 GB with NF4. Sequential model load/unload; no simultaneous residency.

## 9. Limitations
- N=6 families; not publication significance  
- Development-only  
- No candidate-likelihood comparison claimed for Qwen  
- Shared failure ≠ shared neural circuit  

## 10. Tests
ULVO/system suites + 2C-C tests; doctor/portability OK (session log).

## 11. Git
See push SHA.

## 12. Recommendation
- **Shared evidence-insufficiency failure** replicates across architectures at behavioral level.  
- **Next mechanistic experiment:** remain on **SmolVLM2** for Layer-18 work (richer prior instrumentation), **not** because Qwen is worse—but because NF4 SmolVLM2 + existing harness is ready; fork abstention is the clearest open target.  
- **Reconsider** any claim that L18 probe BA implies abstention circuits; behavioral fork failure is robust across models regardless.  
- Larger cross-model eval: CONDITIONAL GO later.

**Stopped after Phase 2C-C.**
