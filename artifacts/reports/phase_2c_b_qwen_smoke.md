# Phase 2C-B Report — Qwen3-VL-2B NF4 Acquisition & Smoke

**Revision:** `89644892e4d85e24eaac8bacfd4f463576704203`  
**License:** Apache-2.0  
**No training, no causal patching, no holdout.**

---

## A. Pinned revision
`Qwen/Qwen3-VL-2B-Instruct` @ `89644892e4d85e24eaac8bacfd4f463576704203`

## B. Download
- **4,266,648,961 bytes (~4069 MiB)**, 12 files, ~1011 s  
- Cache: `E:/cache/huggingface/models--Qwen--Qwen3-VL-2B-Instruct/...`  
- Manifest: `datasets/metadata/qwen3vl2_acquisition_manifest.json`  
- SmolVLM2-2.2B **untouched**

## C. Dependencies
No upgrades. torch 2.6.0+cu124 · transformers 5.20.1 · bnb 0.50.2 · CUDA True · Py 3.11.16

## D. NF4 loading
| Metric | Value |
|---|---:|
| Load time | ~13–15 s |
| Class | `Qwen3VLForConditionalGeneration` |
| Text layers / hidden | 28 / 2048 |
| image_token_id | 151655 |
| torch allocated | **1505 MiB** |
| torch reserved | **1566 MiB** |
| nvidia-smi used | **~2071–2102 MiB** |

## E. Memory
Well under 4096 MiB dedicated; host RAM ample. No OOM.

## F. Visual grounding (4 cases)
| Case | Pred | Latency | img tokens |
|---|---|---:|---:|
| object identity | Yes | 3.2s | 144 |
| left-of | Yes | 0.7s | 144 |
| reverse spatial | No | 0.8s | 144 |
| missing visual | No | 0.5s | 0 |

## G. Cross-modal (3 development cases)
| Variant | GT | Pred | Correct |
|---|---|---|---|
| entailed | Yes | Yes | ✓ |
| contradicted | No | No | ✓ |
| underdetermined | Undetermined | No | ✗ (abstention failure, similar to SmolVLM2) |

## H. Hook compatibility
- Passive hook on language layer **12**  
- logits max\|Δ\| vs unhooked = **0.0**  
- Finite activations; hook removed cleanly  

## I. Comparison vs SmolVLM2-2.2B (smoke only)

| | SmolVLM2-2.2B NF4 | Qwen3-VL-2B NF4 |
|---|---|---|
| Layers (text) | 24 | 28 |
| Hidden | 2048 | 2048 |
| Image tokens (384 canvas) | 1377 | **144** (different tiling) |
| Load torch alloc | ~1.8–3.3 GiB | **~1.5 GiB** |
| Cross-modal smoke | prior pilots | 2/3 correct (fork fails) |
| Passive hook | verified L18 | verified L12 |

Not a superiority claim.

## J. Limitations
Smoke only; no accuracy generalization; tokenizations differ; no cross-model logit calibration.

## K. Tests
ULVO/system suites + 2C-B tests; doctor/portability OK (see session).

## L. Git
See push SHA.

## M. Recommendation
**CONDITIONAL GO** for larger **behavioral** cross-model comparison on development contrast sets.  
**NO-GO** for cross-model mechanistic/patching until a shared adapter + pre-registered protocol.

**Stopped after Phase 2C-B.**
