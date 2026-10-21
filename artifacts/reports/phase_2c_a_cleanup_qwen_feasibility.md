# Phase 2C-A — Legacy 500M Cleanup & Qwen3-VL-2B Feasibility

**Date:** 2026-10-21  
**No Qwen download. No new GPU inference.**

---

## A–D. 500M inventory & cleanup

| Item | Value |
|---|---|
| Path removed | `E:/cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-500M-Instruct` |
| Layout | Nonstandard self-contained snapshot `a7da5b986cb59b408707209984f360a5f4ad7e47` (no refs/blobs sharing) |
| Eligible bytes | **1,019,893,574** (13 files) |
| Measured reclaim | **1,019,920,384 bytes (~972.67 MiB)** |
| Method | Rename to `.pending_delete` then `rmtree` (lock-safe) |
| SmolVLM2-2.2B | **Intact** (~8819 MiB); scan_cache_dir still lists it |
| Historical metadata | Preserved (`datasets/metadata/model_acquisition_manifest.json`) |
| Reacquire | HF `HuggingFaceTB/SmolVLM-500M-Instruct` @ `a7da5b986cb59b408707209984f360a5f4ad7e47` |

Scripts that *mention* 500M are historical pilots; results remain valid via pinned IDs. No shared blobs with 2.2B.

## E–G. Qwen3-VL-2B assessment (from official HF card/config)

| Field | Value |
|---|---|
| Repo | `Qwen/Qwen3-VL-2B-Instruct` **exists** |
| License | **Apache-2.0** |
| Arch | `Qwen3VLForConditionalGeneration` / `qwen3_vl` |
| Params | **2B**, tensor type **BF16** safetensors |
| Text layers | **28** (hidden **2048**, GQA 16/8) |
| Vision | **24** layers; patch **16**; merge **2**; DeepStack 5/11/17 |
| `image_token_id` | **151655** |
| Transformers | Card targets **≥4.57**; local env has **5.20.1** (includes qwen3_vl) |
| `trust_remote_code` | **Not required** (native arch) |
| Attention | SDPA default; FA2 optional (not for Windows/RTX 2050 primary path) |
| 4-bit NF4 | **Plausible** via bitsandbytes (same stack as SmolVLM2); **not measured** |
| BF16-only | **~4 GB weights** → exceeds comfortable 4 GiB dedicated with vision+KV → **not feasible** on RTX 2050 |

### Memory estimates (analytic, not measured)
| Component | BF16 | NF4 (approx) |
|---|---:|---:|
| Text weights | ~4.0 GB | ~1.1 GB |
| Vision encoder | ~0.4–0.6 GB | partial quant |
| Activations/KV/temp | ~1–2 GB | ~1–2 GB |
| **Total vs 4096 MiB** | **Likely OOM** | **Tight but comparable to SmolVLM2 ~3.3 GiB peak** |

**CONDITIONAL GO** for NF4-only download later; **NO-GO** for BF16 dense on this GPU.

## H. Adapter
`src/models/model_adapter.py`: `ModelSpec` + `VLMAdapter` protocol; SmolVLM2 + Qwen specs (revision unpinned for feasibility only).

## I. Shared eval battery
Object identity, left/right, image swap, text reversal, entailed/contradicted/fork, missing visual/text — same ground truth; model-specific chat templates only. No cross-model raw-logit comparison.

## J. Verification
See session: doctor/tests/portability/compileall; 2.2B intact; no new downloads/GPU runs.

## K. Git
See push SHA.

## L. Recommendation
**CONDITIONAL GO** to *authorize a later NF4-only Qwen download* after pinning a revision and confirming `transformers` 5.20.1 loads `Qwen3VLForConditionalGeneration` without FA2.  
**NO-GO** for BF16 inference or automatic multi-model experiments in this phase.

**Stopped after cleanup + feasibility audit.**
