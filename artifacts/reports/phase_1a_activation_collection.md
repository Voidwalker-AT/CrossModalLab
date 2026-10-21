# Phase 1A Report — Sparse Multi-Layer Activation Collection

**Label:** DESCRIPTIVE / NON-CAUSAL  
**Model:** SmolVLM2-2.2B-Instruct @ `482adb537c021c86670beed01cd58990d01e72e4` (NF4)  
**Date:** 2026-10-21  
**Prior commit:** `e52389a` (Phase 0E)

> Activation differences are **not** causal mechanisms. Mean image vectors are **aggregates**, not neurons.

---

## A. Files created / modified

| Path | Role |
|---|---|
| `src/instrumentation/multi_layer_capture.py` | Multi-layer passive capture + taxonomy helpers |
| `configs/phase_1a/activation_case_selection.json` | Frozen 10-case selection + control taxonomy |
| `scripts/run_phase_1a_activation_collection.py` | Collection runner (smoke + resume) |
| `scripts/analyze_phase_1a_activations.py` | Descriptive analysis (L2, cosine) |
| `tests/test_phase_1a_capture.py` | Unit tests |
| `artifacts/instrumentation/phase_1a/*` | Meta JSONL, vectors, analysis, CSV, true-text diagnostics |
| `artifacts/reports/phase_1a_activation_collection.md` | This report |

## B. Scientific control corrections

| Label | Definition | Phase 0D.2 mapping |
|---|---|---|
| **BLANK_IMAGE_MULTIMODAL** | Blank canvas still emits **1377 image tokens** + `pixel_values` | Former `text_premise_control` |
| **TRUE_TEXT_ONLY** | No image tokens, no `pixel_values` | **New diagnostic path in 1A** (2 prompts) |
| **IMAGE_MULTIMODAL** | Relational controlled canvases | chains / forks / reversals / visual controls |

Original Phase 0D.2 artifacts **unchanged**. Blank-image results are **not** reinterpreted as pure text-only performance.

## C. Sample selection (n=10 multimodal)

From `configs/phase_1a/activation_case_selection.json`:

- Correct forwards: `…0019`, `…0010`  
- Incorrect reversals: `revctl_…0019`, `revctl_…0010`  
- Incorrect forward + correct paired reversal: `…0004` / `revctl_…0004`  
- Incorrect forks: `…0073`, `…0083`  
- Correct visual control: `…0122`  
- Blank-image control: `…0164` (**BLANK_IMAGE_MULTIMODAL**)

True matched pairs share `image_sha256` (forward↔reversal). Forks are **not** pairwise counterfactuals.

## D. Layer / token configuration

- Layers: **3, 12, 18, 23** residual (decoder layer output)  
- Selectors: `last`, `first_image`, `last_image`, `mean_image`, `last_text_non_image`  
- Hooks: simultaneous passive forward hooks; output returned unchanged  

## E. Tensor shapes / dtypes

| Item | Observed |
|---|---|
| Native residual dtype | `torch.float32` |
| Stored dtype | `float32` CPU lists |
| Per-selector shape | `[2048]` |
| Sequence length | ~1493–1510 |
| Image token count | **1377** (multimodal & blank-image); **0** (true text-only) |

## F. Actual measurements (summary)

- **10/10** multimodal captures `ok` with layers `{3,12,18,23}`  
- All logits max\|Δ\| vs unhooked baseline: **0.0**  
- No NaN/Inf  
- True text-only diagnostics: **2** prompts, `image_token_count=0`, no `pixel_values`  
- Descriptive artifacts: `descriptive_analysis_v1.json`, `activation_l2_summary_v1.csv`  

### Correct vs incorrect last-token L2 (descriptive only)

| Layer | mean L2 correct | mean L2 incorrect |
|---|---:|---:|
| L12 | 94.910 | 95.050 |

Difference is **tiny** and must not be read as a confidence or causal signal.

### Matched-pair cosine
36 rows (3 pairs × 4 layers × 3 selectors) — same-image forward vs reversal cosines available in analysis JSON. **Not** evidence of logical understanding.

## G. Descriptive layer comparisons
Layer-depth last-token L2 trajectories per sample: `descriptive_analysis_v1.json` → `layer_depth_last_token_l2`.  
N=10; DESCRIPTIVE ONLY.

## H. Output equivalence
Passive multi-hook path: **logits identical (max\|Δ\|=0.0)** on all captures (tolerance documented as `1e-5`). Hooks removed after each sample.

## I. Peak memory

| Metric | Value |
|---|---:|
| Peak PyTorch allocated VRAM | **3614.2 MiB** (max across cases) |
| Host RAM during capture | recorded per sample in meta JSONL |
| Full residual (planning) | ~11.7 MiB/layer FP32 if ever stored (not stored) |
| Selected rows | ~0.039 MiB/layer |

No OOM; 4-layer simultaneous passive hooks **feasible**.

## J. Tests and commands

| Command | Exit |
|---|---|
| `ulvo project doctor .` | 0 |
| `ulvo run . -- python -m unittest discover -s tests` | 0 (**83+ tests OK**) |
| `python -m unittest discover -s tests` (system) | 0 (torch tests skipped as applicable) |
| `ulvo run . -- python scripts/check_portability.py` | 0 |
| `ulvo run . -- python -m compileall -q src scripts tests` | 0 |
| `ulvo run . -- python scripts/run_phase_1a_activation_collection.py --smoke` | 0 |
| `ulvo run . -- python scripts/run_phase_1a_activation_collection.py` | 0 |
| `ulvo run . -- python scripts/analyze_phase_1a_activations.py` | 0 |

## K. Git
See push section of session report (commit after tests).

## L. Remaining confounds

1. N=10 curated cases — not representative  
2. NF4 quantization residual statistics  
3. Single prompt template family  
4. Mean-pool image vector discards spatial structure  
5. Blank-image vs true-text contrast is new and small (2 pure-text prompts)  
6. Cosine/norm differences are correlational descriptions only  

## M. Recommendation

**CONDITIONAL GO** to continue **descriptive multi-layer collection** and design-only probe planning.  

**NO-GO** for probe training or causal patching without separate approval and larger group-disjoint sample collection.

**STOP after Phase 1A.**
