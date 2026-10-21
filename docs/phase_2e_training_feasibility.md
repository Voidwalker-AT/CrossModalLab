# Phase 2E — Training Feasibility, Losses, Baselines & Experiment Design

**No GPU training executed in this phase.**

---

## 1. QLoRA / LoRA feasibility (RTX 2050 4 GB)

### Measured inference anchors (NF4)
| Model | Peak torch alloc (inference) | nvidia-smi observed |
|---|---:|---:|
| SmolVLM2-2.2B | ~1.8–3.3 GiB | ≤~3.9 GiB |
| Qwen3-VL-2B | ~1.5–2.5 GiB | ≤~3.1 GiB |

### Training memory model (analytic, **not measured**)
Training adds **activations + gradients + optimizer state** on top of frozen NF4 weights:

| Component | LoRA (fp16/bf16 base) | QLoRA (NF4 base) |
|---|---|---|
| Frozen weights | ~4.4 GB (2.2B BF16) **too large** | ~1.1–1.5 GB |
| Vision encoder | +0.3–0.6 GB | same / partial quant |
| LoRA grads+opt (r=16, few layers) | ~50–200 MB | same |
| Activations @ T≈1500, H=2048, full backprop | **many GB** | **many GB** |
| With **gradient checkpointing** + batch 1 | Often still **>4 GB** on multimodal long seq | **Marginal** |

**Conclusions (conservative):**
1. **Full BF16 LoRA of 2.2B does not fit** 4 GB.  
2. **NF4 inference ≠ QLoRA training fit.** Optimizer + activations dominate.  
3. **QLoRA on full 1500-token multimodal sequences: UNVERIFIED / likely OOM** without extreme measures (vision freeze, rank, checkpointing, short crops, CPU offload).  
4. **Text-only or short-image** adaptation is the **only plausibly safe** first training probe on this GPU.  
5. Fallback architectures (0.5–1B VLMs) may be required for true on-device training — **not authorized this phase**.

### Recommended feasibility order (future, authorized later)
1. Text-only LoRA on ablation prompts (T≪500)  
2. Multimodal LoRA with frozen vision + checkpointing + T≤512 image tokens  
3. Full contrast-v2 sequences only if (2) measured under ~3.5 GiB allocated  

**Stop conditions:** torch allocated >3800 MiB; host available <1.5 GiB; repeated OOM.

---

## 2. Candidate losses (CPU-defined)

### L1 — Cross-entropy (baseline)
\[
\mathcal{L}_{\mathrm{CE}}=-\log p_\theta(y^*\mid x)
\]
Labels: Yes/No/Undetermined (or format-mapped). Full gradient into trainable adapters only.

### L2 — Counterfactual consistency
For pair \((x,x')\) with labels \(y\neq y'\) (or same-label pairs when required):
\[
\mathcal{L}_{\mathrm{CF}}=\mathrm{KL}\big(p_\theta(\cdot\mid x)\,\|\,q_{y}\big)+\mathrm{KL}\big(p_\theta(\cdot\mid x')\,\|\,q_{y'}\big)
\]
with \(q_y\) one-hot or temperature-scaled. Requires **exact-image** pairs where mandated.

### L3 — Evidence-aware abstention
For logical sufficiency \(s\in\{0,1\}\) (solver: underdetermined vs definite):
\[
\mathcal{L}_{\mathrm{EA}} = s\cdot\mathrm{CE}(y,\text{Undetermined}) + (1-s)\cdot\mathrm{CE}(y,\mathrm{Yes/No})
\]
or margin:
\[
\mathcal{L}_{\mathrm{m}}=\max\big(0,\; m+z_{\mathrm{def}}-z_{\mathrm{und}}\big)\ \text{when } s=1
\]

### L4 — Combined
\[
\mathcal{L}=\mathcal{L}_{\mathrm{CE}}+\lambda_{\mathrm{cf}}\mathcal{L}_{\mathrm{CF}}+\lambda_{\mathrm{ea}}\mathcal{L}_{\mathrm{EA}}
\]
Hyperparameters: \(\lambda\in\{0.1,0.25,0.5\}\); margin \(m\in\{0.5,1.0\}\).  
**Negative controls:** shuffle \(s\); \(\lambda=0\); train on labels permuted within family.  
**Failure cases:** format mismatch (2D); over-abstention hurting supported Yes/No; vision collapse.

**Novelty:** combined loss is **not claimed novel** (see novelty audit).

---

## 3. Baselines & experiment design

| Arm | Description |
|---|---|
| B0 | Frozen base VLM + neutral abstain prompt (2D protocol) |
| B1 | Prompt-only few-shot / chain-of-thought (no weight updates) |
| B2 | Standard LoRA/QLoRA + CE only |
| B3 | B2 + \(\mathcal{L}_{\mathrm{CF}}\) |
| B4 | B2 + \(\mathcal{L}_{\mathrm{EA}}\) |
| B5 | Combined B2+CF+EA |
| B6 | External symbolic verifier (no VLM update): solver/post-check forces Undetermined when underdetermined |

**Metrics:** balanced accuracy; macro-F1; abstention P/R; unsupported definite rate; accuracy on **fully supported** queries; counterfactual consistency; held-out template/scene generalization; train/infer cost (GPU-hours, VRAM).

**Falsification:**
- B5 does **not** beat B0 on fork abstention **without** dropping supported-query BA by > pre-registered δ (e.g. 5 pp).  
- CF term does not improve exact-image pair consistency vs CE-only.  
- B6 matches or beats B3–B5 → prefer symbolic baseline over training complexity.

---

## 4. CPU validation utilities
Implemented (no VLM training):
- `src/training/losses.py` — CE, EA, CF on synthetic tensors  
- `src/training/schema.py` — schema, split leakage, balance, group integrity  
- `scripts/validate_phase_2e_dataset_schema.py`  
- Tests: numerical stability, leakage detection, config serialization  
