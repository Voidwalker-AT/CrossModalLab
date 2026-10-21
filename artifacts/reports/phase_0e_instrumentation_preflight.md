# Phase 0E Report — Mechanistic Design & Passive Instrumentation Preflight

**Date:** 2026-10-21  
**Model:** SmolVLM2-2.2B-Instruct @ `482adb537c021c86670beed01cd58990d01e72e4` (NF4)  
**Base artifacts:** Phase 0D.2 (commit `f73d538`) — preserved unchanged  

**Epistemic status:** Instrumentation preflight and experimental design only. **No neural failure mechanism is claimed.**

---

## 1. Phase 0D.2 scientific-integrity audit

| Item | Result |
|---|---|
| Git sync at start | `main` == `origin/main` @ `f73d538`; clean tree |
| Raw confusion recompute | Yes: 10/2/0; No: 10/8/0; Undetermined: 5/1/0 — **matches** published metrics |
| Reversal-query GT | Independent symbolic recompute **6/6 contradicted/No** |
| Source controlled GT | **0 mismatches** vs solver |
| Matched pairs | **6 true pairs** sharing `image_sha256` + inverted queries |
| Text-only controls | **Blank canvas images still processed by vision tower** (1377 image tokens, identical SHA). Condition is **blank-image multimodal**, not pure text path |
| Prompt leakage | No GT labels in premises/queries except false-positive substring “No” in “No additional statement…” |
| CI dependence | Bootstrap is case-level; matched pairs are **not** independent — documented limitation |

**Stop criterion not triggered** (no material GT/input-integrity error).  
**Material caveat carried forward:** do not interpret text controls as vision-bypass baselines.

## 2. Matched-case selection

`configs/phase_0e/matched_case_selection.json` records correct/incorrect forwards, forks, reversal controls, visual controls, blank-canvas controls, and **true matched pairs**. Merely similar failures are explicitly excluded from “matched” status.

## 3. Model component map (inspected)

- `model.model`: `vision_model`, `connector`, `text_model`
- **24** language layers at `model.model.text_model.layers[ℓ]`
- Each layer: `self_attn`, `mlp`, `input_layernorm`, `post_attention_layernorm`
- Residual width **H=2048**; controlled images insert **1377** image tokens

## 4–6. Design artifacts

Full sparse-capture, probe, and causal-patching designs (not executed) are in:

`docs/phase_0e_mechanistic_design.md`

Highlights:
- Sparse positions: last, first/last/mean image, last non-image text  
- Probe protocol requires **≥200 group-disjoint examples** before accuracy claims  
- Causal patching remains **unauthorized** in this phase  

## 7. Passive smoke telemetry (executed)

Script: `scripts/run_phase_0e_passive_smoke.py`  
Artifact: `artifacts/instrumentation/phase_0e/passive_layer_smoke.json`

| Field | Value |
|---|---|
| Sample | `p0d2_ctrl_forward_chain_0019_forward_chain` (correct Yes) |
| Layer | 12 residual |
| Native dtype | `torch.float32` |
| Stored rows | 5 × `[2048]` FP32 on CPU |
| Logits max \|Δ\| vs no-hook | **0.0** (identical) |
| NaN/Inf | **None** |
| Baseline peak VRAM | 3330.7 MiB |
| Capture pass peak VRAM | 3612.1 MiB (two sequential forwards; still < 4096) |
| Full residual FP32 size | 11.71 MiB |
| Selected rows size | 0.039 MiB |
| Intervention | **None** (passive hook only) |
| **Verdict** | **PASS** |

## 8. Hardware feasibility

Sparse multi-layer passive capture is feasible on RTX 2050 with CPU offload of selected rows. Avoid simultaneous full-sequence caches of many layers without freeing between samples. Peak two-forward smoke peaked ~3.6 GiB — headroom exists but is limited.

## 9. Limitations

1. Controlled synthetic N=36 behavioral base  
2. Text control ≠ pure text path  
3. NF4 residual statistics may differ from dense BF16 models  
4. Single prompt template; single GPU  
5. No probe training; no causal patching executed  

## 10. Recommendation

### **CONDITIONAL GO** for Phase 1 mechanistic work **only if**:
1. Grouped sample expansion precedes any probe accuracy publication  
2. Passive multi-layer sparse capture is run first  
3. Activation patching requires **separate explicit approval**  
4. Causal language remains prohibited until pre-registered intervention criteria are met  

### **NO-GO** for claiming an established failure circuit at this stage.

---

## Validation commands (0E)

| Command | Exit |
|---|---|
| `python scripts/audit_phase_0d2_integrity.py` | 0 |
| `ulvo run . -- python -m unittest discover -s tests` | see below |
| `ulvo run . -- python scripts/run_phase_0e_passive_smoke.py` | 0 (PASS) |
| `ulvo project doctor .` | 0 |
