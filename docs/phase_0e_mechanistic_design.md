# Phase 0E — Mechanistic Experiment Design (No Causal Execution)

**Status:** Design + instrumentation preflight only  
**Model:** `HuggingFaceTB/SmolVLM2-2.2B-Instruct` @ `482adb537c021c86670beed01cd58990d01e72e4` (4-bit NF4)  
**Behavioral basis:** Phase 0D.2 36-case controlled pilot (manifest SHA `b13c4bcc…a46da`)

> **Epistemic boundary:** Nothing in this document establishes a neural failure mechanism.  
> Probe decodability ≠ causal model usage. Activation patching (when later authorized) ≠ unique circuit proof.

---

## 1. Phase 0D.2 scientific-integrity audit (summary)

| Check | Result |
|---|---|
| Confusion recomputed from raw JSONL | Matches report (Yes 10/2/0, No 10/8/0, Undet 5/1/0) |
| Reversal-query GT independent solver recompute | **6/6 match** (all contradicted / No) |
| Source controlled GT re-audit | **0 mismatches** |
| True matched pairs (shared `image_sha256`) | **6** forward↔reversal pairs |
| Text-only controls | Blank-canvas **images still enter vision tower** (1377 image tokens); **not** pure no-image text path |
| Answer-label leakage in premises/queries | Only false positive: `"No additional statement is provided."` contains the word “No” |
| Prompt template | Same instructional Yes/No/Undetermined list for all conditions (task vocabulary, not GT leakage) |
| Bootstrap CIs | Case-level resampling (B=2000); matched pairs share entities → dependence under-stated if treated as independent |

**No material GT/input-integrity error requiring a stop.**  
**Material design caveat:** “text-only” condition is **blank-image multimodal**, not text-only architecture bypass.

---

## 2. Matched-case selection

See `configs/phase_0e/matched_case_selection.json`.

| Role | IDs (examples) | Relation to hypothesis |
|---|---|---|
| Correct forward composition | `…0019`, `…0010` | Baseline composition success |
| Incorrect forward composition | `…0004`, `…0009` | Composition error under identical protocol |
| Incorrect forks (abstention failure) | `…0073`, `…0083`, `…0093` | Evidence-insufficiency failure |
| Incorrect reversal controls | `…revctl_…0019/0010/0003` | Query inversion fails despite shared canvas |
| Correct visual single-premise | visual controls 0122/0117 | Perception competence control |
| Blank-canvas controls | text 0164/0151 | Multimodal blank-image baseline (not pure text) |

**True matched counterfactual pairs:** same `image_sha256`, inverted query entities (6 groups).  
**Merely similar:** different forks/images; visual controls are not composition counterfactuals.

---

## 3. Model component map (verified by inspection)

| Component | Path | Notes |
|---|---|---|
| Multimodal wrapper | `model.model` | children: `vision_model`, `connector`, `text_model` |
| Language backbone | `model.model.text_model` | **24** decoder layers |
| Decoder layer ℓ | `model.model.text_model.layers[ℓ]` | `self_attn`, `mlp`, `input_layernorm`, `post_attention_layernorm` |
| Residual stream | hidden states `[1, T, H]` | **H = 2048** |
| Vision tokens | `<image>` placeholders | **1377** for 384×384 controlled canvases |
| Unembedding | `lm_head` | logits over vocab at last position |

Observed param dtype after NF4 load: embeddings/parameters may appear `float32`; quantized linears use bitsandbytes NF4 with bfloat16 compute (as pinned).

---

## 4. Sparse activation capture specification

### 4.1 Proposed capture sites (later study; not fully executed in 0E)

| Tier | Layer ℓ | Module | Why |
|---|---|---|---|
| Early | 3 | `layers[3]` residual | Early visual/text binding |
| Mid | 12 | `layers[12]` residual | Prior Phase 0D.1 hook surface |
| Mid-late | 18 | `layers[18]` residual | Pre-decision composition |
| Late | 23 | `layers[23]` residual | Near-unembedding decision |

Optional sublayers (when VRAM allows, still passive):
- `layers[ℓ].self_attn` output (attention path)
- `layers[ℓ].mlp` output (MLP path)

### 4.2 Token positions (sparse)

| Selector | Meaning | Addresses |
|---|---|---|
| `last` | Final prompt token (pre-answer) | Decision state for Yes/No/Undetermined |
| `first_image` / `last_image` | Boundary visual tokens | Visual binding quality |
| `mean_image` | Mean-pooled image tokens | Global canvas summary (lossy) |
| `last_text_non_image` | Last non-image token | Text premise / question encoding |

### 4.3 Shapes and memory

Full residual at T=1425, H=2048, FP32:

\[
1425 \times 2048 \times 4 \approx 11.68\ \text{MiB per layer}
\]

Storing 4 layers × full sequence ≈ 46.7 MiB (acceptable), but **not** required for smoke.

Selected 5 positions × 2048 × 4 B ≈ 40 KiB per layer — preferred for exploratory work.

**CPU offload strategy:** inside forward hook, clone selected rows `.detach().to("cpu", float32)`; return original output unchanged; never retain GPU autograd graph (`inference_mode`).

### 4.5 Representation classes

| Class | Definition in this project |
|---|---|
| Image-token representation | Residual rows at `<image>` positions |
| Text-token representation | Residual rows at non-image text positions |
| Final question-token representation | `last` position residual |
| Residual stream | Decoder layer hidden states |
| Attention output | `self_attn` module output (tuple-safe) |
| MLP output | `mlp` module output |

---

## 5. Probe design (diagnostic only)

| Probe | Labels | Input features | Chance baseline |
|---|---|---|---|
| Visual direction | left-right entity order on canvas (from metadata) | `mean_image` / boundary image residuals at early layers | 0.5 (2-way) |
| Logical entailment | Yes vs not-Yes on forward/reverse | `last` residual at mid/late layers | majority / 0.5 if balanced |
| Contradiction | No vs not-No | `last` residual | 0.5 if balanced |
| Evidence insufficiency | Undetermined vs definite | `last` residual; also fork vs chain groups | 0.5 if balanced |

**Protocol requirements:**
- **Grouped split by `matched_group_id` and by source entity set** — never put both sides of a true pair in train and test.
- **Capacity:** linear probe (multinomial logistic / ridge); no deep MLP in the first pass.
- **Uncertainty:** bootstrap CI on test accuracy; permutation test vs chance.
- **Negative controls:** probe on random Gaussian features of same dimension; probe on token-position-matched untrained noise.
- **Sample-size gate:** Phase 0D.2 has only 36 cases — **insufficient** for held-out probe accuracy claims.  
  **Requirement before probe scoring:** collect ≥200 independent grouped examples (or pre-registered expansion of controlled generator) with group-disjoint splits.

Probe success only shows **linear accessibility**, not that the model **uses** the feature for generation.

---

## 6. Causal intervention design (DO NOT EXECUTE in 0E)

### 6.1 Abstention / fork intervention
- **Clean:** correct visual single-premise or entailed chain (non-fork).
- **Corrupted:** underdetermined fork with same entities where possible.
- **What changes:** missing second edge in relational graph (fork) vs complete chain; same canvas entity inventory when matched.
- **Token alignment:** identical prompt template; image token count must match (1377); patch only at aligned residual positions (`last`, or pooled image).
- **Target:** residual at layers {12, 18} — replace clean residual rows into corrupted run (or vice versa) at selected positions.
- **Operation:** `h_corrupt[pos] ← h_clean[pos]` (activation patching), FP32 cast policy recorded.
- **Outcome:** logit margin \(z_{\text{Undetermined}} - \max(z_{\text{Yes}}, z_{\text{No}})\) and argmax label change rate.
- **Negative controls:** random layer; random token position; mean-ablate from unrelated sample.
- **Confounds:** token-count mismatch; prompt wording; quantization noise; position identity confounds (last token always answer prefix).
- **Falsification:** patch fails to shift margin above pre-registered effect size / CI excluding 0; or equally shifts unrelated conditions.

### 6.2 Composition / reversal intervention
- **Clean:** correct forward chain vs **its** true reversal control (same image).
- **What changes:** query entity order only (text), image fixed.
- **Target:** late residual at question token; optionally mid residual.
- **Outcome:** Yes/No logit margin \(z_Y - z_N\).
- **Falsification:** patching mid-layer states does not differentially affect matched pair vs mismatched control pairs.

No causal circuit is claimed by this design alone.

---

## 7. Hardware feasibility

| Item | Estimate |
|---|---|
| Peak inference allocation (0D.2) | ~3330 MiB |
| Free VRAM headroom | ~700 MiB |
| Full residual 4 layers FP32 | ~47 MiB |
| Selected-row smoke | negligible |
| Feasibility | **OK for sparse passive capture**; full multi-layer caches still need sequential free between samples |

---

## 8. Smoke test intent (Phase 0E execution scope)

One passive capture at layer 12 on verified sample `p0d2_ctrl_forward_chain_0019_forward_chain`:
- compare logits with/without hook
- record dtype/shape/NaN
- CPU offload selected rows only
- no probe, no patching

---

## 9. Limitations

1. N=36 controlled geometric scenes only  
2. Text control is blank-image multimodal  
3. CIs ignore matched-pair dependence  
4. NF4 quantization may affect residual statistics vs BF16 baselines  
5. Single GPU / single prompt template  
6. Mechanistic claims remain open hypotheses  

---

## 10. Recommendation

**CONDITIONAL GO** for a Phase 1 mechanistic study that:
1. Expands grouped sample size before any probe accuracy reporting  
2. Runs passive multi-layer sparse captures next  
3. Requires separate authorization before any activation patching  
4. Keeps causal language out until intervention criteria are met  
