# Phase 0D.1: Epistemic Correction and Scientific Audit Report

**CrossModalLab — Academic Research Report**  
**Author:** AI Research Engineer (CrossModalLab)  
**Date:** October 18, 2026  
**Document Status:** Official Epistemic Rectification of Phase 0D Preliminary Findings  
**Target Model:** `HuggingFaceTB/SmolVLM2-2.2B-Instruct` (pinned revision `482adb537c021c86670beed01cd58990d01e72e4`, 4-bit NF4)  
**Governance:** GRAVITAS Protocol & Epistemic Labeling Rules

---

## 1. Executive Summary: Purpose of this Rectification

In Phase 0D, preliminary hardware profiling and pilot probes were executed on `SmolVLM2-2.2B-Instruct`. While the engineering metrics (VRAM footprint, module quantization breakdown, hook execution) were empirically verified, the preliminary interpretation over-claimed several scientific findings based on small-sample observations.

In accordance with Section 2 (Core Invariants) and Section 4 (Scientific Integrity) of `AGENTS.md`, this report explicitly rectifies nine preliminary interpretations. The original Phase 0D report (`artifacts/reports/phase_0d_smolvlm2_feasibility_report.md`) is preserved unaltered in the experimental archive to maintain full historical auditability.

---

## 2. Formal Epistemic Rectifications (9 Items)

### Item 1: Counterfactual Swapped-Image Prediction
* **Preliminary Claim in Phase 0D:** Claimed that the model demonstrated visual sensitivity by dynamically responding to an object position swap.
* **Empirical Reality:** On `ctrl_forward_chain_0001.png`, the prompt *"Is the cyan box to the left of the green cylinder?"* received the greedy output `"Yes."` ($p = 0.9961$). When the cyan box and green cylinder were horizontally swapped in the visual canvas, the model's greedy output **remained `"Yes."`** ($p = 0.6719$).
* **Rectified Epistemic Status:** **`FAILURE OF GROUNDING`**. The model did *not* invert its prediction. The greedy prediction was visually insensitive and maintained the affirmative output despite the visual premise being falsified.

### Item 2: Image-Sensitive Logit Modulation $\neq$ Correct Visual Grounding
* **Preliminary Claim in Phase 0D:** Suggested that because logit margin for `" Yes"` dropped from $+19.8$ to $+15.4$ upon swapping objects, the model possessed visual grounding.
* **Empirical Reality:** A reduction in logit magnitude indicates only that the visual tokens perturb the residual stream; it does *not* prove that the model correctly grounded the spatial relationship. When both an authentic image, an inverted image, a single-object image, and a completely blank canvas all elicit the identical greedy output (`"Yes."`), the system exhibits a pervasive language prior override, not verified visual grounding.
* **Rectified Epistemic Status:** **`UNVERIFIED / PRIOR DOMINANCE`**. Logit shifts without categorical accuracy cannot be cited as evidence of successful grounding.

### Item 3: Choice-Order Invariance Scope
* **Preliminary Claim in Phase 0D:** Asserted that `SmolVLM2-2.2B` was "100% choice-order invariant."
* **Empirical Reality:** Choice-order invariance was evaluated across only three cyclic permutations on a *single* underdetermined test instance (`ctrl_underdetermined_fork_0073`).
* **Rectified Epistemic Status:** **`PRELIMINARY OBSERVATION (N=1)`**. Consistency on three permutations for one sample demonstrates that the model did not immediately flip on that specific token sequence. It does *not* establish general choice-order invariance across the dataset, across different entity pairs, or across all six possible permutations ($3! = 6$).

### Item 4: Directional Spatial Competence Scope
* **Preliminary Claim in Phase 0D:** Characterized the model as having "100% spatial competence on left-of."
* **Empirical Reality:** The evaluation tested exactly two queries ($N=2$): one true $\text{left-of}$ query and one false $\text{left-of}$ query on a single canonical layout.
* **Rectified Epistemic Status:** **`PRELIMINARY OBSERVATION (N=2)`**. Two binary responses do not constitute statistical proof of general spatial competence. Formal claims require testing across multiple distances, palettes, distractors, and object arrangements.

### Item 5: Directional Asymmetry ($\text{left-of}$ vs $\text{right-of}$)
* **Preliminary Claim in Phase 0D:** Suggested that poor performance on two $\text{right-of}$ queries justifies excluding $\text{right-of}$ relations from the research project.
* **Empirical Reality:** The model failed on two $\text{right-of}$ queries ($0/2$). However, scientific research investigating cross-modal relational reasoning must evaluate whether this failure stems from token frequency imbalance in pretraining, syntactic parsing bias, or relational representations.
* **Rectified Epistemic Status:** **`OPEN RESEARCH QUESTION`**. While the benchmark engine defines a canonical directional relation $R = \text{left-of}$ for graph reachability, $\text{right-of}$ queries serve as an essential matched linguistic control to detect syntactic relational blindness.

### Item 6: Mechanism of Negation Conflation
* **Preliminary Claim in Phase 0D:** Declared discovery of a "novel scientific mechanism: negation conflation where the model systematically conflates underdetermination with falsehood."
* **Empirical Reality:** Observed on a *single* fork case (`s6_fork_underdetermined`, $N=1$) that the model emitted `"No."` instead of `"Undetermined"`.
* **Rectified Epistemic Status:** **`FALSIFIABLE HYPOTHESIS`**. An error on one test case is an observed behavioral failure, not a proven internal neural mechanism. Mechanistic attribution requires demonstrating this failure across a controlled distribution and identifying causal mediation via internal circuit components.

### Item 7: Instrumentation Passivity $\neq$ Causal Attribution
* **Preliminary Claim in Phase 0D:** Described the Layer 12 forward hook as proof of readiness for causal mechanism attribution.
* **Empirical Reality:** Registering a forward hook and confirming that $\Delta_{\text{logits}} = 0.0$ proves only *software passivity* and *memory compatibility* (the hook does not corrupt tensors or crash CUDA).
* **Rectified Epistemic Status:** **`ENGINEERING VALIDATION ONLY`**. Passive observation hooks demonstrate that activations can be non-destructively extracted. They provide zero evidence regarding how the model performs reasoning or which components are causally necessary.

### Item 8: Precision Specification of Activation Tensors
* **Preliminary Claim in Phase 0D:** Loosely described activations as `torch.float32`.
* **Empirical Reality:** Inside the GPU computational graph, `LlamaDecoderLayer` operates in `bfloat16`. The tensor captured at Layer 12 is natively `torch.bfloat16`. To perform CPU inspection, variance calculations, and NumPy serialization without CUDA memory pinning, the captured tensor was explicitly detached and cast via `.cpu().to(torch.float32)`.
* **Rectified Epistemic Status:** **`HARDWARE SPECIFICATION`**:
  - In-GPU Residual Stream: `torch.bfloat16` ($[1, L, 2048]$).
  - Host CPU Inspection Tensor: `torch.float32` ($[1, L, 2048]$).
  - Weights: 4-bit NormalFloat4 (NF4) via `bitsandbytes`.

### Item 9: Storage Footprint: Checkpoint vs Cache
* **Preliminary Claim in Phase 0D:** Conflated checkpoint size with total disk allocation.
* **Empirical Reality:**
  - Raw immutable safetensors checkpoint files: **$8.375\text{ GB}$** ($4.96\text{ GB}$ part 1 + $4.03\text{ GB}$ part 2 + metadata).
  - Hugging Face cache structure on Drive `E:`: **$8.375\text{ GB}$** under `E:\cache\huggingface\models--HuggingFaceTB--SmolVLM2-2.2B-Instruct`.
  - In-memory VRAM allocation after 4-bit NF4 quantization: **$1,784.5\text{ MiB}$** static.
* **Rectified Epistemic Status:** **`TECHNICAL SPECIFICATION`**. Explicitly separate disk storage, download payload, and VRAM residency.

---

## 3. Summary of Epistemic Status Post-Correction

```
+------------------------------------+--------------------------+---------------------------+
| Preliminary Phase 0D Claim         | Flaw / Confound          | Rectified Epistemic Label |
+------------------------------------+--------------------------+---------------------------+
| Swapped image inverted prediction  | Output remained "Yes."   | FALSIFIED (No Inversion)  |
| Logit drop proves visual grounding | Output identical on blank| REJECTED (Prior Override) |
| 100% choice-order invariant        | Evaluated on N=1 sample  | PRELIMINARY (Unproven)    |
| 100% left-of spatial competence    | Evaluated on N=2 samples | PRELIMINARY (Unproven)    |
| Exclude right-of from research     | Conflates spec with test | RETRACTED (Retain as Ctrl)|
| Discovered negation conflation     | N=1 behavioral error     | HYPOTHESIS ONLY           |
| Hook proves causal attribution     | Hook is passive code     | ENGINEERING VALIDATION    |
| Activations are float32            | GPU tensor is bfloat16   | TECHNICAL CORRECTION      |
+------------------------------------+--------------------------+---------------------------+
```
