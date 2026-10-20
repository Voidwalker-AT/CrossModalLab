# Decision Record: Research Foundation, Problem Formulation, and Safety Boundaries

**Decision ID:** `ADR-0001`  
**Date:** 2026-10-17 (Updated 2026-10-18)  
**Status:** Accepted  
**Scope:** Phase 0A / D1 (Research Foundation & Dashboard Planning)  

---

## 1. Context & Problem Statement

CrossModalLab was initiated to investigate why compact vision-language Transformers fail to compose relational knowledge across modalities and why they hallucinate answers when presented with insufficient evidence. 

Before introducing code, environments, or model checkpoints, the project required an authoritative foundational framework establishing:
* A formally grounded relational reasoning setting that clearly separates false conclusions from underdetermined conclusions.
* A mechanistic hypothesis framework free from premature causal assumptions.
* Explicit boundaries respecting local hardware constraints (RTX 2050 4 GB VRAM, 16 GB host RAM, Windows 11).
* Strict storage quarantine protecting an ongoing host storage audit on drive `C:`.

---

## 2. Key Decisions & Rationales

### Decision 1: Canonical Setting — 1D Spatial Directional Reasoning
* **Choice:** Formalize reasoning using a single directional relation $R(x, y)$ ("$x$ is strictly to the left of $y$") over distinct geometric primitives.
* **Rationale:** Spatial $\text{left-of}$ is strict, asymmetric, and transitive. It allows exact programmatic verification of ground truth, completely eliminates semantic ambiguity, and cleanly isolates compositional logic from messy real-world commonsense biases.

### Decision 2: Strict Three-Way Epistemic Categorization (Chain vs. Fork)
* **Choice:** Formally categorize relational queries into three mutually exclusive epistemic states:
  1. **Entailed ($\top$):** Forward Chain ($R(A, B) \land R(B, C) \models R(A, C)$).
  2. **Contradicted ($\bot$):** Reverse Chain ($R(B, A) \land R(C, B) \models \neg R(A, C)$).
  3. **Underdetermined ($?$):** Fork ($R(A, B) \land R(C, B) \not\models R(A, C)$ and $\not\models \neg R(A, C)$).
* **Rationale:** Conflating an underdetermined query with a false query is a fatal scientific flaw. Testing whether models abstain on forks isolates *epistemic calibration* from deductive reasoning.

### Decision 3: Competing Mechanistic Hypotheses Over Singular Bias
* **Choice:** Replace preconceived single-cause assumptions with six competing, falsifiable hypotheses ($H_{\text{percep}}$, $H_{\text{text}}$, $H_{\text{comp}}$, $H_{\text{gen}}$, $H_{\text{abstain}}$, $H_{\text{shortcut}}$).
* **Rationale:** Prematurely assuming that failure is localized to cross-attention binding or MLP memory risks confirmation bias. Structuring six falsifiable alternatives ensures objective empirical attribution.

### Decision 4: Rejection of Probe-Only Causal Inferences & Arbitrary Thresholds
* **Choice:** Eliminate arbitrary heuristic thresholds (e.g. `<80%` or `>90%`). Mandate that probe decodability is strictly correlational, and require causal activation interventions (activation patching) alongside path patching and multi-component ablations to substantiate functional claims.
* **Rationale:** A linear probe merely proves that information is extractable by an external observer; it does not prove downstream transformer circuits use that information. Probes failing does not prove feature absence. Activation patching demonstrates causal mediation along a specific path, but does not prove uniqueness without circuit ablation.

### Decision 5: Four-Tier Statistical Decision Framework
* **Choice:** Adopt: (1) 95% bootstrap confidence intervals ($B \ge 1,000$), (2) pre-registered hypothesis tests against chance baselines (binomial tests, McNemar's paired test), (3) Benjamini-Hochberg false discovery rate corrections across multiple components, and (4) practical effect sizes (Normalized Indirect Effect $\ge 0.20$, Cohen's $d$).
* **Rationale:** Eliminates arbitrary magic numbers and aligns the project with publication-grade statistical rigor.

### Decision 6: Strict 4 GB VRAM Boundary & Modular Design
* **Choice:** All empirical pipelines must be designed for batch size 1, forward-only detached activation caching, and compact open-weights architectures ($\le 3\text{B}$ parameters). Primary target candidate is `SmolVLM-500M-Instruct`, with `SmolVLM-2B` as secondary.
* **Rationale:** The laboratory hardware consists of an RTX 2050 Laptop GPU (4096 MiB VRAM). Full model adoption is deferred to Phase 0B empirical memory profiling.

### Decision 7: Absolute Storage Quarantine for Drive C:
* **Choice:** Prohibit all intentional project-related file creations, environment setups, package installs, or model downloads on drive `C:`. Confine all workspace files to `E:\Workspace\Active\CrossModalLab`.
* **Rationale:** Drive `C:` has only ~6 GB free space and is undergoing an active external storage audit. All future caches (`HF_HOME`, `TORCH_HOME`, `UV_CACHE_DIR`) must be explicitly redirected to drive `E:`.

---

## 3. Consequences & Next Steps

* **Positive Consequences:**
  - The repository now has an airtight scientific specification, formal logical definitions, and rigorous interpretability rules.
  - Literature collision risks are explicitly cataloged, preventing false claims of novelty.
  - The physical machine and ongoing C: drive audit are fully protected.
* **Next Immediate Step (Phase 0B / D2):**
  - Establish a clean, isolated virtual environment (`uv venv --python 3.11`) on drive `E:`.
  - Conduct batch-size-1 empirical memory profiling of candidate compact VLMs.
  - Implement the Phase D2 read-only static experiment artifact viewer.
