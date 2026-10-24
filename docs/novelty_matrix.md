# Novelty Collision Matrix & Prior-Art Audit

**Document ID:** `NOV-0001`  
**Phase:** Phase 0A / D1 (Foundation & Specifications)  
**Status:** Living Audit  

> [!WARNING]
> **Scientific Integrity Notice:**
> Novelty is **not** an established fact. It is an open, empirical hypothesis. The following matrix audits direct prior-art threats to identify areas of potential collision and define the precise evidentiary burden required before asserting original scientific contributions.

---

> Phase 3 update (2026-10-24): this is a historical design matrix. Its descriptions of our proposed interventions are not completed findings. [The current collision review](phase_3_novelty_audit.md) adds direct recent overlap; novelty remains UNVERIFIED.

## 1. Prior-Art Collision Matrix

| Prior Work / Benchmark | Core Paradigm & Focus | Target Modality & Scale | Mechanistic Method | Incomplete Evidence / Fork Treatment | Collision Risk | Key Differentiation in CrossModalLab |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Winoground**<br>(Thrush et al., 2022) | Pairwise visio-linguistic compositionality | Multimodal (Image + Caption) | Black-box behavioral metrics (Text, Image, Group score) | **None** (Binary forced-choice retrieval) | **Low** | CrossModalLab uses multi-premise reasoning chains ($A \to B \to C$) and performs layer-wise internal causal interventions. |
| **ARO Benchmark**<br>(Yuksekgonul et al., 2023) | Relation and attribution binding ("bag of words") | Multimodal (Image + Text) | Black-box contrastive score evaluation | **None** (Hard negative ordering) | **Low** | ARO assesses 1-hop relation sensitivity; CrossModalLab investigates multi-premise compositional synthesis and epistemic abstention. |
| **MMVP / Eyes Wide Shut**<br>(Tong et al., 2024) | Visual perception blind spots in CLIP encoders | Multimodal (MLLMs) | Feature-space analysis & behavioral probing | **None** (Perceptual discrimination) | **Low** | Focuses on visual encoder representation deficits; CrossModalLab targets cross-modal composition across multimodal premises. |
| **Faith and Fate**<br>(Dziri et al., 2023) | Compositional limits in multi-step Transformer reasoning | Text-only LLMs | Computation graph analysis, complexity bounds | **None** (Only evaluates fully-specified tasks) | **Medium** | Dziri et al. is unimodal (text-only) and theoretical/behavioral; CrossModalLab extends to cross-modal premises and performs internal activation patching. |
| **Too Late to Recall**<br>(Venhoff et al., 2025) | Two-hop visual factual recall delay | Multimodal (VLMs + LLM backbones) | Attribution patching, linear probing, activation patching | **None** (All queries have deterministic factual answers) | **High** | Venhoff et al. studies parametric recall from single images; CrossModalLab studies synthetic multi-premise relational deduction and explicitly evaluates underdetermined forks. |
| **When RAG Hurts**<br>(Zhao et al., 2026) | Attention distraction from retrieved text over images | Multimodal (LVLMs + RAG) | Attention map inspection & decoding intervention | **None** (Assumes answer is determinable) | **Medium** | Zhao et al. analyzes external text retrieval in LVLMs; CrossModalLab isolates internal multi-premise relational integration and tests epistemic abstention. |
| **Causal Route Gating**<br>(Cheng et al., 2026) | Prior dominance vs visual grounding hallucinations | Multimodal (LVLMs) | Causal route attribution across attention heads | **None** (Object hallucination in generation) | **Medium** | Cheng et al. targets token-level hallucination in descriptive generation; CrossModalLab targets formal relational reasoning and underdetermined logical graphs. |
| **Back Attention**<br>(Yu et al., 2025) | Conflicting logits in multi-hop language reasoning | Text-only LLMs | Logit flow tracking, backward attention routing | **None** (Multi-hop QA with known targets) | **Low** | Text-only multi-hop extraction; CrossModalLab addresses cross-modal binding between vision and text. |
| **Do VLMs Have Internal WMs?**<br>(Gao et al., 2025) | Perception vs prediction benchmark in VLMs | Multimodal (Large commercial & open VLMs) | Behavioral benchmark (WM-ABench) | **Partial** (Tests unanswerable visual questions at behavioral level) | **Medium** | Gao et al. is purely behavioral; CrossModalLab performs layer-by-layer mechanistic interpretability and causal activation patching. |
| **Why Spatial Reasoning Hard?**<br>(Chen et al., 2025) | Attention focus areas on spatial relations | Multimodal (VLMs) | Attention heatmap analysis & AdaptVis decoding | **None** (Single-image 1-hop spatial relations) | **Medium** | Relies on attention visualizations (correlational); CrossModalLab applies causal interventions on multi-premise relational chains. |
| **CF-VLM**<br>(Zhang et al., 2025) | Counterfactual fine-tuning for compositionality | Multimodal (VLMs) | Counterfactual data augmentation & training | **None** (Fine-tuning objective) | **Low** | CF-VLM is a training/fine-tuning approach; CrossModalLab is a diagnostic mechanistic interpretability study of pretrained representations. |
| **MoHoBench**<br>(Zhu et al., AAAI 2026) | Honest refusal on unanswerable visual questions | Multimodal (28 MLLMs) | Behavioral refusal benchmark (12,000+ samples) & SFT/DPO alignment | **Behavioral Refusal** (Evaluates "I don't know" output rates) | **High (Behavioral)** | Establishes that models fail to refuse unanswerable visual questions; CrossModalLab investigates the *internal residual stream circuit* and why readout fails. |
| **VLM-DeflectionBench**<br>(ACL 2026) | Deflection and hallucination under incomplete/noisy evidence | Multimodal (20 LVLMs in RAG) | Behavioral deflection benchmark (2,775 samples) | **Behavioral Deflection** | **High (Behavioral)** | DeflectionBench studies open-domain RAG deflection; CrossModalLab investigates internal multi-premise relational composition ($A \to B \to C$). |
| **CMC-Bench**<br>(ACL MAGMaR 2026) | Cross-modal image-text evidence conflicts | Multimodal (ChartQA & MMMU) | Accuracy degradation under conflict ($\Delta\text{Acc} = 17\text{--}46\%$) | **Conflict Evaluation** | **High (Conflict)** | CMC-Bench evaluates retrieval conflict in documents/charts; CrossModalLab investigates multi-hop deductive chaining and topological underdetermination. |
| **R-Bench**<br>(Wu et al., ICML 2024) | Relationship hallucinations (spatial & action) in images | Multimodal (LVLMs) | Image-level & instance-level evaluation, RA2M alignment | **None** (Pairwise visual relations) | **Medium** | R-Bench evaluates 1-hop image relations; CrossModalLab addresses multi-hop cross-modal composition ($R_{\text{vis}}(A, B) \land R_{\text{text}}(B, C) \models R(A, C)$). |
| **HallusionBench**<br>(Guan et al., CVPR 2024) | Language hallucination vs visual illusion | Multimodal (15 LVLMs) | Control-group paired question-image evaluation | **None** (Consistency testing) | **Medium** | Guides our counterfactual quadruplet design; CrossModalLab adds internal activation patching and layer-wise probing. |
| **PIH-Heads**<br>(Rudman et al., ACL 2026) | Mechanisms of prompt-induced hallucination | Multimodal (3 open VLMs) | Attention head tracking and causal head ablation | **None** (Object counting with false prompt count) | **High (Methodological)** | Proves attention head ablation works in VLMs; CrossModalLab applies activation patching to multi-premise relational reasoning rather than counting. |
| **Premise Blindness**<br>(Xu et al., 2026) | Acceptance of false textual premises over images | Multimodal (LVLMs) | False-premise prompt injection analysis | **None** (False premise injection) | **Medium** | Studies conversational prompt override; CrossModalLab studies deductive reasoning over incomplete premise graphs. |

---

## 2. Synthesis of Collision Threats

### 1. High Methodological Threat: Attention Head Ablation (*PIH-Heads*, ACL 2026)
* **Threat:** Rudman et al. already showed that specific attention heads mediate prompt-copying over visual evidence and that ablating them reduces hallucination by $\ge 40\%$.
* **Our Distinct Niche:** PIH-Heads focuses on *object counting* where the prompt asserts an inflated number. CrossModalLab studies *multi-hop relational composition* ($R(A, B) \land R(B, C) \models R(A, C)$) and *topological underdetermination*, addressing whether the residual stream forms the composed relationship or fails at the readout head.

### 2. High Behavioral Threat: Evidence Insufficiency & Honesty (*MoHoBench* & *VLM-DeflectionBench*, 2026)
* **Threat:** MoHoBench and VLM-DeflectionBench already prove that VLMs fail to abstain on incomplete or unanswerable evidence.
* **Our Distinct Niche:** These benchmarks are purely behavioral (evaluating string responses). CrossModalLab investigates *internal representation*: does the residual stream contain an identifiable latent direction that distinguishes complete chains from underdetermined forks? If so, why does the unembedding head collapse into forced directional guessing?

### 3. High Conflict Threat: Cross-Modal Disagreement (*CMC-Bench*, 2026)
* **Threat:** Catapang et al. already show accuracy drops when text contradicts images.
* **Our Distinct Niche:** CMC-Bench evaluates retrieval contradictions on charts and documents. CrossModalLab evaluates *logical compositionality* where premises are not in direct conflict, but are individually insufficient to entail the query.

---

## 3. Evidentiary Burden for Claiming Novelty

To claim novel scientific contributions, CrossModalLab must strictly avoid claiming novelty for:
- Observing that VLMs make relational errors (established by R-Bench).
- Observing that VLMs fail to abstain on incomplete evidence (established by MoHoBench, DeflectionBench).
- Observing that text priors overpower visual evidence (established by CMC-Bench, Causal Route Gating).
- Performing attention head ablation on VLMs (established by Rudman et al., PIH-Heads).

Potential mechanistic differentiation would require the following new evidence; none is established as CrossModalLab's novelty:
1. **The Mechanistic Locus of Transitive Composition:** Demonstrating which specific layers/modules bind $R_{\text{vis}}(A, B)$ and $R_{\text{text}}(B, C)$ into $R(A, C)$.
2. **Double Dissociation of Representation vs. Readout:** Proving whether underdetermined forks are represented as topologically distinct states in latent space despite behavioral guessing.
3. **Causal Mediation of Failure:** Quantifying the indirect effect of candidate attention heads on relational failure via activation patching with denominator instability controls.
