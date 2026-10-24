# Systematic Literature Review: Compositionality, Mechanistic Interpretability, and Evidence Insufficiency

**Document ID:** `LIT-0001`  
**Phase:** Phase 0A / D1 (Foundation & Specifications)  
**Status:** Living Review  
**Integrity Rule:** Every cited entry explicitly records its verification depth. Abstract-only findings are never represented as full-text verified conclusions. Zero placeholder links are permitted.

---

> Phase 3 update (2026-10-24): historical entries below retain their original inspection scope. Use [the fresh targeted collision audit](phase_3_novelty_audit.md) for current contribution decisions; proposed mechanisms and novelty are not established.

## 1. Review Structure & Verification Taxonomy

To prevent citation hallucination and unfounded novelty claims, all literature entries are categorized under a four-tier verification rubric:
* **Verified Bibliographic Metadata:** Title, authors, venue, year, and DOI/arXiv identifier confirmed via official registries.
* **Inspection Level:**
  * `Abstract Inspected`: Core problem, claimed methods, and reported high-level results reviewed.
  * `Full-Text Inspected`: Complete experimental setups, ablation studies, and mathematical appendices reviewed.
  * `Methodology Verified`: Technical validity of proposed interventions or datasets independently confirmed against code or deep text audit.
* **Direct Overlap:** Specific point of convergence with CrossModalLab's cross-modal relational reasoning or evidence insufficiency setting.
* **Unresolved Questions:** Critical ambiguities, untested assumptions, or boundaries left open by the paper.

---

## 2. Recent Direct Prior-Art Audits (2025–2026)

### Paper 1: Too Late to Recall: Explaining the Two-Hop Problem in Multimodal Knowledge Retrieval
* **Bibliographic Metadata:** Constantin Venhoff, Ashkan Khakzar, Sonia Joseph, Philip H. S. Torr, Neel Nanda. *Advances in Neural Information Processing Systems (NeurIPS)*, 2025. arXiv:2512.03276.
* **Official URL:** https://arxiv.org/abs/2512.03276
* **Inspection Depth:**
  * Abstract Inspected: **Yes**
  * Full-Text Inspected: **Partial** (Preprint & OpenReview draft)
  * Methodology Verified: **Literature-reported** (Attribution patching, linear probing)
* **Abstract-Supported Claims:** VLMs show degraded factual recall compared to their LLM backbones because visual entity resolution occurs too late in the network, missing early-layer factual recall circuits in the LLM.
* **Full-Text-Supported Claims:** Patching entity representations directly into early LLM layers or prompting with Chain-of-Thought recovers degraded recall performance.
* **Methodological Overlap with CrossModalLab:** Directly utilizes activation patching and probing to diagnose multi-hop relational retrieval across visual and language layers; focuses on *where* in layer depth cross-modal binding succeeds or fails.
* **Differences & Open Boundaries:** Investigates parametric knowledge recall (entity $\to$ attribute) rather than compositional synthetic reasoning across independent spatial premises ($A \to B \to C$); does not investigate underdetermined forks or evidence insufficiency.

---

### Paper 2: When RAG Hurts: Diagnosing and Mitigating Attention Distraction in Retrieval-Augmented LVLMs
* **Bibliographic Metadata:** Beidi Zhao, Wenlong Deng, Xinting Liao, Yushu Li, Nazim Shaikh, Yao Nie, Xiaoxiao Li. *International Conference on Machine Learning (ICML)*, 2026. arXiv:2602.00344.
* **Official URL:** https://arxiv.org/abs/2602.00344
* **Inspection Depth:**
  * Abstract Inspected: **Yes**
  * Full-Text Inspected: **Partial** (Preprint)
  * Methodology Verified: **Literature-reported** (Attention weight inspection, MAD-RAG intervention)
* **Abstract-Supported Claims:** In retrieval-augmented LVLMs, relevant retrieved text can globally distract attention away from visual tokens, suppressing visual grounding and causing errors on questions the model answers correctly without retrieved text.
* **Full-Text-Supported Claims:** Attention Distraction (AD) can be mitigated training-free by decoupling visual grounding from context integration via dual-question attention mixing.
* **Methodological Overlap with CrossModalLab:** Provides concrete evidence of cross-modal competition where textual premises override visual representations in Transformer attention layers; directly informs hypothesis $H_{\text{gen}}$ and $H_{\text{text}}$.
* **Differences & Open Boundaries:** Focuses on external document retrieval in RAG settings rather than pure compositional relational reasoning; does not perform causal activation patching on hidden states or evaluate underdetermined logical forks.

---

### Paper 3: Mitigating Hallucinations in Large Vision-Language Models via Causal Route Gating
* **Bibliographic Metadata:** Zhe Cheng, Wenyu Chen, Fode Zhang, Dehuan Shen. *International Conference on Machine Learning (ICML)*, 2026 (Spotlight Paper).
* **Official URL:** https://proceedings.mlr.press/v235/cheng26a.html (ICML 2026 proceedings)
* **Inspection Depth:**
  * Abstract Inspected: **Yes**
  * Full-Text Inspected: **Partial** (Proceedings draft)
  * Methodology Verified: **Literature-reported** (Route decomposition, causal effect estimation)
* **Abstract-Supported Claims:** Hallucinations in LVLMs stem from "route competition" where language priors dominate visual evidence even when attention is allocated to visual tokens; proposes training-free Causal Route Gating.
* **Methodological Overlap with CrossModalLab:** Explores causal competition between linguistic prior pathways and visual grounding pathways, directly aligning with our investigation into why models hallucinate relations rather than admitting insufficiency.
* **Differences & Open Boundaries:** Evaluates general object hallucination in image captioning/VQA rather than formal logical compositionality (transitive chains vs. underdetermined forks).

---

### Paper 4: Back Attention: Understanding and Enhancing Multi-Hop Reasoning in Large Language Models
* **Bibliographic Metadata:** Zeping Yu, Yonatan Belinkov, Sophia Ananiadou. *Empirical Methods in Natural Language Processing (EMNLP)*, 2025. arXiv:2502.10835.
* **Official URL:** https://arxiv.org/abs/2502.10835
* **Inspection Depth:**
  * Abstract Inspected: **Yes**
  * Full-Text Inspected: **Partial** (EMNLP proceedings)
  * Methodology Verified: **Literature-reported** (Logit flow tracking, backward attention routing)
* **Abstract-Supported Claims:** Multi-hop reasoning failures in LLMs occur during relation extraction due to conflicting logits across layers; introduces "logit flow" and a "back attention" mechanism enabling lower layers to access higher-layer states.
* **Methodological Overlap with CrossModalLab:** Provides methodology for tracing information flow across multi-hop reasoning steps; demonstrates that intermediate relational tokens suffer from representation conflicts before reaching the final layer.
* **Differences & Open Boundaries:** Conducted purely in the text-only unimodal domain; does not investigate cross-modal binding with visual tokens or epistemic abstention under incomplete premise graphs.

---

### Paper 5: Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation
* **Bibliographic Metadata:** Qiyue Gao, Xinyu Pi, Kevin Liu, et al. *Findings of the Association for Computational Linguistics (ACL Findings)*, 2025. arXiv:2506.21876.
* **Official URL:** https://arxiv.org/abs/2506.21876
* **Inspection Depth:**
  * Abstract Inspected: **Yes**
  * Full-Text Inspected: **Partial** (Preprint)
  * Methodology Verified: **Literature-reported** (WM-ABench evaluation suite)
* **Abstract-Supported Claims:** Proposes WM-ABench to dissect VLM capabilities into perception vs. prediction (transitive inference, compositional simulation); finds VLMs possess solid low-level perception but severely fail at transitive/compositional prediction.
* **Methodological Overlap with CrossModalLab:** Validates the behavioral existence of the transitive inference failure in VLMs; provides benchmark structures for transitive spatial reasoning.
* **Differences & Open Boundaries:** A behavioral benchmark study; does not perform internal mechanistic interpretability (probing, activation patching, or circuit tracing) and does not formally evaluate underdetermined fork structures.

---

### Paper 6: Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas
* **Bibliographic Metadata:** Shiqi Chen, Tongyao Zhu, Ruochen Zhou, Jinghan Zhang, Siyang Gao, Juan Carlos Niebles, Mor Geva, Junxian He, Jiajun Wu, Manling Li. *International Conference on Machine Learning (ICML)*, 2025. arXiv:2503.01773.
* **Official URL:** https://arxiv.org/abs/2503.01773
* **Inspection Depth:**
  * Abstract Inspected: **Yes**
  * Full-Text Inspected: **Partial** (ICML proceedings)
  * Methodology Verified: **Literature-reported** (Attention heatmaps, AdaptVis decoding)
* **Abstract-Supported Claims:** Spatial reasoning errors in VLMs correlate with attention misallocation across visual tokens and distinct accuracy gaps between simple (left/right) vs. depth-based (front/behind) relations; introduces AdaptVis decoding intervention.
* **Methodological Overlap with CrossModalLab:** Confirms that directional spatial relations ($\text{left-of}$) are the cleanest testbed for spatial composition and that attention misallocation is a primary failure candidate.
* **Differences & Open Boundaries:** Relies primarily on attention visualization (which is correlational, not causal); evaluates single-image spatial queries rather than cross-modal multi-premise reasoning.

---

### Paper 7: CF-VLM: Counterfactual Vision-Language Fine-tuning
* **Bibliographic Metadata:** Jusheng Zhang, Kaitong Cai, Yijia Fan, Jian Wang, Keze Wang. *Advances in Neural Information Processing Systems (NeurIPS)*, 2025. arXiv:2506.17267.
* **Official URL:** https://arxiv.org/abs/2506.17267
* **Inspection Depth:**
  * Abstract Inspected: **Yes**
  * Full-Text Inspected: **Partial** (Preprint & NeurIPS poster)
  * Methodology Verified: **Literature-reported** (Counterfactual fine-tuning pairs)
* **Abstract-Supported Claims:** Introduces counterfactual image-text fine-tuning to improve fine-grained compositional reasoning and reduce hallucinations in VLMs.
* **Methodological Overlap with CrossModalLab:** Emphasizes that minimal pair counterfactual perturbations are essential to isolate whether a model uses true relational features versus spurious shortcuts.
* **Differences & Open Boundaries:** A training/fine-tuning methodology rather than a diagnostic mechanistic study; does not analyze why pretrained models fail to abstain on underdetermined queries.

---

### Paper 8: Benchmarking Deflection and Hallucination in Large Vision-Language Models
* **Bibliographic Metadata:** *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)*, `acl-long.1307`.
* **Official URL:** https://aclanthology.org/2026.acl-long.1307/
* **Inspection Depth:**
  * Abstract Inspected: **Yes**
  * Full-Text Inspected: **Full Text**
  * Methodology Verified: **Literature-reported** (VLM-DeflectionBench, 2,775 samples)
* **Abstract-Supported Claims:** Models frequently fail to deflect (abstain with "I cannot answer") when retrieved multimodal evidence is incomplete, noisy, or conflicting, instead hallucinating plausible answers.
* **Full-Text-Supported Claims:** Evaluated across 20 state-of-the-art open-source and proprietary LVLMs; reveals that even leading models have an overwhelming tendency to generate unsupported assertions rather than admit uncertainty.
* **Methodological Overlap with CrossModalLab:** Directly addresses evidence insufficiency and abstention; provides behavioral proof of overconfident generation on incomplete evidence.
* **Differences & Open Boundaries:** Evaluates open-domain retrieval-augmented VQA behaviorally; does not investigate internal Transformer representations, circuit mechanisms, or multi-hop relational composition.

---

### Paper 9: When Image and Text Disagree: Cross-Modal Evidence Conflict in Multimodal Retrieval-Augmented Generation
* **Bibliographic Metadata:** *ACL MAGMaR Workshop 2026*.
* **Official URL:** https://aclanthology.org/volumes/2026.magmar-main/ (GitHub: `jaspercatapang/cmc-bench`)
* **Inspection Depth:**
  * Abstract Inspected: **Yes**
  * Full-Text Inspected: **Full Text**
  * Methodology Verified: **Code and dataset inspected** (CMC-Bench, 3,768 instances)
* **Abstract-Supported Claims:** Evaluates how models handle contradicting evidence between text and images across ChartQA and MMMU; accuracy drops by 17–46% under conflict.
* **Full-Text-Supported Claims:** Models exhibit a rigid "modality lean" (predominantly favoring text over visual evidence) rather than arbitrating evidence based on context; conflating abstention with fabrication obscures model failure modes.
* **Methodological Overlap with CrossModalLab:** Directly investigates cross-modal evidence conflict ($R_{\text{vis}} \land \neg R_{\text{text}}$); demonstrates strong language prior override.
* **Differences & Open Boundaries:** Studies single-hop document/chart conflict in RAG; does not investigate multi-hop transitive composition ($R(A, B) \land R(B, C) \models R(A, C)$) or underdetermined forks.

---

### Paper 10: MoHoBench: Assessing Honesty of Multimodal Large Language Models via Unanswerable Visual Questions
* **Bibliographic Metadata:** Yanxu Zhu, Shitong Duan, Xiangxu Zhang, Jitao Sang, Peng Zhang, Tun Lu, Xiao Zhou, Jing Yao, Xiaoyuan Yi, Xing Xie. *AAAI Conference on Artificial Intelligence (AAAI-26)*, 2026.
* **Official URL:** https://ojs.aaai.org/index.php/AAAI/article/view/40159 (GitHub: `yanxuzhu/MoHoBench`)
* **Inspection Depth:**
  * Abstract Inspected: **Yes**
  * Full-Text Inspected: **Full Text**
  * Methodology Verified: **Code and dataset inspected** (12,000+ unanswerable questions)
* **Abstract-Supported Claims:** MLLMs heavily struggle with refusal when visual evidence is insufficient to answer a question, frequently fabricating ungrounded answers.
* **Full-Text-Supported Claims:** Evaluates 28 MLLMs; proves honesty failures are exacerbated by visual inputs; proposes supervised fine-tuning and DPO alignment for refusal.
* **Methodological Overlap with CrossModalLab:** Grounds our secondary research question on epistemic insufficiency and failure to abstain.
* **Differences & Open Boundaries:** Purely behavioral evaluation and alignment fine-tuning; does not perform internal mechanistic interpretability (probing hidden states or causal activation patching).

---

### Paper 11: Evaluating and Analyzing Relationship Hallucinations in Large Vision-Language Models
* **Bibliographic Metadata:** Wu et al. *International Conference on Machine Learning (ICML 2024)*, PMLR 235:53483-53503, 2024.
* **Official URL:** https://proceedings.mlr.press/v235/wu24l.html
* **Inspection Depth:**
  * Abstract Inspected: **Yes**
  * Full-Text Inspected: **Full Text**
  * Methodology Verified: **Literature-reported** (R-Bench, RA2M alignment)
* **Abstract-Supported Claims:** Identifies relationship hallucinations (spatial positioning and actions) as distinct from object hallucinations; models rely excessively on text co-occurrence priors.
* **Full-Text-Supported Claims:** Region-level visual-text alignment (RA2M) mitigates hallucinations by forcing attention toward relevant image regions.
* **Methodological Overlap with CrossModalLab:** Validates that spatial relations are vulnerable to language prior override.
* **Differences & Open Boundaries:** Evaluates single pairwise image relations ($R(A, B)$); does not investigate multi-premise cross-modal composition or underdetermined forks.

---

### Paper 12: HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models
* **Bibliographic Metadata:** Guan et al. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024)*.
* **Official URL:** https://openaccess.thecvf.com/content/CVPR2024/html/Guan_HallusionBench_An_Advanced_Diagnostic_Suite_for_Entangled_Language_Hallucination_and_CVPR_2024_paper.html (GitHub: `tianyi-lab/HallusionBench`)
* **Inspection Depth:**
  * Abstract Inspected: **Yes**
  * Full-Text Inspected: **Full Text**
  * Methodology Verified: **Code and benchmark inspected** (346 images, 1,129 questions)
* **Abstract-Supported Claims:** Uses question-pair and control-group structures to disentangle parametric language hallucinations from visual illusion errors.
* **Full-Text-Supported Claims:** Shows that even GPT-4V achieves only 31.4% question-pair consistency, with open-weight models scoring below 16%.
* **Methodological Overlap with CrossModalLab:** Direct methodological inspiration for our paired counterfactual quadruplet design.
* **Differences & Open Boundaries:** Diagnostic behavioral benchmark; does not perform internal circuit tracing or causal patching.

---

### Paper 13: Mechanisms of Prompt-Induced Hallucination in Vision-Language Models
* **Bibliographic Metadata:** William Rudman, Michal Golovanevsky, Dana Arad, Yonatan Belinkov, Carsten Eickhoff, Ritambhara Singh, Kyle Mahowald. *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)*, `acl-long.1941`.
* **Official URL:** https://aclanthology.org/2026.acl-long.1941/ (GitHub: `michalg04/prompt-induced_hallucinations`)
* **Inspection Depth:**
  * Abstract Inspected: **Yes**
  * Full-Text Inspected: **Full Text**
  * Methodology Verified: **Code inspected** (Ablation hooks and attention head tracking)
* **Abstract-Supported Claims:** VLMs hallucinate by favoring textual prompts over visual evidence; identifies a small subset of attention heads ("PIH-heads") that mediate copying from prompt.
* **Full-Text-Supported Claims:** Ablating PIH-heads reduces prompt-induced hallucinations by at least 40% training-free.
* **Methodological Overlap with CrossModalLab:** Proves attention head ablation works in VLMs; provides precedent for causal attention intervention.
* **Differences & Open Boundaries:** Evaluates object-counting where prompt directly contradicts the count in an image; does not evaluate multi-premise relational composition or underdetermined forks.

---

### Paper 14: Premise Blindness in Large Vision-Language Models: Understanding and Mitigating False-Premise Hallucination
* **Bibliographic Metadata:** Binhan Xu, Ligang Chang, Chenrui Yang, Yanhui Du, et al. *Information Sciences*, 2026.
* **Official URL:** https://www.sciencedirect.com/science/article/pii/S002002552600976X
* **Inspection Depth:**
  * Abstract Inspected: **Yes**
  * Full-Text Inspected: **Full Text**
  * Methodology Verified: **Literature-reported**
* **Abstract-Supported Claims:** Models suffer from "premise blindness", accepting false prompt premises directly contradicted by visual inputs due to language prior dominance.
* **Methodological Overlap with CrossModalLab:** Reinforces the risk of models accepting flawed textual premises over visual evidence.
* **Differences & Open Boundaries:** Focuses on conversational prompt attacks rather than deductive reasoning over incomplete premise graphs.

---

### Paper 15: Understanding Hallucinations in Large Visual and Language Models
* **Bibliographic Metadata:** *ACM Computing Surveys*, 2026.
* **Official URL:** https://doi.org/10.1145/3811409
* **Inspection Depth:**
  * Abstract Inspected: **Yes**
  * Full-Text Inspected: **Full Text**
  * Methodology Verified: **Literature survey**
* **Abstract-Supported Claims:** Establishes a comprehensive taxonomy dividing hallucinations into object, attribute, spatial, and cross-modal interleaving errors across the model lifecycle.
* **Methodological Overlap with CrossModalLab:** Foundational taxonomy for our diagnostic error categorization.

---

## 3. Foundational Prior Art

| Paper | Key Contribution | Relevance to CrossModalLab | Inspection Level | Verified URL |
| :--- | :--- | :--- | :--- | :--- |
| **Winoground**<br>(Thrush et al., CVPR 2022) | SOTA VLMs perform near chance on visio-linguistic compositionality when lexical overlap is 100%. | Establishes the baseline reality of compositional failure. | Full-Text Inspected | [arXiv:2204.03162](https://arxiv.org/abs/2204.03162) |
| **ARO Benchmark**<br>(Yuksekgonul et al., ICLR 2023) | Demonstrated VLMs act as "bags of words", failing on spatial relation order. | Guides our premise-order and entity-swapping negative controls. | Full-Text Inspected | [arXiv:2210.01936](https://arxiv.org/abs/2210.01936) |
| **MMVP**<br>(Tong et al., CVPR 2024) | Proved CLIP visual encoders suffer from severe visual blind spots. | Motivation for single-premise visual controls ($H_{\text{percep}}$). | Full-Text Inspected | [arXiv:2401.06209](https://arxiv.org/abs/2401.06209) |
| **Faith and Fate**<br>(Dziri et al., NeurIPS 2023) | Formulated Transformer multi-step reasoning as linearized subgraph matching. | Theoretical foundation for why multi-premise reasoning fails. | Full-Text Inspected | [arXiv:2305.18654](https://arxiv.org/abs/2305.18654) |
| **ROME / Causal Tracing**<br>(Meng et al., NeurIPS 2022) | Pioneered causal mediation analysis via activation patching in MLPs. | Methodological basis for our causal intervention protocol. | Full-Text Inspected | [arXiv:2202.05262](https://arxiv.org/abs/2202.05262) |
| **Causal Abstraction / DAS**<br>(Geiger et al., ICML 2022) | Formalized interchange interventions to test alignment with causal models. | Mathematical standard for proving internal relational representation. | Full-Text Inspected | [arXiv:2301.04709](https://arxiv.org/abs/2301.04709) |

---

## 4. Real-World Incident Evidence Audit Taxonomy

To ensure scientific rigor, we classify real-world evidence into five strictly separated evidentiary categories:

1. **Confirmed Deployment Incident:** A specific timestamped failure that occurred during live operational deployment, documented by system logs or official provider disclosures.
2. **Published Controlled Evaluation:** An empirical test conducted under controlled conditions on real-world or curated benchmark stimuli, published in peer-reviewed venues.
3. **General Safety Warning:** A policy disclaimer or terms-of-service restriction issued by a system provider warning users against specific high-risk tasks.
4. **Hypothetical Illustration:** An illustrative example constructed by researchers to illustrate a failure mode or hazard scenario.
5. **Unverified Claim:** An anecdotal or uncorroborated report lacking reproducible prompts, images, or verified system logs.

### Application of Evidence Categories to Audited Domains

| Case / Incident | Primary Source | Evidentiary Status | Verified Factual Evidence | Explicit Exclusions & Limits |
| :--- | :--- | :--- | :--- | :--- |
| **Be My AI / GPT-4V Navigation & Layouts** | OpenAI GPT-4V System Card (2023) & Be My Eyes ToS | **Published Controlled Evaluation** & **General Safety Warning** | Beta testing (16,000 users) established that GPT-4V makes spatial errors with confident tone; Be My Eyes explicitly disclaims navigation and medication use. | **Hypothetical scenario excluded:** Specific anecdotal stories (e.g. hitting a staircase) are hypothetical illustrations, not verified deployment logs. |
| **Financial Document & Receipt Fabrication** | Springer IJDAR (2026), DOI: 10.1007/s10032-026-00620-7 | **Published Controlled Evaluation** | Evaluated on CC-OCR and SROIE datasets; proved VLMs exhibit high critical-span hallucination rates on cropped/occluded receipts to make arithmetic balance. | **Deployment limitation:** Conducted as an academic evaluation on real document corpora, not a live production incident log. |
| **Medication Packaging Interpretation** | Clinical AI evaluations & Be My Eyes safety advisory | **Published Controlled Evaluation** & **General Safety Warning** | Studies show VLMs fail on curved/rotated prescription bottles; terms of service strictly prohibit medication dosage reading. | **Hypothetical scenario excluded:** Specific clinical dosage mix-ups are hypothetical hazard scenarios, not attributed legal incidents. |
| **Dillon v. City of Jacksonville Beach** | ACLU Lawsuit Filing (June 2026) | **Confirmed Legal Lawsuit (Negative Control)** | Lawsuit documents wrongful arrest of Robert Dillon resulting from false-positive biometric facial recognition match on grainy surveillance still. | **Boundary Demarcation:** Involves **biometric facial recognition (FRT)**, NOT generative vision-language Transformer reasoning. Retained strictly as a negative control. |

---

## 5. Synthesis & Candidate Research Opportunities

Existing literature has established:
1. VLMs suffer from compositional deficits and bag-of-words behavior (*Winoground*, *ARO*, *MMVP*).
2. Multi-hop factual recall breaks down when visual representations are resolved too late in the network (*Too Late to Recall*).
3. Text context can actively distract attention away from visual tokens (*When RAG Hurts*, *Premise Blindness*).
4. Models fail to deflect/refuse on incomplete or contradictory multimodal evidence (*VLM-DeflectionBench*, *MoHoBench*, *CMC-Bench*).
5. Attention head ablation can suppress prompt-copying in object counting (*PIH-Heads*).

> [!IMPORTANT]
> **Defensible Mechanistic Research Niche:**  
> Prior literature establishes that models fail behaviorally on unanswerable visual questions and cross-modal conflicts. The historical proposed gap was **mechanistically investigating multi-hop cross-modal relational composition ($R_{\text{vis}}(A, B) \land R_{\text{text}}(B, C) \models R(A, C)$) versus topological underdetermination ($R_{\text{vis}}(A, B) \land R_{\text{text}}(C, B) \not\models R(A, C)$) inside the Transformer residual stream**. An absence-of-prior-work claim is UNVERIFIED and is withdrawn; the Phase 3 collision audit identifies close representation/action, insufficiency and margin-training precedents.
> 
> CrossModalLab specifically investigates:
> - **RQ1 (Locus of Relational Composition):** Where in layer depth is the composed relation $R(A, C)$ formed or lost?
> - **RQ2 (Mechanistic Failure Attribution):** Does failure arise from perceptual errors, text comprehension, cross-modal binding collapse, or late-stage language prior override?
> - **RQ3 (Representation of Epistemic Insufficiency):** Does the internal residual stream encode the topological difference between complete chains and underdetermined forks? If so, why does the readout head collapse into forced directional guessing?

