# Phase 2E — Novelty Collision Audit

**Epistemic labels:** Entries marked **[Full-Text Inspected]** come from the repository literature review (Phase 0, audited). Entries marked **[Card/Abstract]** from official dataset/model cards fetched in this phase. **[UNVERIFIED]** = cited topic known but not re-inspected this session.

---

## 1. Collision matrix (proposed vs prior work)

| Proposed contribution | Closest prior work | Overlap | Differentiation (plausible gap) | Status |
|---|---|---|---|---|
| LoRA/QLoRA to reduce VLM hallucination | Broad PEFT-hallucination literature **[UNVERIFIED list]** e.g. instruction-tuning + LoRA on LVLMs | High on “LoRA improves VLM behavior” | Need **specifically chain-vs-fork abstention** + counterfactual consistency metrics we already measure | **Not novel alone** |
| Selective prediction / abstention for VLMs | Work on unanswerable VQA, selective classification **[UNVERIFIED]**; MoHoBench **[Card]**; R-Bench metadata | High on “models should abstain” | Our **formal underdetermined fork** (≠ false) + modality-ablation diagnostic | **Incremental gap** if formalized |
| Evidence-aware fine-tuning | VQA-VS / knowledge-based VQA missing-evidence **[UNVERIFIED]** | Medium–high | Cross-modal **two-premise** sufficiency with matched image/text ablations | **Plausible if measured** |
| Counterfactual consistency training | Counterfactual VQA / fairness consistency **[UNVERIFIED]** | Medium | Same-image inverted-query pairs + solver labels (exact pairs v1) | **Incremental** |
| Uncertainty-aware losses (margin, abstain CE) | SelectiveNet / abstention losses **[UNVERIFIED]**; standard CE baselines | High on loss form | Combined with **multimodal premise graph** supervision | **Combination ≠ novelty** |
| Mechanism-guided adapters | Probing + steering literature; our Phases 1–2 **[own work]** | Medium | Activation patching on evidence topology | **Open if causal controls hold** |
| Multimodal RAG / symbolic verification | Zhao “When RAG Hurts” **[Full-Text Inspected, novelty_matrix]**; symbolic VQA verifiers **[UNVERIFIED]** | Medium | **Internal** multi-premise binding vs retrieved text distraction | Distinct if no external retrieval used |
| HallusionBench-style consistency eval | Guan et al. CVPR 2024 **[Full-Text Inspected]** | High on eval protocol | Ours adds **training objectives** on top of consistency-style labels | Eval reused, training target different |
| Winoground/ARO compositionality | Thrush 2022 **[Card/Full-Text]**; Yuksekgonul 2023 **[Full-Text Inspected]** | High on 1-hop word order | Ours: **3-entity chains, epistemic labels, causal hooks** | Distinct task structure |

## 2. What is already established
- VLMs fail compositional spatial language (Winoground, ARO) **[Full-Text Inspected]**  
- Hallucination on missing evidence is common (HallusionBench, MoHoBench, our 2C/2D pilots) **[Full-Text + local]**  
- LoRA/QLoRA are standard parameter-efficient fine-tuning **[UNVERIFIED exhaustive survey]**  
- Selective prediction / abstention objectives are established ML tools **[UNVERIFIED exhaustive]**

## 3. Plausible research gaps (not yet established as ours)
1. **Formal three-way chain/contradiction/fork supervision** with symbolic oracle + **modality ablations** during PEFT.  
2. **Exact-image counterfactual consistency loss** tied to solver-derived labels (Phase 2B-R pairs).  
3. **Cross-model shared insufficiency failure** (2C-C) + format-sensitivity (2D) as a **diagnostic contract** for adapters.  
4. **Mechanism-guided LoRA** where adapter placement is chosen from (non-causal) probe maps — still needs causal proof.

## 4. Explicit non-claims
- No novelty claim until a **fresh exhaustive search** post-design approval.  
- Combining CE + abstention + counterfactual terms is **not** novel by itself.

**See also:** `docs/novelty_matrix.md`, `docs/literature_review.md` (Phase 0 full-text audits).
