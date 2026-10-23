# 1. Mechanistic Analysis of Cross-Modal Relational Reasoning and Evidence Insufficiency in Vision-Language Transformers

Working manuscript, 2026-10-23. Scientific status: preliminary developmental study. Author list, contributions, and venue: TODO, human agreement required. Evidence IDs refer to the [registry](../evidence_registry.md); W identifiers refer to the [primary-source audit](../references/verified_sources.md).

## 2. Abstract — preliminary

Vision-language systems can produce definite answers even when supplied premises do not determine the queried relation. We study this distinction using cross-modal strict-left-of tasks with entailed, contradicted, and underdetermined labels. Our workflow separates behavioral evaluation, external representation probes, and causal interventions, and preserves the limits of each form of evidence. In a repeatedly inspected development slice of 18 examples from six families, a corrected CE-only QLoRA control reached balanced accuracy 0.667 after 50 steps while failing all six underdetermined cases and producing malformed suffixes accepted by the historical parser. A newly verified five-step CE-plus-evidence-margin pilot reached development balanced accuracy 0.500 versus CE’s step-five 0.389. The paired descriptive family-bootstrap interval for their difference was [0, 0.222]; these results do not establish generalization or a beneficial new objective. Numerical audits verified finite adapter gradients, changed adapter weights, invariant frozen weights, and consistent saved states for this bounded run. Existing probes show correlational accessibility; incomplete intervention controls do not establish a relation-specific causal circuit. TODO: independent evaluation, preregistered statistical analysis, replication, and any causal evidence needed to support the title’s intended mechanistic scope.

## 3. Introduction

A grounded answer must depend on what the available evidence entails. Knowing that A is left of B and that C is left of B leaves the ordering of A and C undetermined. A system that answers “No” has confused insufficient evidence with contradiction. This error is especially relevant when premises are divided between an image and a textual statement: success can require both visual grounding and composition across modalities.

The project asks three separate questions. First, how do compact VLMs behave across logically matched evidence states? Second, what task information is externally accessible in their representations? Third, do controlled interventions support a specific internal computation? These questions cannot be collapsed into a single benchmark score. A probe may decode a label that the model does not use, while a generic perturbation may alter outputs without isolating relational reasoning.

Our current contribution is a reproducible exploratory workflow and bounded observations, not a completed mechanism discovery. Existing work already documents multimodal composition, spatial limitations, selective answering, and counterfactual training (W1–W5, W9–W13). The potential value of this study is a carefully controlled link between evidence-state behavior and internal computation; that link remains to be established. We therefore preserve unsuccessful controls and malformed outputs alongside successful engineering checks.

## 4. Problem formulation

Let R(x,y) mean that x is strictly left of y. On distinct objects with consistent spatial evidence, R is irreflexive, asymmetric, and transitive. Let E be the supplied premises and q=R(A,C). The label is Yes when E entails q; No when E entails not-q; and Undetermined when E entails neither q nor not-q. Inconsistent premise sets require a separate rejection policy and are outside this three-state formalization.

Forward chains R(A,B) and R(B,C) entail q. Reverse chains R(B,A) and R(C,B) entail R(C,A), hence not-q by asymmetry. Forks R(A,B) and R(C,B) admit models with either ordering of A and C. Underdetermined thus describes logical insufficiency of E, not the model’s confidence and not a false query. A model may also be uncertain about an entailed relation because it fails to perceive a premise; this is a different source of uncertainty.

The benchmark supplies one premise visually and another textually in controlled families. The intended test is whether the queried relation follows jointly. Single-modality controls and matched evidence changes are needed to distinguish joint composition from shortcuts. TODO: complete a row-level logical/render audit and document coverage of identity, overlap, contradiction, and perceptual ambiguity cases before publication.

## 5. Related work outline

Winoground controls lexical inventory to probe image-language composition (W1). What’sUp and SPHERE diagnose spatial understanding at different levels of complexity (W2/W12). Reporting-bias work provides a competing data-based explanation for weak reasoning (W13), which is not automatically the explanation for our checkpoints. POPE evaluates object hallucination (W3); a logically open relation is distinct from an absent object.

SelectiveNet and reliable VQA formalize rejection, risk, and coverage (W4/W5). R-Tuning teaches refusal around parametric knowledge boundaries (W10). Our proposed margin targets sufficiency of supplied premises, but neither a rejection objective nor uncertainty-aware tuning is a new idea on its own. Counterfactual VQA training provides another direct overlap in controlled evidence variation (W9). QLoRA supplies our memory-efficient adaptation method (W8).

Circuit studies and patching methodology motivate careful interventions and control choices (W6/W7). They do not license transferring a language-model circuit to our VLMs. TODO: full-text comparison, citation expansion, and a direct matrix comparing three-way epistemic tasks and full-sequence evidence margins. All summaries here are supported at abstract level; novelty remains UNVERIFIED.

## 6. Experimental methodology

Experiment identity includes model revision, quantization, source hashes, runtime versions, configuration and manifest hashes, ordered example IDs, source prompt and image hashes, parser version, optimizer settings, and validation membership. Runs use unique directories. Immutable checkpoints contain the adapter, AdamW state, Python/NumPy/CPU/CUDA RNG, and hashed metadata linked to the previous checkpoint. The journal is append-only; missing state never silently restarts training. Historical artifacts are retained.

The Phase 2I-B pilot uses the existing frozen 50-example schedule but executes only its first five entries. This avoids changing sample order when changing the stopping budget. Each of the five belongs to a different training family, and IDs/labels match CE’s first five. Evaluation uses six disjoint validation families with three primary variants each. Source-prompt hashes for the optional abstention instruction identify original records; the augmentation rule is recorded separately. No final-holdout model evaluation was performed in this mission. The complete manifest is read/hashed to select development records; final-holdout images are not used.

For development uncertainty we resample whole families 2,000 times with a fixed analysis seed and retain all three variants. Paired method comparisons use identical sampled families. These descriptive intervals do not correct repeated inspection, model/layer selection, or dependence induced by a narrow generator. TODO: preregister independent endpoints, sample size, effect size, and multiple-comparison policy before confirmatory evaluation.

## 7. Models and hardware

The pilot uses Qwen/Qwen3-VL-2B-Instruct at revision `89644892e4d85e24eaac8bacfd4f463576704203`. Historical probing and intervention artifacts use HuggingFaceTB/SmolVLM2-2.2B-Instruct at `482adb537c021c86670beed01cd58990d01e72e4`. Results are never pooled across revisions or prompts. Quantization is four-bit NF4 with double quantization and BF16 compute. Model files are reused offline; no new weights were downloaded.

The primary machine has an RTX 2050 Laptop GPU with 4,096 MiB dedicated VRAM and approximately 15.65 GiB physical RAM, running Windows 11. The pilot records Python 3.11.16, PyTorch 2.6.0+cu124, Transformers 5.20.1, PEFT 0.21.2, and bitsandbytes 0.50.2. Driver capability and bundled CUDA are separate. A CUDA matrix operation passed before loading. ULVO manages the primary runtime; scientific modules remain portable to ordinary Python.

## 8. Dataset construction

The current manifest is `datasets/cross_modal_contrast_v2/cross_modal_contrast_v2.jsonl` with SHA-256 `cda6379bc841a4dad4bf839d59c6ca4b61d7d90d014be20a599f64bf30bad1d6`. Selected prompts and images are hash-checked. Training/validation family overlap is rejected. The frozen training schedule maximizes family coverage with balanced class exposure, seed 20261111; its 50 entries span 50 families with 17 Yes, 17 No, and 16 Undetermined examples. The five-step prefix contains two Yes, two No, and one Undetermined examples.

Controlled rendered scientific data are distinct from illustrative synthetic demos. Demo records are quarantined and cannot enter empirical runs. A family may share logical or visual structure while using independent canvas jitter; “same visual structure” must not be reported as pixel-identical. TODO: generator specification, complete split counts, manual logical/render checks, licensed-asset provenance, and an independently frozen family set. Do not infer actual split membership from an `ho` substring in a family ID; manifest split fields govern selection.

## 9. Behavioral evaluation

We report balanced accuracy over all three labels, per-class recall, invalid-output rate, abstention frequency, inappropriate abstention on sufficient evidence, and definite responses on underdetermined evidence. Invalid/missing/conflicting outputs remain incorrect rather than being coerced to No or Undetermined. The historical parser `strict_answer_v1` recognizes label prefixes and detects multiple label words; despite its name it accepts some non-label suffixes. Exact-label output counts are therefore reported separately.

Greedy generation uses at most 16 new tokens. The standard and explicit-abstention prompts are separate conditions. Logical underdetermination is a ground-truth class, whereas selective prediction is an action policy; an Undetermined frequency is not by itself a calibrated risk-coverage curve. TODO: independently preregister output-format sensitivity and a calibrated selection protocol before claiming reliable abstention.

## 10. Mechanistic methodology

Representation probes quantify external accessibility of labels in a chosen hidden-state readout. The selected Phase 1F Smol layer-18 last-token probe reports BA 0.777778 on nine validation families, with stored family-bootstrap interval [0.592593,0.962963]. Layer/token/L2 selection used development outcomes, so this is correlational and potentially optimistic. It does not show that the decoder causally uses that representation (E7).

The surviving Phase 2B intervention file has 16 records over two families and only ten intervention rows. Its controls produce larger absolute margin changes than targeted patches in the completed subset. The stored recovery audit reports 26 missing planned records and a prior evidence-loss incident. These observations prevent a confirmatory relation-specific circuit claim (E8). No new causal patching was run here.

Any future causal study must distinguish mediation at the tested site from necessity, sufficiency, or uniqueness. Negative controls, valid corruption pairs, alternate paths, and held-out families are essential. Attention weights are routing measurements; logit-lens vocabulary readouts are correlational; residual dimensions are not automatically neurons or monosemantic features. TODO: separately authorized preregistered interventions or a narrower paper scope.

## 11. Parameter-efficient training

Language attention q/k/v/o projections receive rank-2 LoRA with alpha 4. Vision and multimodal projector parameters remain frozen. Batch size and gradient accumulation are both one. AdamW uses learning rate 0.0001, betas (0.9,0.999), epsilon 1e-8, weight decay 0.01, no scheduler; gradient checkpointing is enabled and generation cache disabled in training. CE supervises only the exact gold answer span: one token for Yes/No and two for Undetermined under this tokenizer.

For candidate a, s_a is the sum of teacher-forced log probabilities of its complete token sequence given the image/prompt and preceding answer tokens. Candidate strings include the leading space: ` Yes`, ` No`, ` Undetermined`. Scores are not length-normalized. Qwen multimodal token types align with appended text, and positions are recomputed to preserve multimodal rotary alignment. Multi-token answers are never reduced to their first token.

For insufficient evidence, ES=max(0,1+max(s_Yes,s_No)-s_Undetermined). For sufficient evidence, ES=max(0,1+s_Undetermined-s_gold). The frozen objective is CE+0.25 ES. This margin encourages the appropriate side of abstention while retaining CE supervision. Its effect is a hypothesis; different candidate lengths and CE averaging versus summed score margins remain confounds. No coefficient or margin was tuned during this pilot. Numerical audits compare all frozen tensors exactly, allowing only verified equivalent dtype changes, and separately require finite changed LoRA weights. Five optimizer calls were executed; further training requires explicit authorization.

## 12. Preliminary results

| Development condition | BA | Recall Yes / No / Undetermined | Invalid outputs | Exact-label outputs |
|---|---:|---|---:|---:|
| Frozen Qwen, standard prompt | 0.388889 | 1/6, 4/6, 2/6 | 0/18 | 18/18 |
| CE step five | 0.388889 | 3/6, 4/6, 0/6 | 6/18 | 0/18 |
| CE+ES step five | 0.500000 | 3/6, 4/6, 2/6 | 0/18 | 18/18 |
| CE step 50, historical | 0.666667 | 6/6, 6/6, 0/6 | 0/18 | 0/18 |

Each row uses the same 18 cases/six families. Source outputs and per-family predictions are in [derived tables](../tables/development_summary.json) and [CSV](../tables/family_predictions.csv). The step-five CE+ES minus CE BA difference is 0.111111, descriptive paired family CI [0,0.222222]. Only two additional cases are correct. Four of six forks remain wrong in the pilot. Six Undetermined predictions include four on sufficient cases, showing that correct abstention cannot be inferred from abstention frequency alone. With the explicit abstention prompt, frozen and step-five CE+ES BA are both 0.444444; a matched CE step-five augmented condition is unavailable.

The five CE+ES CE/ES/total loss triplets are (14.891876,0,14.891876), (16.306824,1.114895,16.585548), (7.565945,0,7.565945), (16.695772,3.702376,17.621367), and (17.152073,2.037066,17.661339). These are different examples, so the sequence is not a learning curve on a fixed input. Step runtimes sum to 22.2652 seconds, excluding baseline/evaluation, parameter audits, and checkpoint writes; the complete execution took 286.972 seconds. All five checkpoints plus initialization validate. No frozen numerical difference was found; all 224 adapter tensors changed relative to initialization. Peak step allocator usage was 3033.665 MiB, peak reserved 3304 MiB, maximum sampled NVIDIA usage 3795 MiB, and minimum sampled available RAM 2912.504 MiB. Sampled usage is not a continuous physical peak guarantee.

Earlier paired data report Smol 8/18 and Qwen 10/18 correct with zero correct forks each, on a different development slice (E6). These counts motivate follow-up but are not pooled with the present slice. The current evidence supports bounded feasibility and preliminary behavioral observations; lower loss or higher development BA does not establish better reasoning.

## 13. Limitations

There is no independent generalization result, multi-seed estimate, validated relation-specific causal mechanism, or exhaustive novelty clearance. Development sets have been reused for decisions. Six evaluation families yield fragile uncertainty estimates; the degenerate CE step-50 bootstrap reflects identical observed family scores, not population certainty. Rendered tasks cover a narrow domain. Candidate token lengths differ, generation can behave differently from teacher-forced scores, and the parser accepts malformed suffixes. Checkpoint validation does not itself demonstrate identical real-model resumed trajectories; CPU regression tests cover restoration, while no extra real-model resumed optimizer step was authorized. Windows sampled memory telemetry does not establish behavior on other hardware. Historical interrupted/deleted evidence cannot be reconstructed from surviving summaries.

## 14. Ethics and real-world relevance

Unsupported confident relations can be harmful in assistive descriptions, navigation, or other systems that act on spatial information. The controlled benchmark is not evidence of deployment readiness, safety, or performance on medical or navigation tasks. Logical abstention must not be confused with broad safety refusal. Document model/data licenses, computation limits, and the role of AI assistance; preserve raw evidence and failed attempts. TODO: author-reviewed license statement, data documentation, compute accounting, and publication-policy disclosures. No human-subject study or deployment evaluation is claimed.

## 15. Future experiments

First, have an independent contributor reproduce the saved-state and parser audit using ordinary Python and review the pilot’s exact source/configuration. Next, preregister a family-disjoint evaluation and matched training replication after explicit authorization, with output-format and token-length analyses planned before outcomes. Evaluate sufficient-case errors and unsupported definite answers jointly. Pursue causal interventions only after repairing the experimental design and receiving separate phase authorization. If causal evidence remains insufficient by the manuscript freeze, narrow the mechanistic claims with the authors. The [submission plan](../submission_plan.md) sets internal milestones without promising acceptance or inventing a venue deadline.

## 16. References

W1 Thrush et al., CVPR 2022; W2 Kamath et al., EMNLP 2023; W3 Li et al., EMNLP 2023; W4 Geifman and El-Yaniv, ICML 2019; W5 Whitehead et al., ECCV 2022; W6 Wang et al., ICLR 2023; W7 Zhang and Nanda, ICLR 2024; W8 Dettmers et al., NeurIPS 2023; W9 Chen et al., CVPR 2020; W10 Zhang et al., NAACL 2024; W11 Agrawal et al., versioned arXiv:1505.00468v7 / ICCV 2015; W12 Zhang et al., ACL 2025; W13 Kamath et al., TACL 2026. Full titles, complete author lists, official links, identifiers, overlap, and inspection confidence appear in the [verified-source audit](../references/verified_sources.md). TODO: normalize bibliography to the ultimately selected venue after full-text review.
