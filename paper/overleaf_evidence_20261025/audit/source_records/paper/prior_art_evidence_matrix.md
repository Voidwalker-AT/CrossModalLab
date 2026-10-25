# Prior-art evidence matrix — October 25, 2026

This targeted source-grounded audit covers 21 papers. Full PDFs were obtained legally from public author/publisher endpoints and hashed privately; selected methods, results and appendices were inspected. It is not an exhaustive systematic review, a full-paper replication, or novelty clearance. PDF page numbers below are one-based file pages. Counts are reported by the source unless marked derived. UNVERIFIED means not established by this inspection, not absent from the paper. No external dataset or model was downloaded.

Public provenance: [paper hashes](../artifacts/publication/20261025/literature_sources.json), [code inventories](../artifacts/publication/20261025/official_code_sources.json), [evaluation-source hashes](../artifacts/publication/20261025/evaluation_code_sources.json), [additional source hashes](../artifacts/publication/20261025/additional_evaluation_sources.json), and [structured extraction](../artifacts/publication/20261025/prior_art_extraction.json). Repository HEADs are mutable: cached file SHA256/Git blob identities delimit this audit; no foreign code was executed.

CrossModalLab comparison uses the successful 12-forward/two-case pilot, six-family development generator and safely failed 108-forward screen (one attempt, zero scored rows). It does not pool those budgets or upgrade old development observations to confirmation.

## hallusionbench

[Full paper](https://arxiv.org/pdf/2310.14566); SHA256 `e36e0ac1e6ffb212f7d90399161f7d7538568f040167d38b83d838b8858463c7`.

| Audit field | Source-grounded finding / assessment |
|---|---|
| Publication identity | Guan et al., HallusionBench, CVPR 2024; arXiv:2310.14566 |
| Inspected locators | PDF pp.3–8, Fig.2, Tables1–3; selected main methods/results, not every supplementary example |
| Precise contribution | Paired visual/language controls distinguish diagnostic error categories. |
| Models / samples / experiments | 455 control pairs; 346 figures; 1,129 questions; 15 models (§5.1). |
| Independence and missing counts | UNVERIFIED independent-family count and total forward/experiment count; do not equate questions, conditions, seeds or model configurations with independent units. |
| Baselines and controls | No-image/original/edited images; random yes/no; manually edited counterfactuals. |
| Evaluation metrics | Question, figure and paired-question accuracy; consistency and bias. |
| Statistics and uncertainty | GPT-4 grading repeated three times (§4.1); no family CI verified. Human checking disclosed. |
| Evidence type | BEHAVIORAL; controlled input changes |
| Ablations | Image edits and model/input-condition comparisons. |
| Limitations / validity threats | Manual construction and perception errors limit attribution; repeated judge calls are not independent examples. |
| Supplement / official code / protocol | https://github.com/tianyi-lab/HallusionBench — evaluation.py and utils.py inspected selectively; judge/API dependence; no foreign execution. |
| Hardware and reproduction | CPU protocol/metric reimplementation is feasible. Original model/data execution has NOT been reproduced; uncached weights, data, APIs or training require separate approval. Published full scale is not a requirement for our narrower claim. |
| Strength relative to CrossModalLab | Much stronger varied behavioral coverage; input controls do not establish an internal circuit. |
| Claim-specific evidence needed | Human-checked perception and matched image/text interventions; report whole-family consistency. |
| Essential vs scale / venue conventions | Essential: validity controls, declared units, defensible comparison and uncertainty appropriate to the claim. Optional scale: their exact model/sample count. Venue-specific formatting is not a scientific validation criterion. |

## rbench

[Full paper](https://arxiv.org/pdf/2406.16449); SHA256 `2c368de9e3b98a7b67a55e544aee9d11583602d2a557c84b52d1198d375a5987`.

| Audit field | Source-grounded finding / assessment |
|---|---|
| Publication identity | Wu et al., Evaluating and Analyzing Relationship Hallucinations in Large Vision-Language Models, ICML 2024; arXiv:2406.16449 |
| Inspected locators | PDF pp.4–5,9, Tables2–3; supplementary tables indexed, not fully audited |
| Precise contribution | Image- and instance-level relation hallucination assessment. |
| Models / samples / experiments | 11,651 questions on 4,034 images: 7,883 image-level + 3,768 instance-level; four evaluated models. |
| Independence and missing counts | UNVERIFIED independent-family count and total forward/experiment count; do not equate questions, conditions, seeds or model configurations with independent units. |
| Baselines and controls | Balanced positive/negative subsets, boxes/masks and object-recognition comparison. |
| Evaluation metrics | Accuracy, precision, recall, F1, yes ratio. |
| Statistics and uncertainty | Five random balanced subsets averaged; this does not certify independent-family uncertainty. |
| Evidence type | BEHAVIORAL |
| Ablations | Image/instance settings; appendix color ablation located, numerical details UNVERIFIED. |
| Limitations / validity threats | Manual/noisy annotations, restricted cause analysis and closed-model budget. Parser may default nonnegative answers to yes. |
| Supplement / official code / protocol | https://github.com/mrwu-mac/R-Bench — eval.py inspected: missing predictions skipped, first-sentence negation parser; five stored balanced subsets averaged. |
| Hardware and reproduction | CPU protocol/metric reimplementation is feasible. Original model/data execution has NOT been reproduced; uncached weights, data, APIs or training require separate approval. Published full scale is not a requirement for our narrower claim. |
| Strength relative to CrossModalLab | Relational hallucination itself is not a new topic; our logical insufficiency differs from binary relation truth. |
| Claim-specific evidence needed | Strict invalid-output accounting, unimodal/perception controls and exact query identities. |
| Essential vs scale / venue conventions | Essential: validity controls, declared units, defensible comparison and uncertainty appropriate to the claim. Optional scale: their exact model/sample count. Venue-specific formatting is not a scientific validation criterion. |

## mohobench

[Full paper](https://arxiv.org/pdf/2507.21503v4); SHA256 `d38bc56af442048b87af3815ea0d46920e968d18df9704ed516b210a02bec98e`.

| Audit field | Source-grounded finding / assessment |
|---|---|
| Publication identity | Zhu et al., MoHoBench, AAAI 2026, 40(34):29205–29213; DOI 10.1609/aaai.v40i34.40159; arXiv:2507.21503v4 |
| Inspected locators | PDF pp.4–6,9,16–18, Tables1,3,11; publisher metadata independently opened |
| Precise contribution | Four unanswerable-visual-question types and honesty alignment. |
| Models / samples / experiments | 12,158 questions; 2,334 images (1,920 COCO + 414 HaloQuest); 28 model configurations. Alignment includes three listed sizes in Table3. |
| Independence and missing counts | UNVERIFIED independent-family count and total forward/experiment count; do not equate questions, conditions, seeds or model configurations with independent units. |
| Baselines and controls | Human QA filtering, judge validation; answerable/general-ability retention via MMMU; SFT/preference methods. |
| Evaluation metrics | Refusal rate, rationality/helpfulness judge scores and aggregate balance metric. |
| Statistics and uncertainty | Reported judge agreement 91.43%, inter-annotator 95.24% on 105 items; confidence intervals UNVERIFIED. |
| Evidence type | BEHAVIORAL |
| Ablations | Noise: 250 images/1,000 questions; SFT, DPO, SimPO, ORPO comparisons. |
| Limitations / validity threats | All-unanswerable main evaluation does not by itself penalize blanket refusal; taxonomy and training generality limited. |
| Supplement / official code / protocol | https://github.com/yanxuzhu/MoHoBench — README and response-evaluation prompts inspected. App.C2: 10k training pairs; 4×A100 80GB, approximately 3h/run; not locally reproducible at that scale. |
| Hardware and reproduction | CPU protocol/metric reimplementation is feasible. Original model/data execution has NOT been reproduced; uncached weights, data, APIs or training require separate approval. Published full scale is not a requirement for our narrower claim. |
| Strength relative to CrossModalLab | Direct overlap with multimodal abstention and alignment; not evidence our margin objective is novel. |
| Claim-specific evidence needed | Matched sufficient controls, separate over-refusal and unsupported answer rates, validated judge if used. |
| Essential vs scale / venue conventions | Essential: validity controls, declared units, defensible comparison and uncertainty appropriate to the claim. Optional scale: their exact model/sample count. Venue-specific formatting is not a scientific validation criterion. |

## cmc

[Full paper](https://aclanthology.org/2026.magmar-main.3.pdf); SHA256 `96fa6f254a4bd227dab19962227d284f2f3258a43174fd0b2efb93fca5d03470`.

| Audit field | Source-grounded finding / assessment |
|---|---|
| Publication identity | Catapang, When Image and Text Disagree: Cross-Modal Evidence Conflict in Multimodal Retrieval-Augmented Generation, MAGMaR 2026; DOI 10.18653/v1/2026.magmar-main.3 |
| Inspected locators | PDF pp.2,4–5,8; Tables1–3 and limitations |
| Precise contribution | Four-condition retrieval evidence conflict evaluation (not the unrelated compression CMC-Bench). |
| Models / samples / experiments | 942 base examples, 3,768 condition instances; ChartQA 500 + MMMU 442; four quantized model configurations. |
| Independence and missing counts | UNVERIFIED independent-family count and total forward/experiment count; do not equate questions, conditions, seeds or model configurations with independent units. |
| Baselines and controls | Four matched evidence conditions; 20% construction QC sample. |
| Evaluation metrics | Acc, modality following/bias, ConfabR, CDR and HR; both-wrong accuracy accepts NEITHER or ABSTAIN. |
| Statistics and uncertainty | UNVERIFIED confidence intervals and seed aggregation in the inspected sections |
| Evidence type | BEHAVIORAL |
| Ablations | Condition/model comparisons; trivial, image-only, text-only and prompt ablations explicitly absent. |
| Limitations / validity threats | Unvalidated judge and chart-heavy heuristic conflicts. HR merges fabrication/refusal; both-wrong accuracy can reward fabrication. |
| Supplement / official code / protocol | https://github.com/jaspercatapang/cmc-bench — README and metrics.py inspected, compute_accuracy_from_behavioral/compute_confab_and_cdr; MLX/M3 Pro 36GB setup is not our CUDA environment; licence TBA in README. |
| Hardware and reproduction | CPU protocol/metric reimplementation is feasible. Original model/data execution has NOT been reproduced; uncached weights, data, APIs or training require separate approval. Published full scale is not a requirement for our narrower claim. |
| Strength relative to CrossModalLab | A published protocol can still omit essential controls; do not copy its composite metric as logical-U correctness. |
| Claim-specific evidence needed | Keep abstention and fabrication separate; add trivial/unimodal baselines and human validity audit. |
| Essential vs scale / venue conventions | Essential: validity controls, declared units, defensible comparison and uncertainty appropriate to the claim. Optional scale: their exact model/sample count. Venue-specific formatting is not a scientific validation criterion. |

## deflection

[Full paper](https://aclanthology.org/2026.acl-long.1307.pdf); SHA256 `1963e93a890b69b441e5f162108a9698ee750182672fd07a98977d3d526b032b`.

| Audit field | Source-grounded finding / assessment |
|---|---|
| Publication identity | Moratelli et al., Benchmarking Deflection and Hallucination in Large Vision-Language Models, ACL 2026; DOI 10.18653/v1/2026.acl-long.1307 |
| Inspected locators | PDF pp.3,6–9,13,15; Tables2–4,7 and AppendixE |
| Precise contribution | Retrieval-dependent curation and parametric/oracle/realistic/adversarial evaluation. |
| Models / samples / experiments | 2,775 samples; 1,246 unique questions; 2,717 unique query images. Abstract says 20 models; Table4 has 19 named rows (14 open + five closed): discrepancy UNRESOLVED. |
| Independence and missing counts | UNVERIFIED independent-family count and total forward/experiment count; do not equate questions, conditions, seeds or model configurations with independent units. |
| Baselines and controls | Four retrieval scenarios; negative context and prompt controls; manual judge validation. |
| Evaluation metrics | Accuracy, deflection and hallucination rates. |
| Statistics and uncertainty | Human judge check: 100 responses/model for four models; Table7 average agreement 92.6%, kappa .91, as reported; no family CI verified. |
| Evidence type | BEHAVIORAL |
| Ablations | Prompt strictness, retrieval noise and context-modality comparisons. |
| Limitations / validity threats | Single pass/query; judge/source biases and restricted licensing/model scope. |
| Supplement / official code / protocol | Official released evaluation code/dataset availability UNVERIFIED; no guessed repository treated as official. |
| Hardware and reproduction | CPU protocol/metric reimplementation is feasible. Original model/data execution has NOT been reproduced; uncached weights, data, APIs or training require separate approval. Published full scale is not a requirement for our narrower claim. |
| Strength relative to CrossModalLab | Stronger retrieval generality, not a neural mechanism; count disagreement must remain visible. |
| Claim-specific evidence needed | Distinguish knowledge availability, evidence sufficiency and response correctness under matched perturbations. |
| Essential vs scale / venue conventions | Essential: validity controls, declared units, defensible comparison and uncertainty appropriate to the claim. Optional scale: their exact model/sample count. Venue-specific formatting is not a scientific validation criterion. |

## winoground

[Full paper](https://arxiv.org/pdf/2204.03162); SHA256 `ca74ebefb947d164b799b0b2db61baeb7e6055381e96cc163384afc0d1c82bbe`.

| Audit field | Source-grounded finding / assessment |
|---|---|
| Publication identity | Thrush et al., Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality, CVPR 2022; arXiv:2204.03162 |
| Inspected locators | PDF pp.4–5,13, Table1/AppendixA; official dataset card read |
| Precise contribution | Matched groups with swapped compositional assignments. |
| Models / samples / experiments | 400 groups; 800 images + 800 captions; 1,600 image-caption pairings. Table1: 20 model configurations, not 20 independent architectures. |
| Independence and missing counts | UNVERIFIED independent-family count and total forward/experiment count; do not equate questions, conditions, seeds or model configurations with independent units. |
| Baselines and controls | Humans; text/image/group metrics require multiple consistent choices; same-word captions. |
| Evaluation metrics | Text, image and all-four group accuracy. |
| Statistics and uncertainty | AppendixA p.13: split 400 into four equal groups, Student-t 95% intervals across group scores; at least ten annotators per pairing (p.5). |
| Evidence type | BEHAVIORAL |
| Ablations | Architectural/model/pretraining comparisons; detailed semantic tags not fully audited. |
| Limitations / validity threats | Small curated groups; image access restricted by Getty/research terms; group subtypes are sparse. |
| Supplement / official code / protocol | https://huggingface.co/datasets/facebook/winoground — official gated card, statistics/compute_statistics.py and model_scores described; no gated acceptance or image download. |
| Hardware and reproduction | CPU protocol/metric reimplementation is feasible. Original model/data execution has NOT been reproduced; uncached weights, data, APIs or training require separate approval. Published full scale is not a requirement for our narrower claim. |
| Strength relative to CrossModalLab | Correct unit is matched group, directly motivating whole-family reporting here. |
| Claim-specific evidence needed | Report all-variant success and family uncertainty, not pooled variant accuracy alone. |
| Essential vs scale / venue conventions | Essential: validity controls, declared units, defensible comparison and uncertainty appropriate to the claim. Optional scale: their exact model/sample count. Venue-specific formatting is not a scientific validation criterion. |

## aro

[Full paper](https://arxiv.org/pdf/2210.01936); SHA256 `5a71b1f5371617b20bca02b273cdcee26ed98443e7e8c7f33cf328dd9a4981a2`.

| Audit field | Source-grounded finding / assessment |
|---|---|
| Publication identity | Yuksekgonul et al., When and why vision-language models behave like bags-of-words, and what to do about it?, ICLR 2023; arXiv:2210.01936v3 |
| Inspected locators | PDF pp.3–4,7,18,20; Tables and matched-epoch AppendixG |
| Precise contribution | Attribution, relation and order tests plus hard-negative contrastive learning. |
| Models / samples / experiments | VG relation 23,937; attribution 28,748; four evaluated VLM families. Exact order-corpus counts UNVERIFIED in this extraction. |
| Independence and missing counts | UNVERIFIED independent-family count and total forward/experiment count; do not equate questions, conditions, seeds or model configurations with independent units. |
| Baselines and controls | Original CLIP and matched CLIP fine-tuning; hard negative image/text combinations. |
| Evaluation metrics | Retrieval accuracy by relation/attribute/order; downstream classification/retrieval retention. |
| Statistics and uncertainty | Five order-permutation seeds; three-seed retrieval mean/std in appendix; not proof of family-independent precision. |
| Evidence type | BEHAVIORAL |
| Ablations | NegCLIP loss/data controls; five-epoch matched comparison and three learning rates (p.20). |
| Limitations / validity threats | Different pretraining/batch budgets; text artifact vulnerabilities require independent audit. |
| Supplement / official code / protocol | https://github.com/mertyg/vision-language-models-are-bows — main_aro.py inspected; seed and per-dataset evaluation; p.20 single RTX2080Ti/batch1024 training, outside our 4GB reproduction budget. |
| Hardware and reproduction | CPU protocol/metric reimplementation is feasible. Original model/data execution has NOT been reproduced; uncached weights, data, APIs or training require separate approval. Published full scale is not a requirement for our narrower claim. |
| Strength relative to CrossModalLab | Relational contrastive learning already established; our paired CE is not a new consistency method. |
| Claim-specific evidence needed | Matched compute/training controls, strong blind baselines, retain general ability. |
| Essential vs scale / venue conventions | Essential: validity controls, declared units, defensible comparison and uncertainty appropriate to the claim. Optional scale: their exact model/sample count. Venue-specific formatting is not a scientific validation criterion. |

## whatsup

[Full paper](https://aclanthology.org/2023.emnlp-main.568.pdf); SHA256 `ac45c83976d316ca06e3df6386a74a2b1cbf1a1f470f2c8db70aaf71935a36b3`.

| Audit field | Source-grounded finding / assessment |
|---|---|
| Publication identity | Kamath et al., What's 'up' with vision-language models? Investigating their struggle with spatial reasoning, EMNLP 2023; DOI 10.18653/v1/2023.emnlp-main.568 |
| Inspected locators | PDF pp.3–4,8,10; Tables1–2 |
| Precise contribution | Controlled physical object arrangements and natural spatial transfer. |
| Models / samples / experiments | 820 controlled photos (408+412); four relations/set. 205 sets is DERIVED 820/4, not independently certified families. COCO-spatial 2,687 images; GQA-spatial 1,451; 18 configurations (12 zero-shot + six tuned). |
| Independence and missing counts | UNVERIFIED independent-family count and total forward/experiment count; do not equate questions, conditions, seeds or model configurations with independent units. |
| Baselines and controls | Text-only/random, pair/set consistency, three corpora and human ambiguity check. |
| Evaluation metrics | Image, opposite-pair and four-relation set accuracy. |
| Statistics and uncertainty | 100 human-reviewed items per corpus; reported human estimates 97.3/99/100%. Family intervals UNVERIFIED. |
| Evidence type | BEHAVIORAL |
| Ablations | Spatial tuning and marginalization; 300k downsampled/4M preposition training comparisons. |
| Limitations / validity threats | Small controlled domain; improvements inconsistent across corpora. |
| Supplement / official code / protocol | https://github.com/amitakamath/whatsup_vlms — marginalization.py pair/set grouping inspected; p.8 four RTX A6000 training, not local full reproduction. |
| Hardware and reproduction | CPU protocol/metric reimplementation is feasible. Original model/data execution has NOT been reproduced; uncached weights, data, APIs or training require separate approval. Published full scale is not a requirement for our narrower claim. |
| Strength relative to CrossModalLab | Real photographs and group controls support broader validity than our rendered evidence. |
| Claim-specific evidence needed | Human-checked natural examples and cross-domain tests without silently masking visible query evidence. |
| Essential vs scale / venue conventions | Essential: validity controls, declared units, defensible comparison and uncertainty appropriate to the claim. Optional scale: their exact model/sample count. Venue-specific formatting is not a scientific validation criterion. |

## rep2act

[Full paper](https://arxiv.org/pdf/2609.32653); SHA256 `6d56a714d110c23ee7969020aa3fd478597d8a1804d5952ca1720cae2b9cbe00`.

| Audit field | Source-grounded finding / assessment |
|---|---|
| Publication identity | He and Chen, From Knowing to Abstaining: Bridging the Representation–Action Gap in Vision–Language Models, arXiv:2609.32653v1; PREPRINT, peer review UNVERIFIED |
| Inspected locators | PDF pp.1,3,6–7,20–21; Tables3,11 and training appendix |
| Precise contribution | Shortcut-filtered answerability/rationales, probes and representation-to-action training. |
| Models / samples / experiments | VAD-R 1,852 = 1,226 train + 326 dev + 300 test; 11 base VLMs (eight open, three closed); two tuned Qwen sizes; six evaluation datasets. |
| Independence and missing counts | UNVERIFIED independent-family count and total forward/experiment count; do not equate questions, conditions, seeds or model configurations with independent units. |
| Baselines and controls | Confidence, uncertainty, contrastive and prompt baselines; domain transfer and dev-selected fusion. |
| Evaluation metrics | Action accuracy, abstention precision/recall/F1 and QA retention. |
| Statistics and uncertainty | Shortcut probes use three runs/grouped CV; exact inferential protocol for all reported improvements UNVERIFIED. |
| Evidence type | BEHAVIORAL + CORRELATIONAL probes; training changes behavior |
| Ablations | Route/margin/contrastive loss components and probe/route selection. |
| Limitations / validity threats | Probe decodability does not prove model usage; preprint conclusions require replication. |
| Supplement / official code / protocol | Official code availability UNVERIFIED; p.20 eight RTX A6000 48GB; train adds 3,406 external examples; full experiment not locally feasible. |
| Hardware and reproduction | CPU protocol/metric reimplementation is feasible. Original model/data execution has NOT been reproduced; uncached weights, data, APIs or training require separate approval. Published full scale is not a requirement for our narrower claim. |
| Strength relative to CrossModalLab | Direct collision with a representation/action-gap narrative and margin-based abstention method. |
| Claim-specific evidence needed | A distinct task-specific controlled finding, independent transfer and matched component ablations; no novel-gap claim. |
| Essential vs scale / venue conventions | Essential: validity controls, declared units, defensible comparison and uncertainty appropriate to the claim. Optional scale: their exact model/sample count. Venue-specific formatting is not a scientific validation criterion. |

## spatial_uncertain

[Full paper](https://arxiv.org/pdf/2605.30557v2); SHA256 `ad843f5ac9245ea5fa350a99e90577dd876430568072c2f4315a034f3f4531f9`.

| Audit field | Source-grounded finding / assessment |
|---|---|
| Publication identity | Zhang et al., Seeing Isn't Knowing: Do VLMs Know When Not to Answer Spatial Questions (and Why)?, arXiv:2605.30557v2; PREPRINT |
| Inspected locators | PDF pp.1,6–7,18; Tables1,4–5 and Appendix experiments |
| Precise contribution | Hold 3D scenes fixed while changing occlusion/perspective evidence. |
| Models / samples / experiments | 240 scenes/43 room types; 1,222 occlusion units and 5,928 questions; eight models (three open/five closed). |
| Independence and missing counts | UNVERIFIED independent-family count and total forward/experiment count; do not equate questions, conditions, seeds or model configurations with independent units. |
| Baselines and controls | Matched viewpoints, shuffled options, text-only and 600 open-ended queries; better-view vs depth interventions. |
| Evaluation metrics | Spatial correctness, abstention and viewpoint-information choices. |
| Statistics and uncertainty | UNVERIFIED confidence intervals and seed aggregation in the inspected sections |
| Evidence type | BEHAVIORAL input interventions + CORRELATIONAL representations |
| Ablations | Prompting/LoRA, depth/text/map/reference view; 209 wall-view interventions in appendix. |
| Limitations / validity threats | Decoding/attention findings are correlational; exact independent scene reuse structure UNVERIFIED. |
| Supplement / official code / protocol | https://spatialuncertain.github.io — linked by paper; resource availability not verified by accessible project response; no data/model downloads. |
| Hardware and reproduction | CPU protocol/metric reimplementation is feasible. Original model/data execution has NOT been reproduced; uncached weights, data, APIs or training require separate approval. Published full scale is not a requirement for our narrower claim. |
| Strength relative to CrossModalLab | A spatial-evidence-insufficiency benchmark is already proposed; our scope is narrower two-premise logic. |
| Claim-specific evidence needed | Demonstrate cross-modal logical dependency and distinguish input information interventions from neural causes. |
| Essential vs scale / venue conventions | Essential: validity controls, declared units, defensible comparison and uncertainty appropriate to the claim. Optional scale: their exact model/sample count. Venue-specific formatting is not a scientific validation criterion. |

## mmaqa

[Full paper](https://arxiv.org/pdf/2604.14799); SHA256 `8bcb4b7690bcf8c21c665f7cc646dc5ac3aeebab952fc602ab3108cfcc768b16`.

| Audit field | Source-grounded finding / assessment |
|---|---|
| Publication identity | Madhusudhan, Yadav and Lacoste, Knowing When Not to Answer: Evaluating Abstention in Multimodal Reasoning Systems, arXiv:2604.14799v1; PREPRINT UNDER REVIEW |
| Inspected locators | PDF pp.1,4–6,14,18; Tables and AppendixA/B |
| Precise contribution | Two-axis modality dependency/evidence sufficiency benchmark and single/multi-agent abstention. |
| Models / samples / experiments | 2,079 = 553 A-MMMU + 1,526 A-MMLBD; 1,137 answerable/942 unanswerable; three base VLMs, standalone + two MAS architectures. |
| Independence and missing counts | UNVERIFIED independent-family count and total forward/experiment count; do not equate questions, conditions, seeds or model configurations with independent units. |
| Baselines and controls | Always/never/random abstain; MaxProb, verbal confidence, self-consistency N=10, Ptrue, CoT; 22 transformations. |
| Evaluation metrics | Answerable/unanswerable accuracy, coverage, abstention rate, MCC; five-way error accounting. |
| Statistics and uncertainty | Three humans review 200 items/subset. PDF p.7 Table1 and p.9 Table3 define binomial SE sqrt(p(1-p)/n): AAC n=300/837, UAC n=253/689. These are item-based SEs, not family-cluster uncertainty. Best MCC is selected across condition/clause variants, not a prespecified held-out optimum. |
| Evidence type | BEHAVIORAL |
| Ablations | Prompt/abstention clauses, sequential/iterative orchestration, input transformations. |
| Limitations / validity threats | VLM QC overlaps evaluated families; oracle-chosen thresholds are diagnostic; huge output budgets; source-family dependence. |
| Supplement / official code / protocol | Official code availability UNVERIFIED; large closed/API and 32B model setup not available on current hardware. |
| Hardware and reproduction | CPU protocol/metric reimplementation is feasible. Original model/data execution has NOT been reproduced; uncached weights, data, APIs or training require separate approval. Published full scale is not a requirement for our narrower claim. |
| Strength relative to CrossModalLab | Direct conceptual collision with cross-modal evidence sufficiency; our exact logical certification may differentiate implementation, not establish novelty. |
| Claim-specific evidence needed | Strong threshold/selective baselines, frozen dev calibration and independent human adjudication. |
| Essential vs scale / venue conventions | Essential: validity controls, declared units, defensible comparison and uncertainty appropriate to the claim. Optional scale: their exact model/sample count. Venue-specific formatting is not a scientific validation criterion. |

## patching

[Full paper](https://arxiv.org/pdf/2309.16042); SHA256 `8ba9f928e321ad2f319bf0ac62ae582d8a70befbe88d0f85773302d64252d396`.

| Audit field | Source-grounded finding / assessment |
|---|---|
| Publication identity | Zhang and Nanda, Towards Best Practices of Activation Patching in Language Models: Metrics and Methods, ICLR 2024; arXiv:2309.16042v2 |
| Inspected locators | PDF pp.1,3,14; main experiment/appendix indexed |
| Precise contribution | Demonstrates methodological sensitivity of activation localization. |
| Models / samples / experiments | PairedFacts 145 matched prompt pairs; IOI 500 prompts; GPT-2 scales/GPT-J among settings; exact total configurations UNVERIFIED. |
| Independence and missing counts | UNVERIFIED independent-family count and total forward/experiment count; do not equate questions, conditions, seeds or model configurations with independent units. |
| Baselines and controls | Token alignment, corruption method and logit/probability/KL metric choices. |
| Evaluation metrics | Raw/normalized logit difference, probabilities and KL localization. |
| Statistics and uncertainty | UNVERIFIED confidence intervals and seed aggregation in the inspected sections |
| Evidence type | CAUSAL interventions within evaluated models/tasks; method sensitivity, not universal mechanism |
| Ablations | Gaussian noise vs symmetric replacement; patch regions/window choices. |
| Limitations / validity threats | Language tasks do not establish VLM validity; denominator instability and corruption artifacts. |
| Supplement / official code / protocol | Paper links TransformerLens/EasyTransformer; cached EasyTransformer example inspected only; exact paper-version code reproduction UNVERIFIED. |
| Hardware and reproduction | CPU protocol/metric reimplementation is feasible. Original model/data execution has NOT been reproduced; uncached weights, data, APIs or training require separate approval. Published full scale is not a requirement for our narrower claim. |
| Strength relative to CrossModalLab | Our CPU tensor replacement tests are engineering checks, far below live controlled intervention evidence. |
| Claim-specific evidence needed | Eligible opposed margins; no-op/unrelated-token/region/sham controls, bidirectional patches and family replication. |
| Essential vs scale / venue conventions | Essential: validity controls, declared units, defensible comparison and uncertainty appropriate to the claim. Optional scale: their exact model/sample count. Venue-specific formatting is not a scientific validation criterion. |

## ioi

[Full paper](https://arxiv.org/pdf/2211.00593); SHA256 `af0ea3ca3912f819d2cd06b0f36f5c20bfcab72fbc83816d830c286af6872c91`.

| Audit field | Source-grounded finding / assessment |
|---|---|
| Publication identity | Wang et al., Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small, ICLR 2023; arXiv:2211.00593 |
| Inspected locators | PDF pp.1,3,9–10; faithfulness/completeness/minimality tests |
| Precise contribution | Circuit-level explanation assessed against explicit counterfactual criteria. |
| Models / samples / experiments | One GPT-2 small; 15 templates; 100,000 generated baseline prompts; 26 heads/seven classes. |
| Independence and missing counts | UNVERIFIED independent-family count and total forward/experiment count; do not equate questions, conditions, seeds or model configurations with independent units. |
| Baselines and controls | Path patching, knockout, alternative/backup heads; 10 random/class/greedy knockout sets. |
| Evaluation metrics | Logit difference; faithfulness, completeness and minimality. |
| Statistics and uncertainty | Baseline 100k prompts does not substitute for intervention-family uncertainty; exact uncertainty procedure UNVERIFIED. |
| Evidence type | CAUSAL within defined IOI circuit/task, with documented incompleteness |
| Ablations | Circuit pruning, backup components and adversarial counterexamples. |
| Limitations / validity threats | Completeness fails under selected knockouts; template count differs from prompt count; no unique universal circuit proof. |
| Supplement / official code / protocol | https://github.com/redwoodresearch/Easy-Transformer — README and circuit_discovery.py inspected. Current example calls a separate automatic-discovery project and is not proof of the original paper pipeline. GPT-2 uncached; download requires approval. |
| Hardware and reproduction | CPU protocol/metric reimplementation is feasible. Original model/data execution has NOT been reproduced; uncached weights, data, APIs or training require separate approval. Published full scale is not a requirement for our narrower claim. |
| Strength relative to CrossModalLab | Much stronger causal evidence despite only one model; more models alone would not close our mechanism gap. |
| Claim-specific evidence needed | Live path controls and competing pathways before circuit wording; current causal claim remains absent. |
| Essential vs scale / venue conventions | Essential: validity controls, declared units, defensible comparison and uncertainty appropriate to the claim. Optional scale: their exact model/sample count. Venue-specific formatting is not a scientific validation criterion. |

## reliable_vqa

[Full paper](https://arxiv.org/pdf/2204.13631); SHA256 `f3f41ab2ed7cd3dfe26477f9145ae01d73ce3839dc91d084fdc0c4a57ce5a838`.

| Audit field | Source-grounded finding / assessment |
|---|---|
| Publication identity | Whitehead et al., Reliable Visual Question Answering: Abstain Rather Than Answer Incorrectly, ECCV 2022; arXiv:2204.13631v3 |
| Inspected locators | PDF pp.8–9,28; AppendixI and evaluation source |
| Precise contribution | Selective VQA with correctness selector and effective reliability. |
| Models / samples / experiments | Four VQA backbones; VQAv2 validation approximately 214k split 86k/22k/106k fit/val/test; exact counts UNVERIFIED here. |
| Independence and missing counts | UNVERIFIED independent-family count and total forward/experiment count; do not equate questions, conditions, seeds or model configurations with independent units. |
| Baselines and controls | MaxProb, calibration, learned selector and oracle; image-disjoint splits. |
| Evaluation metrics | Risk/coverage, AUC, coverage at risk and cost-based effective reliability. |
| Statistics and uncertainty | Ten seeds for Pythia/CLIP-ViL with mean/std; two other backbones use existing single checkpoints (§4/AppendixK). |
| Evidence type | BEHAVIORAL |
| Ablations | Selector features/calibration and validation-to-test threshold transfer. |
| Limitations / validity threats | Selector confidence targets answer correctness, not formal evidence insufficiency. Threshold generalization and dataset shift matter. |
| Supplement / official code / protocol | https://github.com/facebookresearch/reliable_vqa — reliable_vqa_eval.py inspected selectively: separate threshold/evaluation arrays; raw ranking/coverage and costs. No downloaded checkpoints. |
| Hardware and reproduction | CPU protocol/metric reimplementation is feasible. Original model/data execution has NOT been reproduced; uncached weights, data, APIs or training require separate approval. Published full scale is not a requirement for our narrower claim. |
| Strength relative to CrossModalLab | Our candidate argmax is not a calibrated selective predictor; use distinct definitions. |
| Claim-specific evidence needed | Frozen dev thresholds, achieved coverage/risk and utility on independent families. |
| Essential vs scale / venue conventions | Essential: validity controls, declared units, defensible comparison and uncertainty appropriate to the claim. Optional scale: their exact model/sample count. Venue-specific formatting is not a scientific validation criterion. |

## selectivenet

[Full paper](https://arxiv.org/pdf/1901.09192); SHA256 `70c911da4c620285d643f13332ba2dd5bf4743581a1f05f960ba4423cceec8d6`.

| Audit field | Source-grounded finding / assessment |
|---|---|
| Publication identity | Geifman and El-Yaniv, SelectiveNet: A Deep Neural Network with an Integrated Reject Option, ICML 2019, PMLR97:2151–2159; arXiv:1901.09192 |
| Inspected locators | PDF pp.4–6; Tables2–6; official PMLR metadata |
| Precise contribution | Joint prediction/rejection with target coverage and auxiliary head. |
| Models / samples / experiments | Three image datasets + one regression dataset; SVHN 73,257/26,032, CIFAR10 50k/10k, CatsDogs 20k/5k; Concrete 1,030. |
| Independence and missing counts | UNVERIFIED independent-family count and total forward/experiment count; do not equate questions, conditions, seeds or model configurations with independent units. |
| Baselines and controls | Softmax response and MC-dropout, VGG16/MLP adaptations; coverage-matched comparisons. |
| Evaluation metrics | Selective risk and actual coverage, regression MSE. |
| Statistics and uncertainty | Tables report ±; repeat count/error-bar interpretation UNVERIFIED in inspected pages. Coverage calibration uses a held-out-set Hoeffding argument. |
| Evidence type | BEHAVIORAL |
| Ablations | Coverage/representation/calibration experiments. |
| Limitations / validity threats | Target coverage needs checking; rejection does not define a logical third class. Dataset generality differs from a VLM claim. |
| Supplement / official code / protocol | https://github.com/geifmany/SelectiveNet — selectivnet_utils.py inspected: percentile threshold and covered-set risk; optional calibration uses model predictions; no execution. |
| Hardware and reproduction | CPU protocol/metric reimplementation is feasible. Original model/data execution has NOT been reproduced; uncached weights, data, APIs or training require separate approval. Published full scale is not a requirement for our narrower claim. |
| Strength relative to CrossModalLab | Abstention itself is not original; compare retained accuracy at matched coverage. |
| Claim-specific evidence needed | Separate class-U correctness from selective abstention, protect threshold selection. |
| Essential vs scale / venue conventions | Essential: validity controls, declared units, defensible comparison and uncertainty appropriate to the claim. Optional scale: their exact model/sample count. Venue-specific formatting is not a scientific validation criterion. |

## counterfactual_vqa

[Full paper](https://arxiv.org/pdf/2003.06576); SHA256 `2042d2a178bb6316d5941386c8ebda0eb16fc15106190d8b12b3c22abd3c3132`.

| Audit field | Source-grounded finding / assessment |
|---|---|
| Publication identity | Chen et al., Counterfactual Samples Synthesizing for Robust Visual Question Answering, CVPR 2020; arXiv:2003.06576 |
| Inspected locators | PDF pp.1,5,7; Tables1,3–4 and Algorithm references |
| Precise contribution | Mask critical image/question features and synthesize complementary supervision. |
| Models / samples / experiments | Four architectures in Table1: UpDn, PoE, RUBi, LMH; VQA-CPv2 and VQAv2 plus rephrasing tests. Exact evaluated counts UNVERIFIED. |
| Independence and missing counts | UNVERIFIED independent-family count and total forward/experiment count; do not equate questions, conditions, seeds or model configurations with independent units. |
| Baselines and controls | Reimplemented base models; V-CSS/Q-CSS separately and jointly; general-distribution retention. |
| Evaluation metrics | VQA accuracy, rephrasing consensus, attention-influence and confidence-improvement diagnostics. |
| Statistics and uncertainty | UNVERIFIED confidence intervals and seed aggregation in the inspected sections |
| Evidence type | BEHAVIORAL |
| Ablations | Mask counts, mixing weight and architecture/component studies. |
| Limitations / validity threats | Mask validity/grounding and general-distribution tradeoffs; attention diagnostics do not establish a unique neural cause. |
| Supplement / official code / protocol | https://github.com/yanxinzju/CSS-VQA — README, train.py function/branch inspection; training not executed. Original feature extraction/data missing locally. |
| Hardware and reproduction | CPU protocol/metric reimplementation is feasible. Original model/data execution has NOT been reproduced; uncached weights, data, APIs or training require separate approval. Published full scale is not a requirement for our narrower claim. |
| Strength relative to CrossModalLab | Counterfactual/evidence-aware training precedent; five steps of our CE+ES cannot establish superiority. |
| Claim-specific evidence needed | Matched CE vs ES vs paired-data-only, same schedule/seeds and untouched evaluation. |
| Essential vs scale / venue conventions | Essential: validity controls, declared units, defensible comparison and uncertainty appropriate to the claim. Optional scale: their exact model/sample count. Venue-specific formatting is not a scientific validation criterion. |

## rtuning

[Full paper](https://aclanthology.org/2024.naacl-long.394.pdf); SHA256 `80a77ff2da0e96e02e4f2ae860e45de16ef28fc3709db3d817c406b1bcff6119`.

| Audit field | Source-grounded finding / assessment |
|---|---|
| Publication identity | Zhang et al., R-Tuning: Instructing Large Language Models to Say I Don’t Know, NAACL 2024; DOI 10.18653/v1/2024.naacl-long.394 |
| Inspected locators | PDF pp.1,4–5,14; Table6 actual used splits |
| Precise contribution | Refusal-aware instruction data based on estimated knowledge boundary. |
| Models / samples / experiments | Three base models (3B/7B/13B), nine listed datasets; ParaRel 5,575 train/5,584 ID/13,974 OOD; MMLU 2,448/2,439/9,155. |
| Independence and missing counts | UNVERIFIED independent-family count and total forward/experiment count; do not equate questions, conditions, seeds or model configurations with independent units. |
| Baselines and controls | Pretrain-T, same willingly-answered Pretrain-W, vanilla instruction tuning; single/multitask transfer. |
| Evaluation metrics | Accuracy on willing answers, precision/recall and average precision. |
| Statistics and uncertainty | UNVERIFIED confidence intervals and seed aggregation in the inspected sections |
| Evidence type | BEHAVIORAL |
| Ablations | Single/multitask and refusal-form variants; exact global run count UNVERIFIED. |
| Limitations / validity threats | Different covered subsets can inflate naive accuracy comparisons; parametric ignorance differs from logical insufficiency. |
| Supplement / official code / protocol | https://github.com/shizhediao/R-Tuning — README inspected; p.5 A10040GB, one epoch/batch4; full training outside current scope. |
| Hardware and reproduction | CPU protocol/metric reimplementation is feasible. Original model/data execution has NOT been reproduced; uncached weights, data, APIs or training require separate approval. Published full scale is not a requirement for our narrower claim. |
| Strength relative to CrossModalLab | Cannot claim teaching refusal is new; need coverage-controlled external validation. |
| Claim-specific evidence needed | Same-data CE baseline and threshold controls; distinguish unsupported relation from unknown memorized fact. |
| Essential vs scale / venue conventions | Essential: validity controls, declared units, defensible comparison and uncertainty appropriate to the claim. Optional scale: their exact model/sample count. Venue-specific formatting is not a scientific validation criterion. |

## qlora

[Full paper](https://arxiv.org/pdf/2305.14314); SHA256 `33a4e757c19d6ea8d3cca8958fce8fe405e290272e135c99ee66065c775d5bd1`.

| Audit field | Source-grounded finding / assessment |
|---|---|
| Publication identity | Dettmers et al., QLoRA: Efficient Finetuning of Quantized LLMs, NeurIPS 2023; arXiv:2305.14314 |
| Inspected locators | PDF pp.1,4,7,10; Tables2–6 |
| Precise contribution | NF4, double quantization and paged optimizer training. |
| Models / samples / experiments | Over 1,000 tuned configurations reported (p.1), eight instruction datasets; T5/RoBERTa/LLaMA scales, not 1,000 independent architectures. |
| Independence and missing counts | UNVERIFIED independent-family count and total forward/experiment count; do not equate questions, conditions, seeds or model configurations with independent units. |
| Baselines and controls | BF16 full/LoRA, Int8, FP4, NF4 and double-quantization comparisons. |
| Evaluation metrics | GLUE accuracy, RougeL, MMLU, human/GPT-4 rankings and Elo. |
| Statistics and uncertainty | Table6 reported 95% intervals; judge/human agreement and order effects discussed; exact bootstrap scheme UNVERIFIED in inspected pages. |
| Evidence type | BEHAVIORAL |
| Ablations | Quantizer, adapter and dataset/scale comparisons. |
| Limitations / validity threats | Benchmark/judge validity and task dependence; 48GB flagship memory result cannot certify feasibility on 4GB. |
| Supplement / official code / protocol | https://github.com/artidoro/qlora — README inspected; no installation or training reproduction. |
| Hardware and reproduction | CPU protocol/metric reimplementation is feasible. Original model/data execution has NOT been reproduced; uncached weights, data, APIs or training require separate approval. Published full scale is not a requirement for our narrower claim. |
| Strength relative to CrossModalLab | QLoRA is an enabling existing method, not our contribution; local scorer correctness is distinct from training efficacy. |
| Claim-specific evidence needed | Verify exact adapter/frozen states and matched ablation; authorize training separately. |
| Essential vs scale / venue conventions | Essential: validity controls, declared units, defensible comparison and uncertainty appropriate to the claim. Optional scale: their exact model/sample count. Venue-specific formatting is not a scientific validation criterion. |

## sphere

[Full paper](https://aclanthology.org/2025.acl-long.568.pdf); SHA256 `e742e35ca9b1b358621ee3111d81648c1e95055f6198dca40dcfd566c962c4ce`.

| Audit field | Source-grounded finding / assessment |
|---|---|
| Publication identity | Zhang et al., SPHERE: Unveiling Spatial Blind Spots in Vision-Language Models Through Hierarchical Evaluation, ACL 2025; DOI 10.18653/v1/2025.acl-long.568 |
| Inspected locators | PDF pp.4,6,9; Tables2,5 |
| Precise contribution | Human-annotated progression from spatial skills to multi-skill/reasoning. |
| Models / samples / experiments | 2,285 question-answer pairs; Table5 has 20 configurations (16 general-purpose + four spatial variants), not 20 independent architectures. |
| Independence and missing counts | UNVERIFIED independent-family count and total forward/experiment count; do not equate questions, conditions, seeds or model configurations with independent units. |
| Baselines and controls | At least two authors annotate/cross-check; chance baseline; ground-truth intermediate-answer assistance. |
| Evaluation metrics | Validity and accuracy; invalid responses incorrect; question-option chance. |
| Statistics and uncertainty | Five generation seeds with shuffled options; no family CI established from those repetitions. |
| Evidence type | BEHAVIORAL |
| Ablations | Intermediate evidence and egocentric/allocentric comparisons. |
| Limitations / validity threats | Static images, limited manual scope; authors’ in-distribution rationale is not proof of contamination safety. |
| Supplement / official code / protocol | Official resource release availability UNVERIFIED; PDF reports RTX3090 execution; uncached models not loaded. |
| Hardware and reproduction | CPU protocol/metric reimplementation is feasible. Original model/data execution has NOT been reproduced; uncached weights, data, APIs or training require separate approval. Published full scale is not a requirement for our narrower claim. |
| Strength relative to CrossModalLab | Spatial reasoning hierarchy already covered; our exact premise logic needs a narrower positioning. |
| Claim-specific evidence needed | Separate perception/composition, make all evidence available to labeling and assess intermediate-answer control. |
| Essential vs scale / venue conventions | Essential: validity controls, declared units, defensible comparison and uncertainty appropriate to the claim. Optional scale: their exact model/sample count. Venue-specific formatting is not a scientific validation criterion. |

## sugarcrepe

[Full paper](https://arxiv.org/pdf/2306.14610); SHA256 `2522733f4e39d9978e62a9b310ef70857b35cbf59fcebec2fd6959ea3abb3a80`.

| Audit field | Source-grounded finding / assessment |
|---|---|
| Publication identity | Hsieh et al., SugarCrepe: Fixing Hackable Benchmarks for Vision-Language Compositionality, NeurIPS Datasets and Benchmarks 2023; arXiv:2306.14610 |
| Inspected locators | PDF pp.6,9,13; Tables2,6; AppendixA/B |
| Precise contribution | More fluent/plausible hard negatives and adversarial filtering. |
| Models / samples / experiments | 7,512 negatives (sum of seven Table2 cells); 17 CLIP configurations in Table6; human comparison. |
| Independence and missing counts | UNVERIFIED independent-family count and total forward/experiment count; do not equate questions, conditions, seeds or model configurations with independent units. |
| Baselines and controls | Text-only Vera/grammar, normalized ARO+CREPE on same positives, human quality checks. |
| Evaluation metrics | Two-choice retrieval recall/accuracy and text bias scores. |
| Statistics and uncertainty | UNVERIFIED confidence intervals and seed aggregation in the inspected sections |
| Evidence type | BEHAVIORAL |
| Ablations | Construction/filtering and compositional-learning method comparisons; whole appendix not audited. |
| Limitations / validity threats | Two identified biases do not exhaust artifacts; retrieval and CLIP scope. |
| Supplement / official code / protocol | https://github.com/RAIVNLab/sugar-crepe — text_model_eval.py inspected: pairwise text score ranking; models auto-load if executed, therefore not run. Published setup two 48GB A6000s/512GB host. |
| Hardware and reproduction | CPU protocol/metric reimplementation is feasible. Original model/data execution has NOT been reproduced; uncached weights, data, APIs or training require separate approval. Published full scale is not a requirement for our narrower claim. |
| Strength relative to CrossModalLab | Our oracle role audits need strong blind baselines beyond obvious templates; no “bias-free” claim. |
| Claim-specific evidence needed | Blind token/likelihood controls, human audit and generalization beyond tuning images. |
| Essential vs scale / venue conventions | Essential: validity controls, declared units, defensible comparison and uncertainty appropriate to the claim. Optional scale: their exact model/sample count. Venue-specific formatting is not a scientific validation criterion. |

## crepe_bias

[Full paper](https://arxiv.org/pdf/2506.08227); SHA256 `e2cd911a118e60cc8295010d7f22dded71e71f1e1e0b7a8aa9d549f4bd0a917a`.

| Audit field | Source-grounded finding / assessment |
|---|---|
| Publication identity | Udandarao et al., A Good CREPE needs more than just Sugar: Investigating Biases in Compositional Vision-Language Benchmarks, arXiv:2506.08227v1; PREPRINT status in this audit |
| Inspected locators | PDF pp.1–3; Tables1–5; remaining references indexed |
| Precise contribution | Shows distribution asymmetry can support blind attacks across composition tests. |
| Models / samples / experiments | 17 benchmark audit; benchmark-specific counts in tables; exact global model/run count UNVERIFIED. |
| Independence and missing counts | UNVERIFIED independent-family count and total forward/experiment count; do not equate questions, conditions, seeds or model configurations with independent units. |
| Baselines and controls | Token length; summed and normalized language-model likelihood; CLIP references and alternative blind models. |
| Evaluation metrics | Benchmark retrieval accuracy and bias score comparisons. |
| Statistics and uncertainty | UNVERIFIED confidence intervals and seed aggregation in the inspected sections |
| Evidence type | BEHAVIORAL |
| Ablations | Alternative blind models/score definitions and filtering checks. |
| Limitations / validity threats | Detecting shortcuts does not prove a tested VLM uses them; many tests share COCO images. |
| Supplement / official code / protocol | UNVERIFIED official evaluation-code availability |
| Hardware and reproduction | CPU protocol/metric reimplementation is feasible. Original model/data execution has NOT been reproduced; uncached weights, data, APIs or training require separate approval. Published full scale is not a requirement for our narrower claim. |
| Strength relative to CrossModalLab | Direct collision with broad benchmark/length-bias novelty; our exact candidate sensitivity is a narrow observation. |
| Claim-specific evidence needed | Audit joint feature leakage and domain overlap; test real model unimodal baselines before composition claims. |
| Essential vs scale / venue conventions | Essential: validity controls, declared units, defensible comparison and uncertainty appropriate to the claim. Optional scale: their exact model/sample count. Venue-specific formatting is not a scientific validation criterion. |

## Interpretation

Causal evidence can be strong in one model when interventions and counterfactual criteria are rigorous (IOI); twenty behavioral configurations do not create causal evidence. Conversely, CPU correctness of our replacement function supplies no live intervention evidence. Abstention, matched controls, role shortcuts, margin objectives and QLoRA each have clear precedents. The narrow candidate contribution is a transparent formal partial-premise diagnostic and measured compact-model decision behavior under controlled nuisance factors. Its originality and independent empirical value remain UNVERIFIED.

Closest-work breadth is still incomplete: reporting-bias studies and additional 2025–2026 unanswerability benchmarks cited by Rep2Act/MM-AQA are discovery leads, not newly inspected full-text evidence here. Unknown code availability, full run totals, family independence and unclear error bars remain explicit open audit fields.
