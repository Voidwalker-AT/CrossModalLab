# Final manuscript content plan

Provisional target: **8-10 IEEE-style pages including references**, subject to actual university instructions. This is an outline and evidence allocation, not `main.tex` or a finished submission. Suggested working title: *CrossModalLab: Evidence Sufficiency and Reproducible Inspection of Cross-Modal Relational Reasoning*. Confirm the registered university title with the authors.

## Narrative and evidence allocation

The paper asks whether compact VLMs distinguish an entailed relation, a contradicted relation and a relation that the supplied image/text cannot determine. It introduces the task, symbolic verification and inspection system; reports bounded behavior; describes Qwen adaptation feasibility and separate exploratory Smol representation results. It does not promise discovery of an internal causal failure mechanism. Logical worlds are an external ground-truth construction, never a claim that the VLM constructs the same graph.

| Section | Approximate pages | Content / exact evidence | Figures / tables | Citations |
|---|---:|---|---|---|
| Abstract | 0.2 | Problem, task, reproducible system, 18-case/six-family Qwen development setting, separate historical Smol evidence, feasibility and limits | No unsupported novelty or improvement claim | Normally no abstract citations |
| 1 Introduction | 0.7 | Missing evidence differs from falsehood; motivate a concrete fork and auditability. State contributions as engineering/task specification + bounded empirical observations | Fig. 02 evidence states | W1, W2, W5 |
| 2 Related Work | 0.7 | Composition/spatial reasoning; hallucination and abstention; recent insufficiency/representation-action studies; efficient adaptation; probe/patch limits | Avoid a broad literature leaderboard | W1, W2, W18, W19; W4, W5, W14, W15; W8, LoRA; ProbeControls, W7 |
| 3 Problem Formulation | 0.6 | Strict left-of; nonempty compatible total-order worlds; Yes/No/U truth conditions; chain/reverse/fork. Reject inconsistent premises | equations/equations_verified.tex: first two align blocks | Own formal definition; prior work does not prove our implementation |
| 4 CrossModalLab Architecture | 0.5 | Controlled input generation, dual symbolic oracle, model runners, stored results, separate representations, read-only five-view dashboard and reproducibility | Fig. 01 architecture | QwenCard, SmolCard for models only |
| 5 Dataset and Evaluation Protocol | 0.9 | Exposed development units; 18/6 Qwen; distinct 36/30-group Smol legacy battery; 108/36-family probe features; 9216 variants remain six families. Frozen identities, prompts, parser, greedy generation versus teacher-forced scoring, exact-format endpoint | T1 conditions, T2 units; three original stimuli | Use source hashes, not citations as substitutes for experiment provenance |
| 6 Experimental Results | 1.2 | Primary Qwen historical generations: frozen 7/18, prompt 8/18, CE5 7/18, CE+ES5 9/18, CE50 12/18, CE+ES5+prompt 8/18. CE50 U recall 0/6 and exact-label compliance 0/18. One displayed CE+ES fork error. Separate Smol paragraph: 18/36, recalls 10/12, 8/18, 0/6 | Fig. 03 failure, Fig. 04 checkpoint, Fig. 05 recall, Fig. 07 semantics/format; T3 behavior, T4 format. Avoid duplicating every table cell in prose | Empirical sources in audit/dual_model_audit.json |
| 7 Fine-Tuning Feasibility | 0.8 | Existing QLoRA adaptation, rank 2 / alpha 4 / 802816 trainable parameters. Token-averaged gold CE and complete-candidate ES hinge; margin 1, coefficient .25. Five-step CE+ES versus five-step CE is preliminary; CE50 is an unequal budget reference. Describe verification and paused state | T5 config; implemented loss equations, not five-point learning curve | LoRA, W8 |
| 8 Exploratory Neural Analysis | 0.7 | Smol only: four layers/2048 dimensions, sparse passive captures; external probes 81 train/27 validation rows, disjoint 27/9 families. Last-token layer18 BA .777778 with stored descriptive CI [.592593,.962963]. Development selection, metadata role discrepancy and unresampled intervals explicit | Fig. 10 Smol probes; T6 historical measurements | ProbeControls; W7 when distinguishing intervention claims |
| 9 Discussion | 0.4 | Task truth, observable output behavior and decoded representations are different evidence types. Explain why format and semantic correctness diverge and why probe accessibility does not establish model usage | No repeated results table | W5, W14, W15 as related context, not validation of our mechanism |
| 10 Limitations | 0.6 | Few exposed families, synthetic controlled setting, selection bias, different model batteries, parser dependence, unequal budgets, no calibrated uncertainty, incomplete causal controls, no demonstrated real-world/generalization benefit. University contribution is preliminary | Summarize failed 108 screen and historical causal limitations in prose | W7 / ProbeControls as methodological context |
| 11 Conclusion | 0.2 | Reproducible three-state task, observed developmental errors and feasible instrumentation/adaptation; carefully scoped next research | No additional claim | No new evidence |
| References | 1.0 | Fourteen selected sources; document preprints/cards accurately | references/references_selected.bib | Citation map required |

Estimated total: 8.5 pages before final float placement; seven figures and six compact tables may move the layout toward ten pages. This is a planning estimate, not a compiled page count. First shorten repetitive prose/auxiliary tables if the university later imposes a smaller limit; retain uncertainty and provenance statements. Optional supplements are outside the main narrative.

## Suggested abstract content, not a final manuscript

Explain that a missing relational premise can leave a question unresolved even when neither a positive nor a negative answer is justified. State that CrossModalLab combines controlled image/text relational tasks, two symbolic verification methods and an evidence-linked research dashboard. Identify the small, exposed development scope before quoting performance. Mention that Qwen checkpoint behavior separates semantic outcomes from formatting and that an evidence-sufficiency adaptation was executed for only five updates. Describe Smol as a separate historical behavioral/representation investigation. End with the explicit limit: the findings do not establish generalization, corrective-method superiority, calibrated confidence or a causal circuit. Do not call the contribution a novel QLoRA method.

## Main / supplementary / omit decisions

**MAIN PAPER:** three-state definition and oracle; system architecture; original fork failure; all selected Qwen generation conditions with clear step labels; exact-format distinction; historical adaptation configuration and five-step feasibility; brief Smol 36-case class counts and four-layer correlational probe figure; reproducibility and limitations. These give a coherent technical contribution without pretending all phases share a benchmark.

**SUPPLEMENTARY:** exact raw answers/hashes, per-layer/selector probe results and acquisition/evaluation split discrepancy; optional three-case Qwen candidate-score table S1; optional two-case matched pilot table S2; intervention recovery/control audit; full source/hash inventory. Do not merge either candidate table into the main generated-answer metrics. Historical optimizer/RNG receipts may be cited, not redistributed as binaries.

**OMIT FROM MAIN RESULTS:** earlier Smol-500M trials, tiny grounding/order sweeps as broad capability claims, Phase1D confounded probes, Phase1B test-labeled battery, earlier Phase2G control, redundant activation-norm plots, five-loss-point curves, 9216 variants as independent performance samples, failed screen as successful evaluation, causal heatmaps and any unsupported novelty/superiority/calibration claim. Preserve the artifacts and disclose material failure/provenance limits.

## Distinctions that must survive editing

1. External symbolic truth is not a reconstruction of the model's internal reasoning.
2. Historical execution receipts and current CPU arithmetic are not new GPU replication.
3. Candidate argmax and generated answer/parser endpoints must not be pooled.
4. Correlational probe results are not causal explanations; incomplete interventions do not identify a circuit.
5. Development-selected observations are not independent generalization.
6. Executing five adaptation updates establishes bounded feasibility, not superiority or convergence.

Reference novelty collision: W14/W15 already address VLM answerability/representation-action issues. The paper must not relabel evidence insufficiency, abstention or a representation-action gap as newly discovered here. The narrow defensible contribution is the concrete symbolically checked task/system and auditable combination of evidence streams on this hardware, without a priority claim over related work.

Before writing: authors verify names/order and university title/template. After writing: compile the actual manuscript, inspect floats at physical column widths, confirm captions/citations support the surrounding sentences and ensure every main-paper number can be traced to this package. No final source/PDF has been compiled in this selection stage.
