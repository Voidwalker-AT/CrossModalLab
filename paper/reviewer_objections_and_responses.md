# Skeptical reviewer simulation — October 25, 2026

These are three role-based critiques written by the same research assistant after the source audit, not independent human reviews or a prediction of acceptance. Responses distinguish completed work from promises. Sources and page/table locators are in the [evidence matrix](prior_art_evidence_matrix.md).

## Reviewer 1 — Multimodal reasoning and benchmark validity

**Strengths:** Explicit Yes/No/U semantics, dual-oracle labels, raw candidate scores and matched images. The audit openly identifies oracle role predictability rather than hiding it behind aggregate accuracy.

**Major weaknesses:** The nine thousand variants represent six families; most new benchmark rows have no model results. Formal modality necessity does not show the VLM used both modalities. The two-case pilot omits entailed examples. Images are rendered, human visual validity and natural transfer unestablished. HallusionBench, What'sUp, MM-AQA and SpatialUncertain already cover nearby controls and insufficiency.

**Unsupported claims:** A general compositional deficit, spontaneous abstention failure measured through three forced candidates, a new benchmark concept, or broad model superiority.

**Missing experiments:** Matched joint/image-only/text-only VLM controls; perception/answer-rule baselines; independent families; image/premise human review; all-variant consistency and natural transfer if claimed.

**Likely rejection reasons:** Weak construct validity, pseudoreplication, narrow evidence and uncertain novelty. Scaling row counts alone would not fix these issues.

**Minimum defensible changes:** Scope the title/abstract to controlled development diagnostics, distinguish formal information from VLM use, remove generalization claims, and collect independently reviewed/evaluated families before claiming a new empirical benchmark result.

**Current response:** The manuscript now labels development dependence, the new CPU validator refuses incomplete comparisons and the benchmark supplies measured metadata/role ceilings. The 108-forward screen failed after one attempt, so there is no new three-class model result. Strong unimodal VLM baselines and independent evaluation remain MISSING; no completed fix is claimed for them.

## Reviewer 2 — Mechanistic interpretability and causal inference

**Strengths:** Source identities, recorded neural provenance and explicit distinction between probes and model usage. Historical failed controls are retained. CPU no-op/anchor checks are useful engineering preparation.

**Major weaknesses:** Historical probe selection uses development feedback; Qwen failure examples have no linked Qwen neural measurements. Failed relation-specific controls weaken causal attribution. The newly evaluated pair does not satisfy opposed-margin eligibility. No fresh live intervention is authorized or executed.

**Unsupported claims:** A causal neuron, a sufficiency circuit, language-prior override as established cause, or a representation/action gap demonstrated solely by probe accuracy. IOI evaluates faithfulness, completeness, minimality and redundant paths; CrossModalLab currently does none of that at comparable empirical depth.

**Missing experiments:** Independent probe selection/evaluation; exact live anchor alignment; reversible patches with sham/unrelated controls; alternative circuit paths; family replication and sensitivity to patch metric/corruption choice.

**Likely rejection reasons:** A mechanistic title unsupported by causal results, correlation presented as explanation, cherry-picked intervention targets or unstable normalization.

**Minimum defensible changes:** Move representation analysis to explicitly exploratory/supplementary status; describe causal analysis as incomplete/failed specificity. A behavioral paper need not manufacture a mechanism section. A causal paper requires a separate valid experiment with appropriate controls.

**Current response:** No causal claim is upgraded. New CPU tests establish only replacement-function correctness; pilot No/U margins do not justify a patch run. No normalized indirect effect is calculated from a weak denominator. All missing live controls remain UNVERIFIED. Additional causal work needs explicit authorization.

## Reviewer 3 — Experimental methodology, statistics and reproducibility

**Strengths:** Frozen revisions, exact likelihoods/tokenization, process isolation, immutable attempt ledgers and preserved failures. Successful pilot parameters were compared exactly; a failed screen is not quietly retried or pooled as success.

**Major weaknesses:** A one-family pilot cannot support population uncertainty. Five training steps are not an efficacy study. Historical selection contaminates confirmation. The failed parent did not retain its triggering telemetry sample, so the exact safety resource and post-termination parameter invariance are UNVERIFIED. Partner-machine execution and rights checks remain pending.

**Unsupported claims:** Statistical superiority, calibrated class probabilities, universal training stability, confirmed resumed GPU trajectories, or complete reproduction on a second machine.

**Missing experiments/checks:** Family-level prespecified endpoints and paired intervals, retained coverage/over-refusal, matched training schedules/seeds if efficacy is claimed, complete source/environment manifests and independent-machine execution.

**Likely rejection reasons:** Small dependent evidence, post-hoc analysis and weak efficacy baselines; failure to expose resource/protocol departures would be an additional concern.

**Minimum defensible changes:** Freeze the independent protocol before evaluation, report raw denominators and all failures, separate exploratory and confirmatory results, verify ordinary-Python paths and share only lawful reproducible evidence. Use descriptive family CIs with clear limitations rather than treating repetition as independent evidence.

**Current response:** New complete-screen checks reject failure status, missing/duplicate cases, hash mismatches, nonfinite scores and incorrect argmax. The actual failed screen is rejected on CPU; no BA or CI is manufactured. Future gate errors now carry their sample and parent telemetry is written before checking, but this repair has only CPU verification. The original executed source is preserved. Partner confirmation remains NOT VERIFIED.

## Author action gate

All three perspectives converge on validity and independent evaluation before model scaling, training or neural storytelling. The manuscript can describe verified engineering and limited development findings now; the missing experiments are not promised as completed. The [gap report](publication_evidence_gap_report.md) prioritizes compute by claim value and states approval/resource needs. These simulated reviews should be replaced or supplemented by actual contributor/supervisor critique before submission.
