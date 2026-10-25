# Project overview for a Deep Learning evaluator

A visual assistant may see a picture, read a statement and confidently answer a question those inputs cannot settle. CrossModalLab makes that distinction visible: supplied evidence, correct logic and actual model answer are separate.

A **vision-language model (VLM)** processes images and language. **Cross-modal reasoning** combines information from both. Recognizing objects and understanding individual sentences does not guarantee correct composition of their relationships.

## Yes, No and Undetermined

We ask whether A is strictly left of C.

| Available facts | Consequence | Answer |
|---|---|---|
| A left of B; B left of C | A must be left of C | Yes: entailed |
| B left of A; C left of B | C must be left of A | No: contradicted |
| A left of B; C left of B | Either order of A/C is possible | Undetermined: insufficient evidence |

**Undetermined is not false.** It describes missing evidence. A model misreading a sufficient input is a different source of uncertainty.

## What is implemented

Controlled rendered benchmarks, a symbolic solver checked by two algorithms, stored VLM evaluations, checkpoint comparisons, historical instrumentation/probes and a working five-view dashboard. Hashes and original outputs make results traceable. Training utilities preserve state identities. The demonstration uses recorded evidence, not live inference.

The research history evaluates **SmolVLM2-2.2B-Instruct** and **Qwen3-VL-2B-Instruct**, using pinned revisions and four-bit NF4. The dashboard's principal 18-case checkpoint comparison is **Qwen**. Historical neural measurements shown separately are **Smol**, not Qwen. Exact identities are in [the results summary](RESULTS_SUMMARY.md).

## Five views

- **Failure Investigator:** image, text, ground truth, raw answer and checkpoint comparison.
- **Evidence & Reasoning:** premise graph and compatible orderings; symbolic ground truth, not model thoughts.
- **Architecture Explorer:** conceptual Transformer pipeline and separately labeled historical neural diagnostics.
- **Experiment Analytics:** class recall, confusion matrices, format compliance and recorded five-update telemetry.
- **Reproducibility:** sources, identities, hashes and scope, including separately labeled later research updates.

The working project is the primary university deliverable; the paper is preliminary. No discovered causal circuit, independent generalization, real-world transfer or CE+ES superiority is established. See [limitations](LIMITATIONS.md). The value of this implementation is that an examiner can inspect actual evidence and understand both the observations and their limits.
