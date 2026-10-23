# Delivered figures and tables

Five figures have vector SVG and 180-dpi PNG companions. The [manifest](submission_figure_manifest.json) records raw source and output SHA-256 values. No neural traces or activation values were fabricated.

| Figure | Delivered content | Evidence and interpretation |
|---|---|---|
| [1: evidence states](01_evidence_states.svg) | Forward chain, reverse chain and fork | Logical schematic; strict-left-of premises, not neural measurements |
| [2: pipeline](02_research_pipeline.svg) | Separate model outputs and symbolic oracle feed the audit | Conceptual workflow; ground truth is not supplied to the model as an answer |
| [3: checkpoints](03_checkpoint_comparison.svg) | Frozen, CE-5, CE+ES-5, CE-50; semantic BA and exact formatting | E1-E4; same 18 cases/six development families; unequal budgets; historical parser caveat; descriptive intervals in the table JSON |
| [4: per-class performance](04_class_recall.svg) | Yes/No/Undetermined recall across four checkpoints | Six cases per class; development reuse and one seed; no superiority inference |
| [5: failure example](05_recorded_failure.svg) | Original fork image, premises and exact frozen/CE+ES outputs | xc_ho_0011::underdetermined; both raw answers are No; countermodels are logical possibilities, not observations |

Rebuild with Python using the existing matplotlib installation and the source [builder](build_submission_figures.py). It loads fixed dashboard development artifacts, never the full dataset manifest, a model or a final-holdout image. No plotting library is needed at dashboard runtime.

Tables: [six dashboard conditions](../tables/submission_metrics.csv); [raw per-family predictions](../tables/family_predictions.csv); [development intervals and paired comparison](../tables/development_summary.json). The manuscript contains the model/hardware description, four-checkpoint performance table and pilot loss/gradient/resource accounting. No absent CE+ES-50 result is zero-filled. Earlier [conceptual notes](reasoning_and_pipeline.md) remain preserved.
