# Claims registry

Allowed classifications: **ESTABLISHED BY EXPERIMENT**, **PRELIMINARY**, **CORRELATIONAL**, **CAUSALLY SUPPORTED**, **HYPOTHESIS**, **UNVERIFIED**. They describe evidence for a precisely scoped statement, not the maturity of the whole project.

| ID | Approved wording / evidence | Classification | Wording prohibited without new evidence |
|---|---|---|---|
| C1 | The observed five-step CE+ES run completed with finite gradients, changed LoRA tensors and invariant frozen tensors (E3/E9). | ESTABLISHED BY EXPERIMENT | Training is universally stable, or all 50 steps are feasible. |
| C2 | On 18 repeatedly inspected development cases, CE+ES step five had BA 0.500 versus CE 0.389, with a paired descriptive CI including zero (E4). | PRELIMINARY | ES improves generalization, reasoning, calibration, or robustness. |
| C3 | Corrected CE step 50 answered sufficient cases correctly in this slice but missed all six forks; its parser accepted malformed suffixes (E2). | PRELIMINARY | CE solves relational reasoning or cleanly follows output instructions. |
| C4 | Selected Smol representations supported an external linear probe; model-selection bias remains (E7). | CORRELATIONAL | The model causally uses this feature, or a neuron implements sufficiency. |
| C5 | Surviving intervention controls undermine a relation-specific causal interpretation (E8). | PRELIMINARY | A unique necessary/sufficient relational circuit has been discovered. |
| C6 | Language-prior override and composition failures are candidate explanations (E10). | HYPOTHESIS | The internal failure mechanism has been established. |
| C7 | Novelty of the exact research combination requires further collision review (W1–W13). | UNVERIFIED | First, novel, state of the art, or publication-ready contribution. |
| C8 | Generalization benefit of the frozen ES margin remains untested independently (E11). | UNVERIFIED | Five steps validate the proposed scientific method. |

No causal-mechanism claim is currently **CAUSALLY SUPPORTED**. Attention maps and logit-lens readouts are correlational. Probe failure would not prove information absence. Patching would not establish uniqueness without alternative-path and multi-component tests. Keep all future manuscript sentences tied to evidence IDs, revising these rows only after reading new raw evidence.
