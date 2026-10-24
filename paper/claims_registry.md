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

**C9 / E12 — CORRELATIONAL:** Existing Smol residual-output rows and probe results can be visualized with their recorded identities and CPU validation. Neither activation-norm differences nor probe accessibility identifies the cause of a selected Qwen error. Qwen per-case neural measurements remain NOT MEASURED. No existing claim is upgraded by the export.

## Phase 3 additions — 2026-10-24

| ID | Approved wording / evidence | Classification | Prohibited interpretation |
|---|---|---|---|
| C10 / E13 | On 68 exact selected records, correct logical labels coexist with deterministic single-modality role strata; a scheduled-train-fitted lookup reaches BA 2/3 on 18 validation cases. | ESTABLISHED BY EXPERIMENT, dataset diagnostic | Either VLM used this shortcut; the whole dataset or final holdout was inspected; BA alone proves joint composition. |
| C11 / E13 | New counterbalanced development records remove deterministic role strata; partial single-modality predictability remains and all new labels pass the dual oracle. | ESTABLISHED BY EXPERIMENT, generated design property | Chance-level unimodal prediction, independent confirmation or generalization to all images. |
| C12 / E14 | Frozen Qwen's summed-LL answer is No/Yes on the new matched No/U pair; LL/token selects U/U. The cross-model run stopped before Smol loaded. | PRELIMINARY | A completed cross-model replication, generated answers, calibrated confidence, superior normalization, a causal finding or an ES benefit. |
| C13 / E15 | Historical paired models both miss six forks; their descriptive paired accuracy interval includes zero. | PRELIMINARY | Independent new replication or statistically established model superiority. |
| C14 / E16 | Six saved boundaries validate on CPU and reload optimizer moments exactly with zero training steps and unchanged historical files. | ESTABLISHED BY EXPERIMENT, engineering | Fresh real-model/CUDA restoration, current full mixed-dataset verification or equality of resumed trajectories. |
| C15 / W14–W17 | Targeted primary-source review identifies close overlap in representation/action abstention, spatial insufficiency and margin-based training. | Literature-supported collision assessment; novelty UNVERIFIED | First/novel mechanism, unique sufficiency loss or exhaustive prior-art clearance. |
