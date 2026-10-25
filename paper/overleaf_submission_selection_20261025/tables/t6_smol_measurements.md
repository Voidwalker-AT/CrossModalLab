# SmolVLM2 contributes behavioral and representation evidence, separately from Qwen fine-tuning.

Source: audit/dual_model_audit.json#smol; #neural

| Measurement | Observed | Evidence limit |
|---|---|---|
| 0D2 behavior | 18/36 correct; BA 0.425926 | 30 matched groups; different battery from Qwen |
| Class recall Yes / No / U | 10/12; 8/18; 0/6 | Not general model accuracy |
| Passive captures | 10 captures; 200 residual rows | 4 layers; 2048 coordinates; no fresh generation |
| Phase 1F probes | 81 train; 27 validation rows | 27 / 9 family split; 13 specs including PCA |
| L18 last-token BA | 0.777778; stored CI [0.592593, 0.962963] | B=2000 family resamples; selection; correlational |
