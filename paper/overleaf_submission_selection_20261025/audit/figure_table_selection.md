# Final asset selection

Seven main-paper figures; six compact main tables. Two optional supplementary tables. Do not shrink double-column assets into a single column.

- Figure 1: `figures/02_three_evidence_states.png` (SVG also supplied); figure*, width 7.16 inches; sources and captions in `figures/selection_manifest.json`.
- Figure 2: `figures/01_system_architecture.png` (SVG also supplied); figure*, width 7.16 inches; sources and captions in `figures/selection_manifest.json`.
- Figure 3: `figures/03_recorded_failure_examples.png` (SVG also supplied); figure*, width 7.16 inches; sources and captions in `figures/selection_manifest.json`.
- Figure 4: `figures/04_checkpoint_comparison.png` (SVG also supplied); figure, width 3.5 inches; sources and captions in `figures/selection_manifest.json`.
- Figure 5: `figures/05_class_recall.png` (SVG also supplied); figure*, width 7.16 inches; sources and captions in `figures/selection_manifest.json`.
- Figure 6: `figures/07_format_vs_correctness.png` (SVG also supplied); figure, width 3.5 inches; sources and captions in `figures/selection_manifest.json`.
- Figure 7: `figures/10_historical_smol_probes.png` (SVG also supplied); figure, width 3.5 inches; sources and captions in `figures/selection_manifest.json`.

Exact table filenames: `tables/t1_evaluation_conditions.tex`, `t2_dataset_independence.tex`, `t3_qwen_behavior.tex`, `t4_format_compliance.tex`, `t5_training_configuration.tex`, `t6_smol_measurements.tex`; optional `s1_qwen_candidate_scores.tex` and `s2_matched_pilot.tex`. CSV and Markdown counterparts preserve numeric values.

Omit activation-norm plot (no distinct mechanistic result), GPU plot (engineering detail), five-step loss plot (not a learning curve), redundant confusion panels and shortcut-ceiling plot (appendix discussion suffices). Keep every historical original intact. The probe figure keeps stored intervals; endpoints are not freshly re-estimated or corrected for selecting the best layer.
