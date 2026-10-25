# CrossModalLab: final university paper selection

Prepared October 25, 2026 for the October 28 B.Tech demonstration and preliminary paper. This is a selected evidence package and manuscript plan, not the finished paper. No `main.tex` is supplied. The 8-10 page target is provisional; the university has not confirmed a template or page limit.

Start with `audit/final_manuscript_plan.md`, `audit/smolvlm2_evidence_report.md`, `audit/cross_model_comparability.md` and `audit/final_claim_audit.md`. The strongest contribution is a reproducible, symbolically checked three-state relational task and evidence-linked inspection system, illustrated by bounded development behavior and exploratory representation analysis. Neither a novel reasoning mechanism nor an effective new fine-tuning method is established.

The original `paper/overleaf_evidence_20261025/` directory, its ZIP, the October demonstration archives and all experiment outputs are preserved. The requested Smol report is placed here instead of inside the immutable original release. Earlier RAM-blocked status in that release remains historical; the subsequent nine-forward Qwen run is separately reconciled here. **This audit made zero new VLM forwards or optimizer steps.**

## Overleaf integration

Upload `figures/`, `tables/`, `equations/` and `references/` alongside the future manuscript. Keep `audit/` as reviewer/provenance material. A future IEEE-style document needs `graphicx`, `amsmath`, `amssymb`, `booktabs`, `tabularx`, `url` and optionally `hyperref`. Use `IEEEtran` bibliography style with `references/references_selected`; obtain the actual university template before final layout.

`figures/figure_includes.tex` supplies source-linked caption text and correctly chooses `figure` (3.5 inches) or `figure*` (7.16 inches). Select each environment once in the intended section; do not insert all figures together. Use the PNGs with ordinary pdfLaTeX; SVGs are editable alternatives and need conversion before direct pdfLaTeX inclusion. Seven figures are selected, with double-column diagrams and class recall kept full width for legibility. Three original stimuli are supplied separately, unchanged.

`tables/t1_*.tex` through `t6_*.tex` are compact main-paper tables. `s1_*.tex` and `s2_*.tex` are optional supplements, not part of the historical generation aggregate. Use the CSVs to inspect exact numeric source rows. The nine candidate-score values retain full precision in `audit/dual_model_audit.json`; displayed decimals are rounded. `equations/equations_verified.tex` contains verified task, scoring, loss and metric definitions. Integrate definitions where first needed; do not duplicate them across sections.

Citation usage is prescribed sentence by sentence in `references/citation_map.md`. Cite only the selected 14 references that support the final prose. Model cards are documentation; cited arXiv entries remain preprint-version citations where proceedings metadata is not normalized. Primary records were reopened; no new complete full-text literature audit or literature experiment was performed.

## Read-only verification

From the repository root, run `python scripts/verify_final_overleaf_selection.py paper/overleaf_submission_selection_20261025` using the existing environment (ULVO optional for contributors). It checks package syntax, numeric consistency, source hashes, references and selected file hashes without loading a model. ZIP CRC/extraction receipts and fresh test results accompany the package. The full manuscript and IEEE page layout remain uncompiled; structural snippet validation is not a compilation claim.

Full source validation requires the original local activation arrays and historical images; they are deliberately excluded from the distribution. Contributors without those assets can add `--packaged-only` to verify the self-contained selection, arithmetic, references and packaged hashes. That mode explicitly reports zero original source hashes checked. It cannot seal or create a verified archive. The selected-evidence unit tests use this portable mode; the primary-machine audit separately validates all 278 original source hashes. No second-machine execution is claimed.

Further GPU execution is **unnecessary for this university paper**. See `audit/further_gpu_recommendation.md`. Human tasks remain: agree author names/order, confirm university requirements, write the complete manuscript from this outline, compile it in Overleaf, review equations/figures at final print size and rehearse the presentation.
