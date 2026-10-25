# Table ranking and selection

1. C behavioral results with raw counts: main paper, highest value. Include frozen, prompted, CE5, CE+ES5 and CE50; keep checkpoint step labels explicit.
2. D semantic correctness versus exact format: main paper or combine selected columns with C. Essential for interpreting CE50.
3. E configuration: compact main-paper summary; full optimizer/state details supplementary.
4. B dataset independence units: main paper if discussing the 9,216-row design; otherwise concise prose and supplement.
5. G historical Smol probes: optional main-paper row set only with explicit historical/model/selection/CI qualifications.
6. A model identities/conditions: compact methods or supplement; exact revisions must remain somewhere accessible.
7. H evidence limitations: supplement or a short limitations box.
8. F five-step telemetry: supplement. Do not interpret consecutive different-example losses as convergence.
9. I prior research: use a short selected comparison in the main paper; retain the full 21-paper extraction and unknown fields as supplementary.

All A–I have CSV, Markdown and LaTeX. The source-rich CSVs are canonical for exact values and units. The LaTeX versions are long-form record sheets for supplementary use, not nine mandatory main-paper tables. Use amsmath/amssymb/graphicx/longtable/url; select/reformat main-paper columns with care rather than scaling tiny text. Do not use this evidence library to imply more models, samples or experiments than were actually run.
