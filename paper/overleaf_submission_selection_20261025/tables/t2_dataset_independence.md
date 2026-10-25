# Sample counts do not establish statistical independence.

Source: audit/dual_model_audit.json; original tables/b_dataset_units.csv

| Evidence set | Records | Grouping | Interpretation |
|---|---|---|---|
| Qwen generation | 18 | 6 families | Repeated development; six conditions |
| Smol 0D2 | 36 | 30 matched groups | Includes six pairs and entity reuse |
| Smol Phase 1F | 108 | 27 train / 9 validation families | 81 / 27 rows; development-selected probes |
| Factorial variants | 9216 | 6 families; 72 images | Design diagnostics, no new VLM evaluation |
| Qwen new scores | 3 | 1 family | 9 forwards; candidate scoring only |
