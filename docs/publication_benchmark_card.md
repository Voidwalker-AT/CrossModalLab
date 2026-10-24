# Controlled relational factorial development benchmark v1

Date: October 25, 2026. Evidence: `artifacts/publication/20261025/benchmark/{manifest.json,records.jsonl,shortcut_audit.json}`. Generated geometry is empirical benchmark material, not a synthetic tutorial demonstration. Records explicitly use `is_synthetic_demo: false`; no photographic realism or deployment validity is claimed.

## Construction and split

Seed 2026102502, six distinct unordered entity triples. Each family exhausts six role permutations, both AB/BC premise directions, both query directions, both modality assignments, two visual layouts, question-first versus premise-first text, two answer orders and image-first versus text-first presentation. There are 1,536 variants/family and 9,216 total: Yes 2,304, No 2,304, Undetermined 4,608. Ninety-six image files are shared by matched variants. Literal 2D names reuse the existing renderer mapping; no neural or realistic image measurements are synthesized.

Every row passes reachability and linear-extension verification. Query reversal exchanges Yes/No and preserves Undetermined while retaining image bytes. Modality order is an explicit message-content factor and must be respected by any future evaluation harness; it is not claimed evaluated by the fixed two-case pilot.

First three complete families fit baselines, remaining three evaluate them. All partitions are DEVELOPMENT and have now been inspected. No family or unordered triple crosses partitions. Exact image/prompt hashes do not cross partitions; expected sharing within families is retained. Entity identities, pair configurations, language grammar and renderer can be shared. No frozen holdout or mixed-split source file is read. The new generator is deterministic given source, Pillow/runtime, seed and output location; output paths are included in records, so fresh-directory reproduction must compare logical content/image bytes rather than blindly equating path-dependent record hashes.

## Diagnostic baselines and information gain

| Feature | Held-family lookup BA | Inspected-stratum BA ceiling |
|---|---:|---:|
| Majority | 1/3 | 1/3 |
| Text-only oracle-parsed roles | 2/3 | 2/3 |
| Image-only oracle-parsed roles | 2/3 | 2/3 |
| Role assignment / modality assignment | 1/3 | 1/3 |
| Prompt/content/answer-order template | 1/3 | 1/3 |
| Unordered entity identity | 1/3 | 1/3 |
| Rendered position geometry | 1/3 | 1/3 |
| Combined nonsemantic metadata above | 1/3 | 1/3 |
| Dual-verified joint symbolic oracle | 1 | 1 |

Raw accuracy differs because U has half the support: majority U yields 1/2 accuracy but BA 1/3. Lookup ties follow the fixed Yes/No/U order; unseen features use the training-majority label. The ceiling computes the maximum class-weighted correct mass per inspected feature stratum, so unseen held-family keys cannot make an identity shortcut appear absent. Floating-point summaries approximate these exact fractions.

For this construction, combining both premises supplies an additional 1/3 BA over the best oracle-parsed single-modality roles. A single premise may exclude one answer without establishing the query. Statistical independence of each modality from labels is neither required nor claimed. These baseline features assume perfect extraction of entity roles/geometry; they are not text-only or image-only VLM forwards. The new benchmark has **zero model outcomes**. Higher-than-2/3 VLM BA alone would still not prove an internal compositional mechanism.

## Reproduction and limits

`python scripts/build_publication_benchmark.py --output .cache/publication_benchmark_rebuild --families 6` (or prefix with `ulvo run . --`). Requires existing core dependencies, no model, network or ULVO API. Existing output directories are rejected; never overwrite the versioned evidence.

Automated tests verify full factor support, dual-oracle labels, query-reversal consistency, exact baseline ceilings, family isolation and historical preservation. Semantic manual inspection of every renderer/color-name pairing is pending. Six families and their many variants do not establish broad visual generalization. Other untested feature classes, model pretraining contamination and perceptual errors remain possible. Family-level independent confirmation requires a separately frozen collection.
