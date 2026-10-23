# Packaged development evidence

These are the 18 existing Phase 2I validation examples from six families, copied from `datasets/cross_modal_contrast_v2/` to make the offline dashboard portable. No new benchmark examples or model outputs were generated. Every image preserves its original byte SHA-256, and the case records preserve source prompt/image hashes, split, premises, query and gold label. The Python adapter validates these against all displayed condition records.

All records are controlled scientific benchmark stimuli with `is_synthetic_demo:false`; they are not fabricated tutorial predictions. No final-holdout examples are packaged. Model answers are read from the original tracked experiment JSON files, not embedded as new evidence here. The dashboard must not be used to infer that these repeatedly inspected development families provide an independent test.
