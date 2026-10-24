# Phase 3 dataset and split audit

2026-10-24. Source: `artifacts/phase3/20261024_v2/audit.json`, its selected-record extract and evaluation manifest. Reproduce with `python scripts/run_phase3_cpu_audit.py --output <fresh-checkout-directory> --checkpoints` on the primary machine with existing trusted checkpoint binaries. The optional checkpoint check needs Torch; dataset generation and symbolic checks use ordinary Python/core dependencies.

## What was actually inspected

The audit reconstructs only the 50 scheduled training examples and 18 displayed validation examples from their pinned generator. All 68 prompt hashes, image hashes and labels must match the saved run identities; original selected PNGs are checked. It never opens the original mixed train/validation/test JSONL or the frozen final record file. Reconstruction images are local duplicates, not new observations. The existing metadata/checksum registries may be read to identify the restricted partition; no frozen row or image is inspected. Full held-out disjointness is therefore **UNVERIFIED**, not certified by the selected audit.

The 68 records cover 56 families: 50 scheduled train families and six validation families, with no family overlap. They have 68 distinct prompt hashes and 68 distinct image hashes, with zero exact cross-split prompt/image duplicates and zero shared unordered entity triples in this selected scope. Eleven entity names overlap train/validation, and the renderer and prompt grammar are shared. Family-disjoint does not mean vocabulary-, template- or distribution-independent. Only the first five scheduled train examples have completed CE+ES updates; the other 45 are planned training, not new evaluation.

Historical split metadata reports 12 originally called test families already exposed to development and a replacement Phase 1C locked partition. The manifest's `ho` names are entity-pool conventions, not proof of frozen membership. The Phase 2C-C 18-case slice and dashboard 18-case slice are development observations; they are not pooled as independent replications. Final-holdout integrity and exposure history need a separately authorized custodian review before one-time evaluation.

## A confirmed conditional-label shortcut

The historical three-case construction includes only forward chain, reverse chain and one fork orientation. A textual role pattern corresponding to bridge left-of query object occurs only with Yes. A reverse visual role occurs only with No. An oracle-parsed single-modality lookup fitted on the 50 scheduled train records reaches balanced accuracy 2/3 on the 18 separate validation records, using either text roles alone or image roles alone. Each modality has two role groups, one deterministic. This is a dataset diagnostic using supplied symbolic roles, **not a measured VLM behavior or proof of shortcut use**. It assumes perfect extraction of those roles from the visible modality.

Earlier constant-condition, bag-of-words and layout baselines do not exclude this structured shortcut. A solver's finding that either individual premise does not logically entail the query does not imply that labels are statistically unpredictable from that premise within a restricted generator. Do not interpret BA 2/3 on this slice as evidence of joint cross-modal reasoning without counterfactual controls.

## Visual identity and nuisance confounds

Inspection of the first new image and the legacy palette confirmed `violet_star` renders as a diamond. Other legacy names use stylized proxies: pyramid→diamond, prism→hexagon, cylinder→rounded rectangle. Historical evidence is preserved; its perception interpretation must acknowledge these mappings. The new development version uses literal 2D shape names and records the renderer-to-prompt mapping. The first generated candidate with legacy names is preserved as superseded; it received no model forwards.

Original v2 family variants also use different layout RNG seeds and rotating paraphrases. Those are controlled families but not pixel-identical counterfactuals. They do not isolate a changed text premise alone. The new factorial set renders once per visual direction and reuses the exact image across text/query changes. Visual reversal keeps layout RNG fixed. Bounding boxes are supplied by the renderer; independent pixel-level segmentation/manual checks beyond inspected examples remain pending.

## New development data, not a new final test

`artifacts/phase3/20261024_v2/factorial_development/` contains 12 families × 8 combinations of visual direction, textual direction and query direction, seed 2026102403: 96 records and 24 PNGs. Each family has two Yes, two No and four Undetermined cases. Use balanced accuracy/class recall, report this deliberate 1:1:2 support, and do not present raw accuracy against a 1/3 baseline. A constant-U rule has raw accuracy 1/2 and BA 1/3.

Mixed-pool rendered entity triples cannot duplicate a within-pool contrast-v2 triple by construction; this does not require opening held-out records. Vocabulary aliases change to literal shape names while rendering primitives/grammar remain related. This is a documented development shift, not proof of universal semantic independence from every historical dataset.

All 96 labels pass independent reachability and linear-extension checks. The new set has four role groups per modality, zero deterministic role groups and maximum label purity 1/2. **The single-modality lookup BA remains 2/3.** One premise can eliminate a label without determining the answer; above-chance partial predictability is inherent here, not necessarily label leakage. Counterbalancing removes deterministic strata, not all logical information. This nuance replaced an initially incorrect chance-level expectation in the regression test before any GPU result.

## Admission rules

Only image pixels and exact prompt text reach a processor. Filenames, variant IDs, ground-truth labels, oracle explanations, condition metadata and hashes are evaluator-side. Identical answer-choice boilerplate names all labels and is not a per-case answer cue. Reject unknown/test/final splits, demos, duplicate IDs, wrong oracle labels and family-crossing partitions. New development data become exposed upon inspection; never rename them as independent confirmation. A new final evaluation requires an independently frozen protocol and separate human authorization.
