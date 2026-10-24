# Evidence standards for the CrossModalLab publication track

These are project acceptance criteria derived from the [prior-art audit](../paper/prior_art_evidence_matrix.md), not an official IEEE policy or a claim that every published paper meets them. Venue-specific templates, page limits and acceptance criteria must be checked after choosing a venue. The university October 28 archive remains frozen.

## Essential standards

1. State the exact claim and its population. A rendered development diagnostic, natural-image transfer, corrective method and causal circuit require different evidence.
2. Validate the construct: truth follows from all available evidence; insufficient is not false. Publish both oracle agreement and independent visual/human validity checks. Formal modality necessity does not establish model modality use.
3. Preserve matched controls: input hashes, entity/role/order/layout factors and all variants. Report image-only/text-only and trivial/bias baselines where composition is claimed. Do not add a control after seeing results and label it preregistered.
4. Count independent units honestly. Distinguish images, questions, matched groups, families, model configurations, seeds, candidate forwards and optimizer steps. Do not turn 9,216 variants/six families into an independent sample of 9,216.
5. Freeze splits, calibration, endpoint, exclusions and analysis before independent evaluation. Use a custodian-created split; do not inspect the old frozen holdout to repair development shortcomings.
6. Report class support, invalid/missing predictions, per-class recall, unsupported-answer and over-refusal rates. Argmax likelihood among candidates is not free generation, spontaneous abstention or calibrated confidence. State summed versus mean token likelihood explicitly.
7. Quantify uncertainty at the dependence unit. Use paired family resampling with at least 1,000 bootstrap replicates for descriptive intervals when supported; prespecify confirmatory tests/effect sizes and multiplicity correction. One family/two labels cannot support three-class balanced accuracy or a generalization CI. Undefined metrics stay undefined.
8. For causal claims, verify live alignment, no-op/sham/unrelated controls, bidirectionality, competing paths and replication. Attention/probes/logit lens remain correlational. Even patch success does not prove circuit uniqueness. See IOI pp.9–10 and patching best practices pp.3,14.
9. For a corrective method, compare same data, schedule, initialization/seed and compute against CE and paired-data-only alternatives; include retention and independent evaluation. QLoRA and generic abstention margins are prior work. Five steps cannot establish superiority.
10. Preserve provenance and failures. Exact model/source/config/input identities, attempted-forward ledgers, all metrics and resource samples accompany analysis. Missing checkpoints are failures, not implicit step-zero restarts. A stopped worker cannot receive a successful final parameter audit by inference.
11. Report legal and practical reproducibility. Code is not a substitute for data/model rights. Dataset-level annotation licensing does not automatically license every image. Keep private model weights, caches, credentials, copyrighted full papers and optimizer binaries out of evidence packages.
12. Verify contributor portability with ordinary Python. ULVO/WSCTL may help the primary machine but cannot be scientific dependencies. A local successful test is not second-machine confirmation.

## What is optional or venue-specific

More models, larger datasets and expensive interventions are useful only when they test a claim that needs them. A strong single-model causal study can exceed a broad uncontrolled leaderboard in evidentiary strength. A narrowly scoped behavioral report need not include neural interventions. Exact model counts in HallusionBench or MoHoBench are not minimum publication quotas. Font, column count, reference style and page limits belong to venue preparation; none repairs missing validity controls.

## Current gate status

VERIFIED: generated logical labels and declared factors; successful 12-forward matched pilot; exact successful-pilot parameter comparisons; preserved one-attempt failed screen; CPU admission/logic tests. PRELIMINARY: limited development behavior and historical five-step comparison. CORRELATIONAL: selected Smol probes. UNVERIFIED/MISSING: independent generalization, strong unimodal VLM controls, real-world transfer, new method efficacy, causal mechanism, exhaustive novelty clearance, second-machine execution.

No further GPU inference is automatically authorized after the failed 108-forward admission. Resource-plan estimates in the gap report are proposals only. CPU analysis, manuscript corrections and reproducibility work may continue within the authorized scope.
