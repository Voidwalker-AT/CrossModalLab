# CrossModalLab manuscript workspace

Working title: **Mechanistic Analysis of Cross-Modal Relational Reasoning and Evidence Insufficiency in Vision-Language Transformers**.

This is a preliminary research draft, not a submission-ready claim of a discovered circuit or an effective new loss. Start with [the manuscript](manuscript/draft.md), [evidence registry](evidence_registry.md), [claims registry](claims_registry.md), and [submission plan](submission_plan.md). Verified literature metadata and the limits of inspection are in [references](references/verified_sources.md). [Figure inventory](figures/inventory.md) separates conceptual schematics from empirical results.

The unit of independent evaluation is the family, not each related variant. Development results have been inspected repeatedly. They cannot serve as an independent test. The final holdout is excluded from this task. TODO markers identify missing evidence; they are not completed experiments.

Rebuild the stored-data tables without models, CUDA, ULVO imports, or network access:

```sh
python paper/tables/build_development_tables.py
```

On the primary machine the same script can run through `ulvo run . -- python ...`. Full scientific source also supports ordinary Python with the documented core/GPU requirements; do not install a second environment just to read this paper. Local model/optimizer/RNG binaries remain excluded from Git. Small raw completions, identities, hashes, telemetry, and checkpoint metadata provide reviewable evidence; a partner needs the documented local assets to reproduce GPU execution.
