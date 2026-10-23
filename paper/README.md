# CrossModalLab manuscript workspace

Working title: **Mechanistic Analysis of Cross-Modal Relational Reasoning and Evidence Insufficiency in Vision-Language Transformers**.

This is a preliminary research draft, not a submission-ready claim of a discovered circuit or an effective new loss. Start with [the manuscript](manuscript/draft.md), [evidence registry](evidence_registry.md), [claims registry](claims_registry.md), and [submission plan](submission_plan.md). Verified literature metadata and the limits of inspection are in [references](references/verified_sources.md). [Figure inventory](figures/inventory.md) separates conceptual schematics from empirical results.

The unit of independent evaluation is the family, not each related variant. Development results have been inspected repeatedly. They cannot serve as an independent test. The final holdout is excluded from this task. Uncompleted research is described explicitly in the limitations. The portable draft and printable [HTML article](manuscript/draft.html) use a conventional two-column academic layout without claiming compliance with any specific IEEE venue.

Rebuild the stored-data tables without models, CUDA, ULVO imports, or network access:

```sh
python paper/tables/build_development_tables.py
```

Figure authoring uses an existing matplotlib installation: `python paper/figures/build_submission_figures.py`. Printable HTML authoring uses an existing Python Markdown installation: `python paper/manuscript/build_html.py`. Neither authoring library is required to read the delivered files or run the dashboard; this preparation installed nothing. The [October 28 package instructions](../submission/2026-10-28/README.md) include the exact demonstration script, limitations and archive verification.

On the primary machine the same script can run through `ulvo run . -- python ...`. Full scientific source also supports ordinary Python with the documented core/GPU requirements; do not install a second environment just to read this paper. Local model/optimizer/RNG binaries remain excluded from Git. Small raw completions, identities, hashes, telemetry, and checkpoint metadata provide reviewable evidence; a partner needs the documented local assets to reproduce GPU execution.
