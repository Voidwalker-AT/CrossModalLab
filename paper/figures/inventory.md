# Figure and table inventory

| Figure | Intended content | Evidence / status | Release condition |
|---|---|---|---|
| 1 | Forward/reverse chain versus fork | [Logical schematic](reasoning_and_pipeline.md); conceptual, no empirical numbers | Confirm strict left-of semantics; mark as schematic |
| 2 | Research pipeline and model components | [Pipeline schematic](reasoning_and_pipeline.md); generic architecture, not a layer-accurate model rendering | Verify checkpoint-specific module mapping before architectural attribution |
| 3 | Frozen Qwen versus corrected CE at steps 5 and 50 | `../tables/development_summary.json` derived from stored Phase 2I-A raw completions | Include family-cluster uncertainty, parser caveat, repeated-development-use label; no independent-generalization caption |
| 4 | Underdetermined failures | Real raw completions in `../tables/family_predictions.csv` plus source report | Preserve example IDs and hashes; add existing rendered image only after local inspection, never generate a replacement |
| 5 | Probe accessibility / intervention controls | Phase 1F summary and Phase 2B surviving intervention records | Correlational probe panel only; no validated relation-specific causal circuit currently available |
| 6 | CE+ES ablation | Five-step pilot only after checkpoint/parameter acceptance | A feasibility pilot panel must not be labeled a completed 50-step ablation or an efficacy result |

Table 1: model/revision/quantization/hardware. Table 2: family-disjoint split and sampling protocol. Table 3: stored development metrics and descriptive clustered intervals. Table 4: pilot loss/gradient/resource/checkpoint trajectory. Table 5: limitations and evidence classification. Tables 3–4 are generated from real records; missing experiments remain missing rather than zero-filled. No fabricated quantitative plot has been created.
