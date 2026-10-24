# Milestones — university demo October 28; research target December 1

**Priority update, 2026-10-23:** the October 28 university Deep Learning submission prioritizes the working interactive dashboard and reliable artifact-backed demonstration, with a preliminary paper. December 1 remains the longer-term publication-quality research target. No university template/page limit or specific IEEE venue is confirmed. Dashboard implementation takes precedence over manuscript polish and further training. See [the active checklist](submission_checklist.md) and [dashboard instructions](../dashboard/README.md). The five-step CE+ES run remains paused; no continuation steps were taken after this priority change.

As of 2026-10-23, 39 calendar days remain. This is an internal manuscript target, **not a verified conference deadline**. No venue is selected. Venue-specific current scope, abstract/full-paper deadlines, timezone, page limits, anonymity, artifact and AI-assistance policies, preprint policy, and author eligibility must be checked on an official live call before a venue decision. Readiness is conditional; acceptance is never promised.

| Dates | Deliverable | Evidence gate |
|---|---|---|
| Oct 23–27 | Finish pilot audit, source integration, partner reproduction, full-text collision matrix | Honest failed-run reporting if necessary; no sustained training without separate approval |
| Oct 28–Nov 3 | Freeze research question, parser sensitivity analysis, independent family manifest, statistical plan | Explicit authorization before any independent/final-holdout access; no adaptive selection after outcomes |
| Nov 4–10 | Authorized matched controls and independent behavioral evaluation | Reproducible seeds, equal budgets, raw outputs, family-disjoint evaluation; stop on resource/state failures |
| Nov 11–17 | Mechanistic claim decision and replication | If circuit evidence remains absent, narrow the title/claims with authors instead of implying completed mechanism discovery |
| Nov 18–24 | Figures, uncertainty, full draft and partner audit | All numbers trace to immutable raw evidence; complete limitations and licenses |
| Nov 25–30 | Venue-policy check, author review, reproduction package, freeze | Human review, appropriate acknowledgments, no missing core evidence disguised as TODO |
| Dec 1 | Internal submission decision | Submit only if actual venue and scientific gates are satisfied; otherwise deliver a clearly preliminary report |

## Essential remaining experiments

1. Resolve any pilot feasibility/acceptance failure before approving sustained CE+ES. Compare CE and CE+ES under identical sample order, supervision spans, parser, prompts, and training budget. Audit candidate token-length effects separately without changing this frozen pilot retrospectively.
2. Before evaluation, preregister a family-disjoint independent evaluation with all three states, enough independent families to meet a chosen precision/power target, and no exposure during model/layer/parser selection. Determine sample size from a prespecified effect or interval width, not an arbitrary accuracy target. A fresh independent set may be needed because existing development families were repeatedly inspected. Final holdout use still requires explicit authorization.
3. Repeat matched training across independently preregistered seeds after approval; report variability. Five steps with one seed cannot estimate method effectiveness or stability.
4. Evaluate invalid and exact-label outputs alongside the historical parser; include underdetermined recall, false definitive answers, over-abstention on sufficient evidence, and risk/coverage only when scores and a selection rule are properly defined. Do not tune the selection threshold on independent outcomes.
5. Mechanistic work requires separately authorized, preregistered interventions with valid no-op, same-label, random-source, unrelated-token, and bidirectional controls. Current Phase 2B controls prevent a relation-specific causal claim. Test paths and alternatives before necessity/sufficiency or uniqueness claims.

## Minimum statistical and reproducibility package

Use family-cluster paired bootstrap (at least 1,000 draws; planned 2,000) with model comparisons paired on identical IDs. Report 95% descriptive intervals and raw family counts. Repeated-development-use intervals do not repair selection bias. Specify chance baseline and primary endpoint before independent evaluation; correct head/layer/method multiple comparisons with a prespecified Bonferroni or BH procedure. Report effect sizes and uncertainty, not just significance. A claim of practical benefit requires a prespecified relevant effect and adequate precision.

Release small manifests, split rules, render seeds, asset hashes, source/runtime identities, objective definitions, exact commands, raw completions, checkpoint metadata, and structured failure logs. Exclude model weights and optimizer binaries from Git; provide a separately approved distribution/access procedure if needed. The partner must reproduce CPU checks in ordinary Python and independently verify the selected GPU protocol on suitable hardware. Clarify data/model licenses and authorship. Freeze all claims against the registries before release.

## Collision and scope risks

The October 24 Phase 3 follow-up is documented in [the progress audit](../docs/research_progress_audit.md) and [publication gap analysis](../docs/publication_gap_analysis.md). The versioned October 28 evidence archive remains unchanged. New dataset diagnostics, a partial six-forward Qwen pilot, three figures and a literature collision update supplement the preliminary manuscript; they do not close the independent-evaluation or causal-evidence gaps. The [Phase 3 roadmap](../docs/phase_3_research_plan.md) governs the next approval gates.

Winoground, What’sUp, SPHERE, reliable VQA, SelectiveNet, R-Tuning, counterfactual VQA training, QLoRA, and current reporting-bias work already cover major components ([audited sources](references/verified_sources.md)). The exact combination’s novelty remains UNVERIFIED. Missing independent evaluation and missing causal evidence are the primary scientific blockers. A systems-feasibility pilot and a polished draft cannot replace them.
