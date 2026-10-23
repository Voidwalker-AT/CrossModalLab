# Scientific dashboard refinement — October 28 demonstration

## Inspection and scope

Retained the existing React/TypeScript/Python dashboard, visual theme, artifact catalog, checkpoint selector, case filters, source images and comparison table. The inspected baseline already provided reliable stored-data access and metrics. Its main weaknesses were a static graph without an explicit query, separated input/logic/prediction panels, insufficient distinction between output format and semantic correctness, limited architecture explanations, and missing visible evaluation-contract details.

No dependencies, experiment settings, datasets, model environments or historical results were changed. No training, model download, inference, final-holdout access or causal intervention was performed. The existing CE+ES run remains paused at five steps; CE step 50 is historical and separate.

## Scientific and visual refinements

1. **Interactive evidence graph:** keyboard/click inspection of nodes and premise edges; visual premises solid green, textual premises dashed gold, direction shown by arrowheads. A dashed purple query is explicitly labeled query-only, not supplied evidence. Query visibility and proof highlighting can be toggled. Entailed cases highlight forward premise paths; contradicted cases highlight reverse paths, with asymmetry explained. Forks have no proof path, and their compatible orderings show both query truth values. Countermodels are labeled logical possibilities, not new observed scenes. Existing reachability and exhaustive-linear-extension solvers determine status; witness paths contain only recorded premise edges.
2. **Failure Investigator:** desktop input → ground-truth logic → recorded prediction layout, with the same panels stacked on mobile. Includes exact input image, full prompt/abstention augmentation, class, raw output, parsed label, exact-format compliance, semantic correctness, category, source record and hashes. Unsupported Yes/No and incorrect abstention are distinct; malformed generation is an independent format category. Thus a correct parsed answer can still be shown as malformed. No category claims an internal neural cause.
3. **Architecture Explorer:** six selectable conceptual stages cover vision encoding, visual-token processing, multimodal integration, language decoder, output projection and full candidate-sequence scoring. Language layer count/hidden size are sourced from pinned-run metadata. Attention, MLP, residual connections, multimodal contextual interaction and the research role/limits of intermediate activations are explained. No measured attention/activation values are fabricated. Undetermined's two-token summed-score limitation is explicit.
4. **Analytics:** added per-class recall with support counts, actual CE/ES/total loss traces, and peak allocated/reserved GPU allocator memory traces. Exact values are available in point titles. Existing BA, macro-F1, invalid rate, exact compliance, unsupported definitive answers, over-abstention, confusion matrices, checkpoint comparisons and host-memory/gradient table remain available. Captions explain development reuse, different examples per training step, and absence of CE+ES step-50 results.
5. **Reproducibility:** added full model ID, dataset path/hash, split and family count, historical parser, generation settings, objective/seed, configuration and schedule hashes, and explicit paused-at-five status. Clarified that the displayed current Git SHA identifies dashboard code, not the original execution commit for every artifact. Existing artifact hashes, revision, runtime and evidence export remain available.
6. **Usability:** retained the original design with stronger caption/status contrast, a compact horizontal case library in Failure Investigator, responsive comparison panels, keyboard graph controls, hover explanations, semantic/format badges, and existing loading/error/empty states. Selected caption/status text pairs have measured contrast ratios 5.55–7.06:1; this is not a full accessibility certification.

## Security findings

Reachable-history scan at base `2136f89405eefc2b767ba10513852ee779646ce8`: 848 objects, 446 unique blobs, 18,564,962 bytes; no credential-pattern matches. A fresh staged-file scan is part of the commit gate. Patterns cover Google OAuth/client-secret JSON, Google API keys, GitHub/OpenAI-style tokens, AWS access keys and private-key headers. No values were printed. The externally shared OAuth file was not opened or copied. Pattern scanning is scoped and does not guarantee absence of every secret format or inspect dangling/external/private files.

The loopback-only binding, Host checks, allowlisted cases, contained static/image paths, SHA-256 validation, read-only methods, CSP and text rendering remain in place and pass regressions. No history rewrite was needed. Details: `dashboard/SECURITY_AUDIT.md`.

## Fresh verification

| Check executed | Observed result |
|---|---|
| `npm --prefix dashboard/frontend run check` | TypeScript passed |
| `npm --prefix dashboard/frontend run build` | Production build passed, 16 modules |
| `ulvo run . -- python -m unittest discover -s tests -p test_dashboard.py -v` | 10 dashboard tests passed |
| `ulvo run . -- python -m unittest discover -s tests -v` | 292 passed, no skips |
| `python -m unittest discover -s tests -v` | 292 passed, 92 optional-dependency skips |
| `ulvo run . -- python scripts/check_portability.py` | Passed |
| `ulvo run . -- python -m compileall -q src scripts tests dashboard/backend` | Passed |
| `ulvo project doctor .` | Passed; demo-server leases remain active |
| Historical integrity | 133 earlier historical hashes and 40 pilot-preflight hashes unchanged; paper tables rebuild byte-identically |
| Protected dependency/config/source diff | No ML/core/frontend dependency, lockfile, environment, experiment config, dataset or scientific src changes |
| Installed Edge browser checks | Desktop 1440×1100 and mobile 390×844 passed; all five views |

New regressions cover existing-edge witness paths, both fork countermodels, semantic/format categories, and portable evaluation-contract metadata. The category unit sentinels are explicitly synthetic and excluded from empirical analytics. Existing tests still cover original image hashes, portable fallback, demo rejection, API immutability/path containment and Host/header restrictions.

Browser checks cover left/center/right order, keyboard node/edge inspection, query toggling, compatible orderings, entailed/contradicted highlighting, missing fork proof, semantic versus format correctness, six architecture stages, layer selection, recall and telemetry charts, provenance and downloads, all 18 original images and all five mobile layouts. Page errors: zero. External requests: zero. A first browser script used an overly specific accessible-name expression for the candidate-scoring button and timed out; the locator was corrected to the actual component text, and the complete check passed. No application retry or experimental retuning occurred.

New screenshots are named `dashboard_refinement_*_20261023.png`. Earlier foundation screenshots/results were preserved. The machine-readable browser record is `dashboard_refinement_browser_20261023.json`.

## Demonstration readiness

**CONDITIONALLY READY for the October 28 university demonstration rehearsal.** The requested refinement is implemented and verified locally. Partner-machine rehearsal and confirmation of university submission requirements remain NOT TESTED/unconfirmed. Live neural measurements, independent generalization, causal circuits and further training remain outside this refinement. The next presentation should contrast a fork countermodel with a contradicted reverse path, then show a semantically correct but malformed CE answer.

Checks executed/passed are listed above. Failed checks remaining: none. Skipped/not applicable: 92 optional ML tests under standard Python; live inference/training/holdout/causal experiments intentionally not run. Overall confidence: high for the tested local artifact-backed dashboard; conditional for the final university presentation until partner rehearsal is complete.
