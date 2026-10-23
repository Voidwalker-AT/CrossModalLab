# CrossModalLab dashboard foundation — October 28 demonstration track

## Scope and result

The user clarified that October 28 is the university Deep Learning project demonstration/preliminary-paper milestone, with the working interactive dashboard as the primary deliverable. December 1 remains the longer-term research/publication-quality target. No mandatory university template or specific IEEE venue is confirmed. Further training was deferred at the user's direction.

Implemented a functioning local React + TypeScript workbench with five navigable views: Architecture Explorer, Evidence & Reasoning, Failure Investigator, Experiment Analytics, and Reproducibility. The Python backend uses the standard library and existing symbolic solver; no Python package was installed. It runs with ordinary Python and does not import Torch or require ULVO/GPU/model weights. The same-origin loopback server serves the production build and read-only allowlisted evidence API.

## Dependencies and execution

The user explicitly approved the plan in `dashboard/DEPENDENCY_PLAN.md`. Node 24.15.0 and npm 11.12.1 were verified before installation. Exact direct versions: React/React DOM 19.3.1, Vite 8.3.5, TypeScript 7.0.2, React/React DOM types 19.3.0. Vite's registry metadata requires Node ^20.19.0 or >=22.12.0. Installation used `npm install --ignore-scripts --no-audit --no-fund`, a workspace npm cache, and workspace TEMP/TMP. It added 23 packages locally. The lockfile records integrity hashes and transitive/platform dependencies. No global install, ML dependency change, or environment/toolchain provisioning occurred.

Frontend scripts run TypeScript checking before the Vite production build. Built assets, node_modules and caches are ignored. Direct versions and package-lock are committed. Backend startup: `python dashboard/backend/server.py --port 8765`; frontend build: `npm --prefix dashboard/frontend run build`. Primary users may run the Python command through ULVO. Full partner setup/demo instructions are in `dashboard/README.md`.

## Real evidence, not fabricated internals

The catalog joins original CE and CE+ES experiment JSON on all 18 sample IDs and source prompt/image hashes. Six conditions include frozen standard/abstention, CE step 5, CE+ES step 5 standard/abstention, and historical CE step 50. Only the six repeatedly inspected validation families are exposed. The dashboard packages their exact source records and 18 original PNG files (36,376 bytes total) to work without an external dataset image cache. Every image hash matches its original. No final-holdout records/images, weights or optimizer binaries are packaged.

Ground-truth graphs are recomputed using the existing reachability solver and exhaustive linear-extension oracle; disagreement rejects the case. Premises show visual/textual modality and left-of direction; forks show the compatible orderings. Raw model output is preserved. Class recall, balanced accuracy, macro-F1, invalid-output rate, unsupported definite answers, over-abstention, exact answer compliance, and confusion matrices are recomputed from recorded outputs. The historical semantic parser's acceptance of malformed suffixes remains visible: CE step 50 has semantic BA 2/3 but exact-label compliance 0/18.

Architecture dimensions come from pinned-run metadata. Layer selection explores a functional schematic, not measured layer activations. No attention maps, neuron traces, calibrated confidence, or verified causal circuit are invented. Evidence export includes source hashes and limitations. The five-step telemetry is actual stored data; it is not labeled as a completed 50-step ES ablation. Supplied-evidence sufficiency and model confidence remain distinct. The two-token Undetermined summed-score limitation is disclosed.

## Restoration audit and deferred research

Before the priority change, the read-only real-model audit validated run `56e277e421e442b08d33b107fc934494`, immutable identity, checkpoints 0–5 and journal. The pinned local Qwen NF4 model loaded; saved adapter values, AdamW moments/counters and Python/NumPy/CPU/CUDA RNG restored exactly. Frozen base parameters passed numerical comparison. **Zero optimizer steps were performed** and historical run files remained unchanged. The run remains paused at five. Report: `artifacts/training/phase_2i_b_continuation_20261023/resume_validation_74fc874e242448e1b58350a72be19091.json`. This audit is not a new training result or resumed-trajectory experiment.

## Verification

| Check executed | Result |
|---|---|
| `npm --prefix dashboard/frontend run build` | TypeScript + Vite production build passed |
| `npm ls --depth=0` in frontend | All six exact direct versions present |
| `ulvo run . -- python -m unittest discover -s tests -p test_dashboard.py -v` | Seven tests passed |
| `ulvo run . -- python -m unittest discover -s tests -v` | 289 tests passed, no skips |
| `python -m unittest discover -s tests -v` | 289 tests passed; 92 optional-dependency skips |
| `ulvo run . -- python scripts/check_portability.py` | Passed |
| `ulvo run . -- python -m compileall -q src scripts tests dashboard/backend` | Passed |
| `ulvo project doctor .` | Passed; active leases include the running demo server |
| Ordinary Python ArtifactStore import | 18 cases available; Torch absent from imported modules |
| Historical preservation and table rebuild | All 133 recorded historical hashes unchanged; paper tables byte-identical |
| Requirements/config/dataset/scientific source diff | No changes to ML requirements, ULVO manifest, configs, datasets or src |
| Browser interaction | Passed with installed Edge, desktop 1440×1100 and mobile 390×844 |

Seven API regressions cover real metric semantics, split/demo/oracle/image identity, synthetic analytics rejection, portable-image fallback, read-only methods, unknown-case/path containment, loopback Host restrictions/security headers and repeated-read immutability. They perform no optimizer steps or model load. Existing tests' optional synthetic fixtures remain separate from empirical data.

Browser checks exercised navigation, error/class/search filters and empty state, decoder-layer selection, checkpoint metric changes, original image decoding, provenance download and mobile containment. JavaScript page errors: zero. External network requests: zero. The initially unavailable bundled Chromium was not downloaded; testing used the installed Edge browser. A mobile grid overflow was reproduced and fixed, then the same check passed. An empty CSS import warning was removed and a final clean build passed. A temporary edit helper hit Windows' default text encoding; it was corrected to explicit UTF-8 before its intended edit was applied. Final application checks passed.

Screenshots: `dashboard_foundation_desktop.png`, `dashboard_foundation_architecture.png`, `dashboard_foundation_analytics.png`, `dashboard_foundation_mobile.png`; machine-readable browser result: `dashboard_browser_check.json`. Browser actions used an already available testing runtime; no browser library was added to project dependencies.

## Remaining work and verdict

**CONDITIONALLY READY for university demonstration rehearsal.** The working dashboard foundation and portable setup are verified locally. A physical second-machine partner run, university-format confirmation and presenter rehearsal are NOT TESTED and remain required before declaring the final October 28 submission ready. The preliminary manuscript/evidence registries are available; manuscript polish and publication-quality independent evaluation remain on the research track. The dashboard does not provide live GPU instrumentation or causal-intervention execution.

Next increment: partner production-build dry run and a rehearsed case-by-case presentation, then prioritize supported measurements and usability feedback. Do not launch further experiments merely to populate the interface. The specification's larger intervention lab, general activation explorer and calibration workflow remain deferred.

Checks executed and passed are listed above. Failed checks remaining: none. Skipped/not applicable: optional ML tests in standard Python; live GPU/inference, training, final-holdout evaluation and intervention execution were outside this dashboard scope. Overall confidence: high for the locally verified artifact-backed foundation; conditional for the university demonstration until partner rehearsal and format requirements are confirmed.
