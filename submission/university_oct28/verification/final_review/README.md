# Final dashboard review — October 25, 2026

**CONDITIONAL GO for the university demonstration.** Browser and software checks passed. Actual classroom projection, a timed human rehearsal and a second physical machine remain unverified. This is a presentation review, not new scientific evaluation.

Starting checkout and freshly fetched origin/main: `b1ade6a42da6bdfe7aad85f6bb18e0337ce79984`. Work was isolated on `codex/final-dashboard-review`. Historical raw evidence and the frozen October 28 archive were preserved.

## Observed problems and corrections

| Priority | Observed before correction | Minimal correction | Observed after correction |
|---|---|---|---|
| P1 | SVG labels fell below 12 rendered pixels in laptop/mobile cards; no enlargement control existed. | Enlarge the existing image, symbolic graph, historical Smol plots and Qwen telemetry, without duplicating figure state or adding packages. | 24 enlargement interactions passed across three sizes. Smallest enlarged chart label: 14.25 pixels. Mobile uses horizontal panning inside the enlarged figure. |
| P1 | Decoder layer numbers were 10 pixels and pale; confusion headings and comparison labels retained pale inherited colors. | Increase layer numbers to 13 pixels; darken affected labels and navigation; increase the evidence legend. | Fresh rendered screenshots and computed styles confirm the correction. This is a targeted contrast review, not a comprehensive WCAG certification. |
| P2 | The recorded answer was several panels below the original input, especially on mobile. | Add a compact selected-case summary before the input; put the correct/recorded answer and evidence-insufficiency explanation into the first tour stop. | Correct `Undetermined` versus recorded `No`, case ID and checkpoint are together. Detailed prompt/hash/provenance sections remain expandable. |

No broken scientific result or causal claim was found. No redesign, dependency change, backend change, experimental inference or training was performed.

## Actual browser screenshots

Edge/Chromium was launched on this machine, against the real WSCTL-started production server. Every view was captured at viewport sizes 1920×1080, 1366×768 and 390×844. Full-page PNG heights exceed viewport heights; `-viewport.png` files are exact requested viewport dimensions. These are rendered application captures, not design mockups.

| View | Before: full page, projector / laptop / mobile | After: full page, projector / laptop / mobile |
|---|---|---|
| Failure Investigator | [1920](before/1920-failure-investigator.png) / [1366](before/1366-failure-investigator.png) / [390](before/390-failure-investigator.png) | [1920](after/1920-failure-investigator.png) / [1366](after/1366-failure-investigator.png) / [390](after/390-failure-investigator.png) |
| Evidence & Reasoning | [1920](before/1920-evidence-reasoning.png) / [1366](before/1366-evidence-reasoning.png) / [390](before/390-evidence-reasoning.png) | [1920](after/1920-evidence-reasoning.png) / [1366](after/1366-evidence-reasoning.png) / [390](after/390-evidence-reasoning.png) |
| Architecture Explorer | [1920](before/1920-architecture-explorer.png) / [1366](before/1366-architecture-explorer.png) / [390](before/390-architecture-explorer.png) | [1920](after/1920-architecture-explorer.png) / [1366](after/1366-architecture-explorer.png) / [390](after/390-architecture-explorer.png) |
| Experiment Analytics | [1920](before/1920-experiment-analytics.png) / [1366](before/1366-experiment-analytics.png) / [390](before/390-experiment-analytics.png) | [1920](after/1920-experiment-analytics.png) / [1366](after/1366-experiment-analytics.png) / [390](after/390-experiment-analytics.png) |
| Reproducibility | [1920](before/1920-reproducibility.png) / [1366](before/1366-reproducibility.png) / [390](before/390-reproducibility.png) | [1920](after/1920-reproducibility.png) / [1366](after/1366-reproducibility.png) / [390](after/390-reproducibility.png) |

The before capture left the evidence gallery collapsed. Its initial DOM diagnostic counted some hidden lazy images as unloaded; that was a test visibility mistake, not a broken-image finding. The final test explicitly opens the gallery, scrolls every visible original into view, decodes it, and independently verifies all 18 image response hashes. The after Reproducibility capture therefore intentionally shows the expanded gallery.

[Mobile enlargement](after/mobile-enlarged-graph.png) shows the pan/scroll behavior. [Error-state capture](after/server-error-fixture.png) uses an explicitly injected HTTP 503 browser fixture; it is UI test evidence, not an experimental result. The loading screen was also observed while that fixture held the request pending.

## Guided-tour rehearsal: expected versus actual

All seven stops and the entire reverse route passed using keyboard Enter/Space at all three viewport sizes. First Previous and final Next are disabled. Every stop retains CE+ES step 5 and selects `xc_ho_0011::underdetermined`; the case-specific views expose the original recorded example. Architecture and analytics present their separately identified evidence, not new measurements for that case.

| Stop | Expected view / content | Actual navigation |
|---|---|---|
| 1 — Why missing evidence matters | Failure Investigator; correct Undetermined versus recorded No; calibrated-answer motivation | Matched |
| 2 — Read the image and text together | Failure Investigator; image disc left of wedge, text block left of wedge; disc–block query | Matched |
| 3 — Try both possible orders | Evidence & Reasoning; fork graph; C<A<B query false and A<C<B query true | Matched; both buttons update pressed state |
| 4 — Inspect the recorded failure | Failure Investigator; raw CE+ES5 answer No, not a causal explanation | Matched |
| 5 — Compare the checkpoints | Failure Investigator; frozen, CE5, CE50 and CE+ES5 extracted No; malformed CE suffixes preserved | Matched |
| 6 — Explore separate historical measurements | Architecture Explorer; conceptual Qwen pipeline, separately labeled SmolVLM2 captures and correlational probes | Matched; layer 18 control responds |
| 7 — Read the results with their limits | Experiment Analytics; CE+ES5 9/18 versus CE5 7/18; descriptive paired interval includes zero | Matched |

No broken interaction was observed in the tested route. Enlargement opens with focus on Close, confines Tab/Shift+Tab, closes with Escape and restores focus. Both countermodel controls work. All five navigation modules work; mobile navigation scrolls horizontally without page overflow. Actual five-minute speech duration and novice comprehension within 60 seconds remain **UNVERIFIED** until a human rehearsal.

## Scientific consistency

- 18 development-validation cases, six dependent families, one five-step CE+ES pilot seed. These are repeatedly inspected development results.
- CE+ES is paused at five updates; CE step 50 remains a different checkpoint. No CE+ES50 exists.
- Historical Smol captures/probes are separate from the selected Qwen model. Probe decodability and activation magnitudes are correlational; no relation-specific causal circuit is established.
- The symbolic graph shows independently solved supplied premises. Purple query arrows are explicitly not facts or internal model traces. Fork countermodels permit both answers; Undetermined differs from No.
- Strict output-format compliance is separate from the historical prefix-based semantic parser. Malformed CE completions are retained.
- The separate failed 108-forward screen has one attempt and zero completed score rows; it is not a successful evaluation. The successful two-case pilot is separately identified.
- 9,216 generated variants come from six families and 72 images; they are not 9,216 independent samples. No superiority/generalization claim was added.
- 173 historical run-file hashes, 123 neural source hashes, 18 original image responses and the frozen October archive passed integrity checks. Twelve pre-existing untracked evidence files were preserved and excluded from this commit.

## Verification receipts

| Check | Result / exact execution |
|---|---|
| TypeScript | PASS — `npm --prefix dashboard/frontend run check` |
| Production frontend | PASS — `npm --prefix dashboard/frontend run build` |
| Dashboard API/export/launcher/story tests | PASS, 24 — `ulvo run . -- python -m unittest discover -s tests -p "test_dashboard*.py" -v` |
| Full ULVO discovery, guarded | 343 discovered: 340 passed, three closed-data tests skipped; zero restricted opens — `ulvo run . -- python scripts/verify_phase3_integrity.py` |
| Full literal ULVO discovery | Same 343 / three skips — `ulvo run . -- python -m unittest discover -s tests -v`, with documented `CROSSMODAL_EXCLUDE_MIXED_DATASETS=1` |
| Ordinary Python guarded discovery | 343 discovered: 246 passed, 97 unavailable-stack/closed-data skips — `python scripts/verify_phase3_integrity.py` |
| Portability | PASS — `ulvo run . -- python scripts/check_portability.py` |
| Compilation | PASS — `ulvo run . -- python -m compileall -q src scripts tests dashboard` |
| Project doctor | PASS — `ulvo project doctor .` |
| Real browser | 15 view/viewport combinations; 21 forward tour stops plus full reverse traversal; 24 enlargement checks; original gallery loading, 18 image hashes, read-only APIs, loading/error states; no page errors, external requests or document overflow |

Raw outputs: [check manifest](manifest.json), [browser receipt](after/browser.json), [scientific integrity](scientific_integrity.json). Per-process closed-data guards do not constitute machine-wide filesystem auditing. No final holdout was opened by the guarded checks.

The browser regression is [dashboard/tests/final_review.cjs](../../../../dashboard/tests/final_review.cjs). It uses an already installed Playwright module through `CROSSMODAL_PLAYWRIGHT_MODULE`; the test does not install dependencies. Optional `CROSSMODAL_BROWSER_CHANNEL`, `CROSSMODAL_DASHBOARD_URL` and `CROSSMODAL_REVIEW_OUTPUT` select the existing browser, server and output directory. The default browser is installed Edge. Python dashboard operation requires neither Playwright nor ULVO/WSCTL. Existing Node v24.15.0/npm 11.12.1 were used with unchanged package versions and lockfile.

WSCTL launch responded with `read_only: true` and `gpu_loaded: false`. On Ctrl+C, the server reported a clean stop; the batch shim exited 1 after confirmation. The owned process was absent and port 8765 was released, independently checked in [shutdown.json](shutdown.json). Closing the injected error-state browser fixture also produced a server-side client-disconnect diagnostic for an in-flight provenance response; it did not break the application or change evidence.

**Passed:** all final required checks. **Failed final checks:** none. **Skipped:** closed datasets and unavailable ordinary-Python ML dependencies, as above. **Confidence:** high for the tested software route, conditional for room-scale readability, assistive-technology behavior and live delivery; no physical projector, screen-reader session or partner-machine rehearsal was performed. See the [presenter checklist](PROJECTOR_CHECKLIST.md).
