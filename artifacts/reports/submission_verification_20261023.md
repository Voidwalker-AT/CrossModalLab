# October 28 submission preparation: verification

Prepared 2026-10-23 against baseline `e3b0869595e31f9b0473a237b7de4ad41048a60f`. The final archive records its actual source commit independently in `PACKAGE_MANIFEST.json`.

## Dashboard and visual audit

All five rendered views were checked with installed Edge through the existing browser automation runtime. Desktop 1440x1100 and mobile 390x844 screenshots are stored as `dashboard_submission_<view>_<desktop|mobile>_20261023.png`. All original 18 evidence images loaded. The review inspected typography, panel ordering, legend/query distinction, controls, graphs, charts, navigation, contrast and mobile containment. No verified dashboard defect required a source redesign or repair.

Fresh interaction checks passed: node Enter/edge Space inspection; query toggle; compatible fork orderings; forward/reverse premise proof paths; no fork proof path; class/checkpoint switching; correct semantic answer versus malformed output; decoder-layer and candidate scoring selection; recall/loss/memory charts; provenance download; all five mobile views without horizontal document overflow. No JavaScript page errors or external requests were observed. This is scoped browser evidence, not a full accessibility certification or a human projector rehearsal.

## Demonstration and manuscript

The five-minute script uses `xc_ho_0011::underdetermined`, `::entailed` and `::contradicted` with actual stored outputs and full companion JSON. The fork's frozen and CE+ES-5 raw answers are exactly `No`. CE-50's contradicted raw completion has a leading space, No and 15 U+FFFD characters; semantic parsing and exact formatting remain separate. The scripted route was exercised mechanically; human delivery time is not measured.

The preliminary article includes the requested scientific sections, conclusion and 13 primary-source reference entries. It is supplied as editable Markdown and browser-printable two-column HTML, without asserting a confirmed IEEE venue/template. Five original figures have vector SVG and PNG companions. The six-condition CSV is generated from stored development records. All five manuscript images rendered successfully and the HTML had no mobile horizontal overflow. Claims/evidence registries were preserved unchanged. Literature summaries retain their recorded abstract/metadata inspection limits; exhaustive novelty and full-text review remain incomplete.

## Exact checks executed

| Check | Result |
|---|---|
| `npm --prefix dashboard/frontend run check` | PASS: TypeScript |
| `npm --prefix dashboard/frontend run build` | PASS: production bundle |
| `ulvo project doctor .` | PASS |
| `ulvo run . -- python -m unittest discover -s tests -v` | PASS: 296 tests, including 10 dashboard/API and four archive-integrity tests |
| `python -m unittest discover -s tests -v` | PASS: 296 discovered, 204 passed / 92 expected absent-Torch skips on Python 3.13 |
| `ulvo run . -- python scripts/check_portability.py` | PASS |
| `ulvo run . -- python -m compileall -q src scripts tests submission paper/figures/build_submission_figures.py paper/manuscript/build_html.py` | PASS |
| `ulvo run . -- python .cache/mission_quality.py` | PASS: local links, JSON/syntax, 133 historical hashes and byte-identical original table rebuild |
| `python .cache/dashboard_refinement_quality.py` | PASS: 40 pilot-history hashes; selected text contrast 5.55-7.06 |
| `node .cache/dashboard_submission_smoke.cjs` | PASS: original running app |
| `node .cache/dashboard_packaged_smoke.cjs` | PASS: extracted offline archive |
| `node .cache/submission_visual_review.cjs` | PASS: ten desktop/mobile screenshots, five HTML images, mobile containment |
| `python submission/smoke_package.py <extracted-root>` | PASS: documented ordinary-Python server CLI, health/API/static page and 18 PNGs |
| `python submission/verify_package.py <extracted-root>` | PASS: inventory, sizes and SHA-256 identities |

The standard-Python table rebuild initially changed three last-digit floats, reflecting Python 3.13 versus recorded 3.11 summation behavior. Rebuilding with 3.11 restored the original bytes; no original result was revised. No test failure remains unresolved. Exact command output is retained locally under `.cache/submission/`.

## Portability and package safety

The prebuilt package ran locally on ordinary Python 3.13; a separate isolated-process inspection confirmed neither Torch nor ULVO was imported. It requires Python 3.11+, no GPU/model/CUDA/ULVO/npm at runtime. Node 24.15.0 and npm 11.12.1 are present; locked Vite's requirement is Node `^20.19.0 || >=22.12.0`. No dependency installation or upgrade was performed.

The explicit allowlist excludes the full dataset, final-holdout images, checkpoint/model/optimizer binaries, caches, node_modules, Git metadata and external credentials. Archive creation scans selected payloads for credential patterns and records every file's byte identity. Existing reachable Git history was separately scanned (460 blobs, 19,495,814 bytes) with no matches in the stated token/private-key patterns. This is a scoped scan, not a universal secrecy guarantee. Historical research files and all 11 previously untracked evidence files remain preserved. No new inference, training, restricted evaluation or causal patching was performed.

Physical partner-machine testing remains **PENDING**. So do author/affiliation agreement, university format confirmation and a human timed/projector rehearsal. No external submission was requested or performed.

## Verdict

**CONDITIONAL GO for October 28**: the software and preliminary materials are prepared and locally verified. Finish the human/partner checks above before the live session and submission. Scientific efficacy, independent generalization, calibrated abstention, novelty and a relation-specific causal mechanism remain unverified. Overall confidence is high for the scoped local demo; publication-quality research remains incomplete. No full new GPU experiment was appropriate for this preparation.
