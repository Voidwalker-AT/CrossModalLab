# Human-centered dashboard and evidence release — October 24, 2026

**VERIFIED:** the dashboard, explained figures and extracted offline gallery work on the primary Windows machine. **CONDITIONAL GO** for the October 28 demonstration: partner hardware, a human timed/projector rehearsal and final university submission requirements remain pending. Human understandability was reviewed by inspecting rendered screenshots and reading the default flow; no participant usability study is claimed.

## 1. Human-readable interface

The existing ivory/green workbench and all five views are retained. Larger body text, stronger contrast, larger original images and additional spacing prioritize explanation. The reading hierarchy is Understand → Explore → Verify. Exact model identifiers, raw strings, tensor tables and hashes remain available in expandable panels. The research source evidence and claims registries were not edited.

## 2. Failure Investigator before and after

Before: three competing small-text columns, a visible image hash, raw suffixes in repeated four-column cards, and a second repeated answer table. After: original input and logical conclusion share a readable desktop row, followed by the recorded answer and “Why does this matter?” Mobile stacks the same content. A reader can identify the image/text question, correct answer, observed error and scientific limit without expanding technical details.

The featured case is `xc_ho_0011::underdetermined`, actually development validation despite its name. Both golden disc and crimson block are left of the teal wedge. Both relative orders of disc and block satisfy the premises. All four standard-prompt checkpoints have extracted answer No and are incorrect here. Frozen and CE+ES five-step strings are clean; the CE checkpoints have malformed suffixes. No case-specific improvement is implied.

Malformed-output panels distinguish extracted answer, correctness and exact format. A shortened escaped preview never replaces the exact raw string. Browser checks copied the complete CE-50 contradicted-case string to the clipboard and compared it exactly with the source artifact. Fifteen recorded U+FFFD replacement characters are described as such; their generation/decoding cause remains **UNVERIFIED**.

## 3. Logical graphs

Plain labels explain Supported by the evidence, Contradicted by the evidence and Not enough information. Original object names, image/text colors, directed left-of arrows, query-only labeling, witness-path highlights and compatible logical orders remain available. Captions explain the selected premises directly. Countermodels are possible logical worlds, never additional observed scenes. Exported symbolic figures now have their own complete layout instead of being cropped from a larger comparison.

## 4. Neural plots

Historical frozen SmolVLM2 measurements are separated visibly from Qwen behavior. Each norm/probe chart explains the metric, selected tokens, sampled layers, units, colors, observed values and limits. Tables and identities are expandable. The default last-token probe has stored balanced accuracies 33.3%, 51.9%, 77.8% and 74.1% at layers 3, 12, 18 and 23. Its development-selected settings and 27-case/nine-family validation scope remain visible. Probe decodability does not establish model use.

The Qwen case explicitly says internal measurements have not yet been recorded for this example. No attention map, neuron trace or established causal circuit was added. All 123 curated neural source hashes were checked unchanged. Existing failed/incomplete causal controls remain unaltered.

## 5. Analytics

Recall, balanced accuracy, confusion counts, CE, ES and GPU allocator memory receive plain definitions, directions of interpretation, sample counts and limitations. Five-step losses are discrete observed points; different examples are not joined into a purported learning curve. The pilot remains one seed, five updates, 18 development cases and six families. Two correct abstentions coexist with four incorrect abstentions. No CE+ES step-50 result or superiority claim is introduced.

## 6. Offline gallery

The primary gallery is a guided story from problem → input → logic → recorded answers/checkpoints → separate historical neural measurements → results → limits. All 102 exported PNGs remain accessible in expandable collections. Each has a human title, explanation, category and provenance link. Images enlarge through ordinary local links. The extracted gallery passed 226 local-link checks, decoded 107 image occurrences representing 102 unique PNGs, and loaded the manuscript's five figures with zero network requests. Desktop/mobile layouts and keyboard story navigation passed.

## 7. Guided demonstration

The optional seven-stop tour uses existing views and Next/Previous controls. It selects the exact featured Qwen case and CE+ES five-step condition while preserving normal navigation. Enter-key operation, stage boundaries and previous/next transitions were checked. The timed script is [human_centered_demo.md](../../dashboard/human_centered_demo.md). This is prepared material, not a claim of a completed human rehearsal.

## 8. Shareable release

Use `submission/CrossModalLab_Research_Evidence_Oct28_HumanCentered_20261024_r3/Gallery.html` or the adjacent ZIP:

`submission/CrossModalLab_Research_Evidence_Oct28_HumanCentered_20261024_r3.zip`

- 14,329,203 bytes (13.67 MiB), 191 files including the manifest.
- 102 PNGs, 66 SVGs, original images, exact recorded-answer JSON, CSVs, manuscript, scripts/notes and reproducibility documentation.
- ZIP SHA-256: `24cfd58d51640ed1baa6fb3696dbd0d6f22f4cec13e18c973b1a07f0f2ae1283`.
- Source implementation commit: `5e23570fd83a38afc2c796b9ee66b60be63f1811`.
- CRC, extraction inventory, sizes and every payload SHA-256 passed. No weights, tensor binaries, optimizer states, caches or credentials are included.

The original October 24 ZIP remains byte-identical at SHA-256 `319cecc62081a98c784d39dcca1aad531c22f8c8c9dd0569aaae73d2f3d54dd2`. Earlier folders and candidates remain preserved. The r2 candidate is superseded: four long file URLs failed to decode at 267-character extraction paths; identical bytes decoded at the shorter original path. Derived activation filenames were shortened, with full sample IDs retained in metadata. The fresh r3 extraction passed the same offline check. A source-footer/caption overlap found during visual review was also corrected before the released r3 figures were generated.

## 9. Verification

Checks executed after the relevant final source edits:

| Check | Command / evidence | Result |
|---|---|---|
| Types | `npm --prefix dashboard/frontend run check` | PASS |
| Production | `npm --prefix dashboard/frontend run build` | PASS |
| Dashboard/API regression | `python -m unittest discover -s tests -p 'test_dashboard*.py' -v` and full suites | PASS; read-only, allowlist, identity and exact-string checks |
| ULVO doctor | `ulvo project doctor .` | PASS |
| Full ULVO | `ulvo run . -- python -m unittest discover -s tests -v` | 310 passed |
| Ordinary Python | `python -m unittest discover -s tests -v` | 310 discovered; 218 passed, 92 optional-Torch skips |
| Portability | `ulvo run . -- python scripts/check_portability.py`; ordinary-Python counterpart | PASS |
| Compilation | `ulvo run . -- python -m compileall -q src scripts tests dashboard` | PASS; bytecode redirected to ignored workspace cache |
| Browser | `node dashboard/capture_evidence.cjs <release-folder>` | PASS; five desktop/five mobile views, 18 original images, 18 comparison/18 graph exports, keyboard, exact clipboard, default closed raw details, download |
| Offline | `node dashboard/verify_offline_gallery.cjs <extracted-release-folder>` | PASS; all images explained, 226 links, zero network/page errors, mobile containment and keyboard |
| Historical evidence | Recorded 133-file/40-file hash inventories; 123 neural source hashes; deterministic table rebuild | Unchanged; table rebuild byte-identical |
| Staging | Credential-pattern scan, forbidden binary/cache exclusions, dependency/config comparison, `git diff --cached --check` | PASS for release staging |

Four new CPU regressions verify full logical metadata/countermodels, truthful all-wrong checkpoint conclusions, escaped/self-contained explanatory gallery content, and rejection of finalized-release collisions before mutation. Existing SVG regressions continue to verify original image bytes and every raw completion across all 18 examples.

Type/build/runtime checks used only already installed dependencies. No dependency changes, model downloads, new research training/optimizer steps, final-holdout evaluation or causal patching were performed. Unit suites exercise existing utilities with fixtures, not a new experiment. Verification logs are local under `.cache/human_*` and `.cache/evidence_verification/`; the small committed manifest contains payload identities and the browser log hash.

## 10. Git and launch

The recovery branch is preserved. Source commits are `03f3a155805f85e80bda235ff157b8a2fcad0f0a` and `5e23570fd83a38afc2c796b9ee66b60be63f1811`; this report and the r3 manifest form the release-record commit. Remote integration uses ancestor checks and ordinary pushes to main/recovery, with the final SHA reported in the completion message. No global/repository author settings were changed; identity was applied per commit. Historical untracked evidence and the superseded local r2 manifest are not staged.

`wsctl run crossmodallab dashboard` successfully launched ordinary Python on an alternate port 8766 for isolated verification, with health `{status:ok, read_only:true, gpu_loaded:false}`. The owned verification server exited cleanly and port 8766 was released; the pre-existing port-8765 server was preserved. Refresh the frontend; restart that existing server normally if it has cached an earlier Python comparison exporter. No WSCTL global file or registry changes were made in this refinement.

## 11. Limits

Partner-machine execution, screen-reader testing, a participant comprehension study, projector timing, author agreement and final university formatting remain pending. Existing observations remain exploratory: reused development data, small family counts, one pilot seed, unequal checkpoint budgets, malformed-output sensitivity and incomplete causal controls. Shorter extraction paths remain advisable on Windows. No generalization, causal mechanism, method superiority, novelty or publication acceptance is established.

## 12. October 28 verdict

**CONDITIONAL GO:** the verified application and offline sharing package are ready for a rehearsal and review by the professor/partner. Complete the human/second-machine/submission checks before claiming the external demonstration or submission itself is complete.
