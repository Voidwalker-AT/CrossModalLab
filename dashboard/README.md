# CrossModalLab scientific dashboard

For the university assessor, start with the [October 28 overview, exact demo route, results and checklist](../submission/university_oct28/README.md). This is navigation around the existing application; historical release packages remain unchanged.

Read-only React + TypeScript research workbench with a Python standard-library backend. Five functional views: Architecture Explorer, Evidence & Reasoning, Failure Investigator, Experiment Analytics, and Reproducibility. It uses recorded Qwen development results, exact benchmark images, and the existing dual-algorithm symbolic solver. No model, CUDA, PyTorch, ULVO, inference service, or external API is required to **run the dashboard**.

## Run on either contributor's machine

On the registered primary machine, run `wsctl run crossmodallab dashboard`. The portable equivalent is `python -B scripts/launch_dashboard.py`. Both genuinely start the server at localhost:8765, build stale frontend files using already installed locked packages and stop on Ctrl+C. Neither installs dependencies. A busy port is reported without terminating its owner; use `--port 8766` with the portable launcher. [Registration, exit-code repair and evidence-export procedures](EVIDENCE_EXPORT.md) document the local WSCTL dependency separately.

Failure Investigator now adds four standard-prompt checkpoint cards and downloadable self-contained comparison SVGs. Reproducibility includes a collapsible gallery of all 18 original inputs. Architecture Explorer adds [historical Smol neural diagnostics](NEURAL_EVIDENCE.md) with measured norms and external probe curves, clearly separate from Qwen. Qwen per-case activations remain NOT MEASURED; no neural failure mechanism is inferred.

Prerequisites: ordinary Python 3.11+ and an existing Node/npm installation. Vite 8.3.5 declares Node `^20.19.0 || >=22.12.0`; this machine was verified with Node 24.15.0 and npm 11.12.1. Python backend imports only standard-library/project modules. Core scientific/GPU environments are separate and unchanged.

From the repository root, after the approved dependencies are installed:

```sh
npm --prefix dashboard/frontend run build
python dashboard/backend/server.py --port 8765
```

Open **http://127.0.0.1:8765**. Primary-machine equivalent: `ulvo run . -- python dashboard/backend/server.py --port 8765`. Stop with Ctrl+C. The server binds only loopback and serves the production frontend and API from one origin. The production build needs no network access. Built files are generated locally, not committed.

For a fresh partner checkout, reproduce dependencies using the committed lockfile:

```sh
cd dashboard/frontend
npm ci --ignore-scripts --no-audit --no-fund
npm run build
cd ../..
python dashboard/backend/server.py
```

`.npmrc` keeps the npm cache at workspace `.cache/npm`, pins exact direct versions, and disables lifecycle scripts. On the primary machine also redirect process temporary files before installation/build/server commands:

```powershell
$env:TEMP = Join-Path (Get-Location) '.cache/recovery-temp'
$env:TMP = $env:TEMP
$env:npm_config_cache = Join-Path (Get-Location) '.cache/npm'
```

Run that block from the repository root; ensure the workspace temporary directory exists. Do not install anything into the Python/ULVO/CUDA environments for this dashboard. Partner installation is an explicit setup action; the application never installs or downloads dependencies automatically.

Development hot reload: run the Python server first, then `npm --prefix dashboard/frontend run dev`. Vite proxies `/api` to loopback port 8765. For the university demo, use the production build above.

## Five-minute demonstration

1. Open Failure Investigator with **CE + ES · step 5**. Inspect the first recorded underdetermined fork: image, textual premise and query.
2. Read the symbolic graph and its two consistent orderings. Explain why neither query direction follows; this is ground-truth logic, not a neural reasoning trace.
3. Inspect the stored answers across frozen Qwen, CE step 5, CE+ES step 5 and CE step 50. Exact malformed CE output is available in expandable details with a copy action; exact-format compliance is separate from historical semantic parsing.
4. Filter to errors or another logical class; switch checkpoints. Open Analytics for confusion matrices, class-sensitive metrics and actual training telemetry.
5. Open Architecture Explorer and select a decoder layer. Architecture equations are conceptual; no measured per-neuron activations or established circuit is claimed.
6. Open Reproducibility and export provenance/evidence JSON. Explain the six-family development limitation and absent independent final evaluation.

### Refined scientific walkthrough

Failure Investigator now places **supplied input → symbolic ground truth → recorded model output** in a readable question → logic → answer sequence. The input and logical explanation share the desktop row; the verdict follows below. Mobile stacks the same panels. The case library, model/checkpoint selector and original comparison table remain available. The output panel distinguishes semantic correctness from exact-format compliance. Unsupported Yes, unsupported No, incorrect abstention, wrong definite answers and unparseable answers are behavioral categories; malformed generation is a separate format category. A correct extracted answer can still be malformed.

The graph labels the query with a dashed purple arrow explicitly marked **query only, not a premise**. Visual premises use solid green edges and textual premises dashed gold edges. Click a node or edge, or focus it with Tab and press Enter/Space, for its explanation. Toggle query visibility and logical-path highlighting. Entailed paths follow the query; contradicted paths run in the reverse direction. Forks have no proof path. Select their compatible orderings to see one world where the query is true and another where it is false. These are logical countermodels, not generated scenes or additional empirical examples.

For the demonstration, start with the underdetermined case and switch between its two orderings. Then select a contradicted case with CE step 50: the extracted answer is correct, but the raw suffix makes the generation non-compliant. This demonstrates why answer correctness and instruction-format compliance are reported separately.

Architecture Explorer shows six conceptual stages: vision encoder, visual-token processing, multimodal integration, language decoder, output projection and candidate scoring. It exposes attention/MLP/residual explanations and the limits of intermediate-activation evidence. Layer selection is conceptual; no measured activations are invented. Candidate scoring explains full sequence summation and the two-token Undetermined limitation.

Analytics adds per-class recall and plots of actual five-step loss and allocator memory records. Hover points for recorded values. Losses belong to different training examples; they are not a fixed-input learning curve. Reproducibility adds model ID, source dataset identity/hash, split, parser, generation settings, configuration/schedule hashes, seed and explicit paused-at-five status.

The refinement adds no dependency and changes no experiment result. Selected text foreground/background pairs were checked above 4.5:1 contrast, with keyboard controls, restrained interactions and no external network requests in browser verification. This is not a claim of a full accessibility certification. See [the credential/API audit](SECURITY_AUDIT.md).

## Evidence and scientific limits

`assets/development_cases.json` contains only the 18 frozen validation cases, with byte-identical PNG copies under `assets/images/`. Original dataset paths/hashes remain attached. The backend checks split, demo flag, source prompt/image identities, candidate outputs, and agreement of reachability and exhaustive linear-extension solvers. It does not read the frozen final-holdout manifest or images. Controlled scientific stimuli are real benchmark artifacts (`is_synthetic_demo:false`), distinct from fabricated tutorial data.

Metrics are recomputed from stored raw answers using `strict_answer_v1`. This historical parser accepts some malformed suffixes. Exact compliance requires the stripped raw text to equal Yes, No, or Undetermined. No malformed answer is presented as a clean generation. Candidate Undetermined has two tokens; the frozen ES objective uses summed, unnormalized full-sequence likelihoods. The dashboard does not invent probability calibration, attention maps, neural traces, or new model results.

The run remains **paused at five steps**. Real-model restoration was separately verified with zero optimizer steps; continuation was deferred when the dashboard became the priority. CE's historical 50-step result is displayed as such, never as a 50-step CE+ES result. Existing failed/inconclusive causal controls remain in the evidence registry; no new patching is executed.

## API and tests

Read-only endpoints: `/api/health`, `/api/workspace`, `/api/provenance`, `/api/image/<allowlisted sample_id>`. Unknown cases/routes return 404. Failed image integrity returns 422. POST/PUT/PATCH/DELETE return 405. Filesystem paths cannot be supplied through the API. Static files are confined to `frontend/dist`; non-loopback Host headers are rejected.

```sh
python -m unittest discover -s tests -p test_dashboard.py -v
npm --prefix dashboard/frontend run check
npm --prefix dashboard/frontend run build
```

Browser verification covers five views, error/class/search filters, checkpoint metrics, layer selection, original images, export and mobile containment. Screenshots and the browser result are under `artifacts/reports/dashboard_foundation_*` and `dashboard_browser_check.json`. No browser-testing package is added to the project: verification used an already available browser automation runtime and the installed Edge browser.

## Scope and next increment

This completes an artifact-backed dashboard foundation. Live model execution, general activation exploration, intervention execution, and calibrated uncertainty are not implemented. Next: partner dry run, university feedback, and incremental supported measurements. The October 28 university deliverable is the working dashboard/demo plus preliminary paper; December 1 remains the research/publication-quality target, with venue and acceptance unconfirmed.

## Human-centered walkthrough (October 24 refinement)

The default reading flow explains what the model saw, what the evidence establishes, what it answered and why that distinction matters. Hashes, exact raw strings and model identifiers remain in expandable verification panels. Every main chart defines the metric and its limits. Historical SmolVLM2 measurements are explicitly separate from Qwen behavior.

Choose **Start the five-minute guided tour** for seven stops with Next/Previous controls. It selects `xc_ho_0011::underdetermined` and CE+ES step five; normal navigation is preserved. See [the timed script](human_centered_demo.md). Case evidence and the tour run without loading a model. Human rehearsal and independent partner-machine verification remain pending.

The new offline story gallery and versioned ZIP are documented in [EVIDENCE_EXPORT.md](EVIDENCE_EXPORT.md). Older releases remain preserved.
