# Scientific evidence gallery and sharing

The working terminal command on the registered primary machine is:

```powershell
wsctl run crossmodallab dashboard
```

It launches the real production frontend and read-only standard-library backend on `http://127.0.0.1:8765`. The launcher fingerprints frontend inputs, builds when needed using installed exact locked dependencies, and never installs anything. Port conflicts report the process PID/name and return 2 without stopping it. Alternative: `python -B scripts/launch_dashboard.py --port 8766`, or set `CROSSMODAL_DASHBOARD_PORT=8766` before the WSCTL command. Ctrl+C closes the server; the Windows WSCTL batch shim may ask `Terminate batch job (Y/N)?`, in which case answer Y. It leaves no model or server child running.

## Registration and external repair

WSCTL v1.0's supported registry is the workspace-level `workspace.json`. The reviewed entry is [wsctl.project.json](wsctl.project.json). Only that entry was added, after a byte-for-byte backup inside ignored `.cache/wsctl/`; all 28 previous projects and unrelated settings were compared unchanged. A future primary setup can preview `python -B dashboard/register_wsctl.py --registry <workspace-root>/workspace.json`, then use `--apply` only after authorization. An existing different entry is never overwritten automatically.

The installed WSCTL dispatcher initially hid child failures behind exit 0. After explicit user approval, only run-dispatch status propagation was repaired in the two external WSCTL scripts described in [the repair note](WSCTL_EXIT_CODE_FIX.md). Backups and the global files are local only and are not committed to CrossModalLab. Regression command: `pwsh -NoProfile -File dashboard/verify_wsctl.ps1 -WorkspaceRoot <workspace-root>`. Success, fixture failure 7, missing command 1, occupied port 2 and existing-project resolution are checked. The existing WSCTL panel checks were also executed. This local external repair is not a portable project requirement.

The partner can build the frontend from its existing locked dependencies and run `python -B scripts/launch_dashboard.py`, or use the earlier prebuilt submission archive with `python -B dashboard/backend/server.py`. Neither path needs WSCTL, ULVO, CUDA or weights. Physical partner-machine verification remains pending.

## Central evidence folder

Each versioned `submission/CrossModalLab_Research_Evidence_Oct28_<release>/` folder contains a browser-openable `Gallery.html`, `README.md`, machine-readable `MANIFEST.json` and nine numbered directories. The adjacent ZIP has the same basename. Both generated payloads are Git-ignored; a small release manifest is recorded under `artifacts/reports/`. Original research files are never moved or deleted.

The earlier verified October 24 release remains at `submission/CrossModalLab_Research_Evidence_Oct28_20261024/`, with its original `CrossModalLab_Research_Evidence_Oct28.zip` unchanged. The human-centered refinement is `submission/CrossModalLab_Research_Evidence_Oct28_HumanCentered_20261024_r2/`, with an adjacent ZIP of the same versioned basename. Initial unfinalized candidates are also preserved. Always pass the same fresh `--output` path to preparation, browser capture and finalization. Finalization rejects an existing manifest, ZIP or report before writing anything.

The primary gallery now follows a single case from problem and input to logical ground truth, recorded answers, separate historical neural evidence, results and limitations. All visuals remain in expandable collections. Each image has a descriptive title, explanation, evidence category and a local provenance-manifest link; click the image to enlarge it. Exact Qwen prompts/answers are also in `08_Reproducibility/Exact_Recorded_Answers.json`. The original PNG bytes and raw completions are preserved.

The gallery contains desktop/mobile application captures; all 18 original validation PNGs; four-checkpoint input/output comparisons for every case; symbolic graph exports; all 200 activation rows and historical Smol norm/probe plots; all six condition confusion matrices; pilot telemetry; paper figures; registries, provenance and the demonstration guide. SVGs scale cleanly and PNGs work in slides. Open it without Python, Node, WSCTL or ULVO. No large tensor, model, adapter, optimizer, cache or credential file is included.

## Rebuild procedure

1. Use the existing curated neural bundle. Revalidate it from local original tensors only through an explicitly reviewed version of `scripts/build_neural_evidence.py`; it refuses silent overwrite. Never acquire new GPU evidence for this export.
2. With already installed matplotlib, run `python -B scripts/export_research_evidence.py`. It creates a new evidence folder and never overwrites an existing one. Use `--output <new-folder-within-checkout>` for another release. Authoring libraries are optional and are not required to view the result.
3. Capture actual desktop/mobile views and render the exported SVGs to PNGs using the available browser automation runtime. Every capture must have a source, sample ID where relevant and classification in `08_Reproducibility/browser_capture_log.json`. The supplied browser procedure is `dashboard/capture_evidence.cjs`; it uses an explicitly selected existing Playwright installation and installed Edge, with no downloads or package installation.
4. Run `python -B scripts/export_research_evidence.py --finalize` with the same output path. It checks the catalog, scans payloads for credential patterns, writes SHA-256/size/provenance metadata, creates the ZIP, extracts it into workspace-local scratch space and checks every payload hash and CRC. Refuse mismatches rather than silently dropping files.

Interpretation rules are in [the neural audit](NEURAL_EVIDENCE.md). Smol observations must not be attached to a Qwen failure. No validated relation-specific causal mechanism or CE+ES superiority is claimed. Static plots are evidence exports, not live model execution. Human rehearsal, professor-specific file requirements and partner testing remain separate tasks.

## Refinement release verification

The browser capture procedure checks all five desktop/mobile views, graph proof paths, case-specific checkpoint conclusions, seven tour stages, Enter-key activation, default collapsed raw details, exact raw clipboard content and absence of external requests. The offline gallery is checked from an extracted ZIP with HTTP blocked, all images decoded, mobile overflow checked and local links validated. PNG renderings of comparisons are derived from the exact metadata-bearing SVGs. Loss exports use only five discrete measured points, not smooth learning curves.

Use `python -m unittest discover -s tests -p 'test_dashboard*.py' -v` for dashboard, launcher, integrity and narrative/export regressions. Full release checks additionally include the ULVO and ordinary-Python suites, TypeScript, production build, portability and compilation. See the run-specific verification report under `artifacts/reports/`.
