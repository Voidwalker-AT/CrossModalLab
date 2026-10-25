# Five-minute live demonstration

This is a prepared route, not evidence of a completed human/projector rehearsal. Use the existing guided tour.

## Start from the repository root

Primary registered machine:

```sh
wsctl run crossmodallab dashboard
```

Portable alternative with production frontend already built:

```sh
python dashboard/backend/server.py --port 8765
```

Open **http://127.0.0.1:8765**. The health URL **http://127.0.0.1:8765/api/health** should return `status: ok`, `read_only: true`, `gpu_loaded: false`. Ordinary Python 3.11+ serves the build without CUDA, weights, ULVO or WSCTL.

A fresh source checkout needs Node `^20.19.0 || >=22.12.0`, npm and installed exact locked frontend dependencies. With those installed, run `npm --prefix dashboard/frontend run check` then `npm --prefix dashboard/frontend run build`. First-time installation is a separate explicit action in [dashboard setup](../../dashboard/README.md); do not install/upgrade during the demo. The portable convenience launcher `python -B scripts/launch_dashboard.py` rebuilds stale assets using existing packages only. The frozen prebuilt archive needs no Node/npm at runtime.

## Five-minute speaking route

Click **Start the five-minute guided tour**. It selects **CE + ES · step 5**, `xc_ho_0011::underdetermined`. Seven tour stops have Next/Previous controls; group them into this delivery:

| Time | Show / action | What to say |
|---|---|---|
| 0:00–0:30 | Research problem | A confident answer is not always supported. We distinguish false claims from questions the evidence cannot decide. |
| 0:30–1:10 | Original image/text | The image gives golden disc left of teal wedge; the text gives crimson block left of teal wedge. Is the disc left of the block? Combining both sources is cross-modal reasoning. |
| 1:10–1:55 | Evidence & Reasoning; click both ordering buttons | Disc–block–wedge and block–disc–wedge both fit. The answer is Undetermined. Green/gold arrows are premises; purple is the question. This is symbolic ground truth, not a neural thought trace. |
| 1:55–2:30 | Failure Investigator | This five-update CE+ES checkpoint answered exactly No. The evidence does not justify that answer. The neural cause remains unverified. |
| 2:30–3:05 | Four standard-prompt checkpoint cards | All four extracted answers are No here. CE outputs also have malformed suffixes. Correctness and formatting differ; this case shows no training improvement. |
| 3:05–4:00 | Architecture Explorer; layer control and historical Smol section | The pipeline is conceptual. Historical measurements below are Smol, not this Qwen case. An external probe can decode information without proving model usage. |
| 4:00–5:00 | Analytics, then exit tour/open Reproducibility | These are 18 reused development cases/six families. CE+ES5 has 9/18 correct versus CE5's 7/18, but the descriptive family interval includes zero. No CE+ES50 or independent generalization result exists. Sources make the measurements checkable. |

## Exact cases for optional comparison

Use CE+ES step five. The query is always **golden disc left of crimson block**.

| Sample ID | Image premise | Text premise | Gold / exact output |
|---|---|---|---|
| `xc_ho_0011::underdetermined` | Disc left of wedge | Block left of wedge | Undetermined / `No` |
| `xc_ho_0011::entailed` | Disc left of wedge | Wedge left of block | Yes / `Yes` |
| `xc_ho_0011::contradicted` | Wedge left of disc | Block left of wedge | No / `No` |

The `ho` substring is not proof of final-holdout use: these exact allowlisted records are development validation. Manifest selection governs scope. The [existing human-centered script](../../dashboard/human_centered_demo.md) has more speaking detail; [results](RESULTS_SUMMARY.md) identify sources without requiring an examiner to read JSON.

## Safe troubleshooting

- Port occupied: do not stop an unfamiliar owner. Use `python dashboard/backend/server.py --port 8766` and open that port, or the convenience launcher with `--port 8766`.
- API healthy but page missing: build the frontend with installed locked packages. The direct backend does not build it automatically.
- Missing Node/packages: use the approved prebuilt archive or arrange explicit setup beforehand. A model download is not a frontend fix.
- Evidence/hash error: report it and use the preserved approved archive or screenshots. Do not alter labels, disable integrity checks or regenerate experiments.
- Offline/projector problem: keep the frozen release's `Gallery.html`, paper HTML and [screenshots](screenshots/README.md) available. State when showing a static fallback.

Stop your server with Ctrl+C. If the WSCTL batch shim asks `Terminate batch job (Y/N)?`, answer Y for that owned launch. Confirm the port is released; do not stop unrelated Python/Node/GPU processes. [Verification](verification/README.md) distinguishes automated checks from pending partner-machine and human rehearsal work.
