# CrossModalLab: October 28 demonstration package

University Deep Learning project demonstration and **preliminary** research paper, prepared October 23, 2026. The December 1 target is a submission-quality research manuscript; no venue or acceptance is promised.

## Start the packaged dashboard

Extract the complete archive to a writable directory. From its root:

```sh
python dashboard/backend/server.py --port 8765
```

Open **http://127.0.0.1:8765**. Stop with Ctrl+C. Python 3.11+ is the only runtime prerequisite for the prebuilt package. No pip command, model, CUDA, GPU, ULVO, internet connection or npm command is needed to run it. If the port is occupied, use `--port 8766` and open that port. Do not stop unfamiliar processes.

The source checkout additionally needs the approved locked frontend dependencies to rebuild. See [dashboard setup](../../dashboard/README.md). Vite requires Node `^20.19.0 || >=22.12.0`; this machine uses Node 24.15.0 / npm 11.12.1. Do not upgrade packages for the demonstration.

## Presentation materials

- [Five-minute script](five_minute_demo.md): timed speaking route and controls.
- [Exact demonstration records](demo_examples.json): full prompts, raw outputs, hashes and solver orderings.
- [Preliminary manuscript](../../paper/manuscript/draft.md) and printable `paper/manuscript/draft.html`.
- [Figure inventory](../../paper/figures/inventory.md) and [numeric table](../../paper/tables/submission_metrics.csv).
- [Reproducibility and limitations](reproducibility.md).
- [Submission checklist](../../paper/submission_checklist.md).

The archive root contains `PACKAGE_MANIFEST.json` with a SHA-256 for every payload file and the source Git identity. It contains only selected development evidence and runtime sources. No adapter/model weights, optimizer binaries, caches, credentials, node_modules, full benchmark manifest or final-holdout images are included. The original research checkout remains the authoritative full history; this package is a read-only presentation subset.

## Before the live session

Use the production bundle, connect power, set browser zoom to 100%, choose a desktop projector resolution of at least 1280 pixels, and close notifications manually. Run the script once with the actual display. Keep the manuscript HTML and figure PNGs open as an offline fallback. Saved screenshots document the inspected interface; they are a fallback, not a claim that a static image is an interactive application.

Physical partner-machine startup, author agreement, university file-format requirements and the human timed/projector rehearsal remain pending until actually completed. No external submission has been made.
