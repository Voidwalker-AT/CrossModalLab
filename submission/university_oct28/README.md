# October 28 university submission workspace

**CrossModalLab: Deep Learning demonstration and preliminary paper, October 28, 2026.** Prepared October 25. This is a human-facing entry point to existing implementation/evidence, not a new experiment, second dashboard or completed submission.

## Reading order

1. [Project overview](PROJECT_OVERVIEW.md): plain-language problem and implementation.
2. [Demo guide](DEMO_GUIDE.md): exact startup and five-minute route with real case IDs.
3. [Results summary](RESULTS_SUMMARY.md): identities, denominators and parser distinctions.
4. [Limitations](LIMITATIONS.md): completed implementation versus unverified research.
5. [Checklist](CHECKLIST.md): pending human, format and submission actions.

Technical navigation: [repository map](../../docs/REPOSITORY_MAP.md). The existing [paper source](../../paper/manuscript/draft.md), [printable HTML](../../paper/manuscript/draft.html) and [actual screenshots](screenshots/README.md) support the presentation. The paper was not rewritten for this organization.

From the repository root, run `wsctl run crossmodallab dashboard` on the registered machine, or `python dashboard/backend/server.py --port 8765` with the frontend built. Open **http://127.0.0.1:8765**, then **Start the five-minute guided tour**. No GPU/model loading occurs. First-time prerequisites and safe fixes are in the demo guide.

The original [October 28 release instructions](../2026-10-28/README.md), galleries and frozen ZIP remain unchanged. This folder adds navigation without copying datasets or moving research artifacts. Frozen releases and the later live paper/research updates have different dates and scope; do not silently rebuild the archive.

Share the agreed commit and approved presentation archive, not the entire local workspace. Exclude weights, optimizer binaries, caches, credentials and global registry backups. Local ignored ZIPs are not necessarily on GitHub. Partner startup, human rehearsal, author agreement, university format and actual upload remain pending. December 1 is a research target, not guaranteed publication. [Fresh verification](verification/README.md) records what was actually checked.
