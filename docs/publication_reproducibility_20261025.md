# Publication research reproducibility — October 25, 2026

This supplements the immutable October 28 demonstration; it does not replace it. The research-only evidence distribution includes the updated manuscript, figures, source-grounded comparisons and small owned raw artifacts. Copyrighted source PDFs/text, external natural photographs, caches, weights, credentials and optimizer binaries are excluded. The package is a review/evidence bundle; reproducing Python research commands uses the Git checkout and its existing dependencies.

## Recorded outcomes

Successful isolated pilot: twelve additional forwards; Smol Yes/Yes and Qwen No/Yes on the same gold No/U pair. One exposed family/two cases; no three-class accuracy/CI. Full likelihoods, token IDs, hashes and parameter comparisons remain in `artifacts/publication/20261025/pilot/`. Historical six-forward Qwen evidence is separate.

Separately approved 108-forward development screen: one attempted Smol forward, zero completed scores, no Qwen worker, no retry. Generic RAM/VRAM gate failure; exact trigger and post-termination parameter invariance UNVERIFIED. Original executed source is preserved. CPU repair adds sample-bearing gate exceptions and writes parent samples before checking; it has not been GPU executed. No new research optimizer steps, causal patches, model downloads, dependency changes or frozen-holdout evaluation occur.

The evidence audit inspects selected full-text methods/results/appendices for 21 papers, with public identifiers and source hashes. Unknown full run counts, unavailable resource/code verification and uncertain independence remain explicit. It is not an exhaustive systematic review or independent reproduction.

## CPU commands and portability

From the full repository root, ordinary Python is supported:

```sh
python scripts/check_portability.py
python scripts/verify_phase3_integrity.py
python -m unittest discover -s tests -p test_matched_screen.py -v
python scripts/analyze_publication_screen108.py --output .cache/new_screen_cpu_analysis.json
python scripts/build_publication_figures.py --output .cache/new_publication_figures
python paper/manuscript/build_html.py
```

Analysis/figure outputs must be fresh checkout-local paths. The failed GPU runner cannot be restarted into its spent admission directory. Do not execute it as a reproduction command. Full guarded discovery excludes three legacy tests that would read mixed/final record files; any unexpected restricted open fails. Standard Python without Torch skips optional Torch checks. No dependency installation is part of these commands.

Fresh local verification: guarded ULVO discovery 343 tests, 340 pass/three restricted-data skips; guarded ordinary Python 246 pass/97 skips. Both record zero denied open attempts. Literal full unittest commands also pass with `CROSSMODAL_EXCLUDE_MIXED_DATASETS=1`. ULVO doctor, portability, Python compilation, TypeScript check and frontend production build pass. Eleven newly added CPU tests cover screen admission/accounting and complete matched analysis; dashboard assertions verify failed-screen attribution separately from the successful pilot. Full raw verification logs are preserved under the dated verification directory.

Primary tool versions checked: Node v24.15.0, npm 11.12.1; ordinary Python 3.13.0; existing ULVO Python 3.11. Locked React/Vite/TypeScript packages are unchanged. The [dashboard instructions](../dashboard/README.md) remain authoritative. Run `python -B scripts/launch_dashboard.py` (or the registered local `wsctl run crossmodallab dashboard`); dashboard startup uses stored data and standard Python, with no model/GPU loading. Browser checks freshly confirm the publication panel, separate failed-screen notice, two real inputs/four model rows, desktop/mobile layout and ten manuscript figures. The owned test server shuts down cleanly. A harmless `/favicon.ico` 404 remains; no page errors were observed.

Partner-machine execution remains NOT VERIFIED; local ordinary-Python success is not a second-machine claim. Human visual annotation, individual natural-image rights, authorship and venue selection also remain pending.

## Distribution and source integrity

After committing verified source, build a fresh local review package:

```sh
python scripts/package_publication_evidence.py --output submission/CrossModalLab_Publication_Research_20261025_r1
```

The builder records the actual source revision, whitelists small evidence/source types, hashes every member and verifies ZIP bytes without including itself in the content hash. It refuses pre-existing destinations and excludes all private caches. Links outside the review bundle point to the declared GitHub source revision; it is not represented as an independently runnable model environment. The local folder and ZIP are Git-ignored; only their public manifest/verification summary is committed. No model/cache distribution is implied.

The original October 28 ZIP SHA256 remains `24cfd58d51640ed1baa6fb3696dbd0d6f22f4cec13e18c973b1a07f0f2ae1283`. Historical scientific source and artifact hashes remain separately checked. For claim gates and cost estimates, see the [gap report](../paper/publication_evidence_gap_report.md) and [standards](international_research_standards.md).

An independently reproduced Git byte-identity defect affected the recorded Qwen loader: its working-tree CRLF hash matched execution, while Git's normalized LF blob did not. Scoped `.gitattributes` entries now preserve exact bytes for the files named by these execution/generator manifests. The loader's Python AST remains identical; no scientific implementation, frozen prompt or model configuration changes. Staged source hashes are compared to both execution plans before committing. This preserves identity across checkout line-ending preferences without changing global Git settings.
