# CrossModalLab repository map

Start with the [university workspace](../submission/university_oct28/README.md) for assessment, or [contributor setup](CONTRIBUTING.md) for development. Existing paths are unchanged.

| Find this | Location | Responsibility |
|---|---|---|
| Dashboard | [dashboard/README.md](../dashboard/README.md), `dashboard/frontend/src/`, `dashboard/backend/` | Five views, existing tour and read-only server; ignored `dist/` is generated |
| Symbolic reasoning | [solver.py](../src/symbolic/solver.py), `relations.py`, `graph.py` | Reachability/exhaustive-order ground truth, not model reasoning |
| Benchmarks | `src/benchmark/`, `datasets/`, `configs/` | Generators, manifests, asset identities and split rules |
| VLM evaluation | [behavioral_harness.py](../src/evaluation/behavioral_harness.py), [candidate_scorer.py](../src/evaluation/candidate_scorer.py), `src/models/` | Generation/parser and distinct candidate-scoring procedures; execution separately gated |
| Neural instrumentation | [passive_capture.py](../src/instrumentation/passive_capture.py), `multi_layer_capture.py` | Passive captures and integrity; dashboard does not execute them |
| Probes | `src/probes/`, `artifacts/probes/` | External decodability with selection/confounding limits |
| Causal investigation | `src/causal/`, `artifacts/causal/` | Controls/protocols and historical inconclusive results; no established circuit |
| Fine-tuning | `src/training/`, [run_state.py](../src/training/run_state.py), `artifacts/training/` | Loss/scorer, checkpoint/optimizer/RNG utilities and historical runs |
| Experimental results | `artifacts/evaluations/`, `artifacts/training/`, `artifacts/phase3/`, `artifacts/publication/` | Small records, telemetry, identities, failures and partial evidence |
| Paper | [draft.md](../paper/manuscript/draft.md), `paper/figures/`, `paper/tables/`, evidence/claims registries | Exploratory manuscript and traceable claims |
| University materials | [submission/university_oct28/](../submission/university_oct28/README.md) | Overview, demo, results, limitations and checklist |
| Historical presentation | [submission/2026-10-28/](../submission/2026-10-28/README.md), local versioned ZIPs | Original release instructions; frozen archive unchanged |
| Paths/environment | `src/config/`, requirements files, `.ulvo/manifest.toml` | Ordinary Python plus optional primary ULVO; scientific code does not require WSCTL |
| Specifications | [research_spec.md](research_spec.md), [dashboard_spec.md](dashboard_spec.md) | Design/history, including planned capabilities; current implementation is in the dashboard README |

The interface depends on `dashboard/backend/artifacts.py`, selected `dashboard/assets/`, original result paths and source hashes. Moving a file may invalidate provenance even when its contents look unchanged. Keep historical results, checkpoint directories and established caches in place. The [inventory](university_submission_inventory.md) records hygiene and preservation checks.
