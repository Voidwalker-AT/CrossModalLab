# CrossModalLab

**Academic Project Title:**  
*Mechanistic Analysis of Cross-Modal Relational Reasoning and Evidence Insufficiency in Vision-Language Transformers*

[![Research Status](https://img.shields.io/badge/Status-Phase%202E%20(Behavioral%20%26%20Feasibility%20Complete)-blue.svg)]()
[![Hardware Bound](https://img.shields.io/badge/Hardware-4GB%20VRAM%20%7C%2016GB%20RAM-orange.svg)]()
[![Methodology](https://img.shields.io/badge/Methodology-Mechanistic%20Interpretability-purple.svg)]()

---

## 1. Project Overview

Compact vision-language models (VLMs) can recognize objects and answer simple visual questions, yet still fail at **compositional relational reasoning** across multimodal premises. When a model must chain a visual premise with a textual premise to infer an unstated relationship—or admit that evidence is insufficient—it often forces a definite Yes/No instead of abstaining.

**CrossModalLab** studies these failures with:

- A symbolic chain-vs-fork engine (entailed / contradicted / underdetermined)
- Controlled synthetic benchmarks with solver-verified ground truth
- Behavioral evaluation of SmolVLM2-2.2B and Qwen3-VL-2B
- Sparse residual-stream capture and exploratory probing
- Pre-registered causal intervention designs (execution gated)

> [!NOTE]
> Probe decodability ≠ causal model usage. No unique circuit has been established. Behavioral failures are observations, not proven neural mechanisms.

---

## 2. Formal Setting

Relation \(R\) is strict **left-of** on distinct entities:

| Topology | Logic | Expected label |
|---|---|---|
| Forward chain | \(R(A,B) \land R(B,C) \models R(A,C)\) | Yes (entailed) |
| Reverse chain | reverse chain ⇒ \(\neg R(A,C)\) | No (contradicted) |
| Underdetermined fork | \(R(A,B) \land R(C,B)\) does not decide \(R(A,C)\) | Undetermined |

**Underdetermined ≠ False.** Forcing a direction on a fork is an abstention / epistemic-calibration failure.

---

## 3. Models & Hardware

| | Primary | Secondary |
|---|---|---|
| Model | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | `Qwen/Qwen3-VL-2B-Instruct` |
| Revision | `482adb537c021c86670beed01cd58990d01e72e4` | `89644892e4d85e24eaac8bacfd4f463576704203` |
| Quantization | 4-bit NF4 (bitsandbytes) | 4-bit NF4 (bitsandbytes) |
| License | Apache-2.0 | Apache-2.0 |

**Lab hardware (primary):** NVIDIA RTX 2050 (4 GB VRAM), 16 GB RAM, Windows 11.  
Shared experiment config: `configs/experiment_config.json`.

**Established:** symbolic engine, controlled benchmarks, multi-phase behavioral audits, passive activation capture, cross-model smoke tests.  
**Not established:** causal circuits, publication-grade probe generalization, effective fine-tuning.

---

## 4. Quick Start

### Interactive research dashboard — October 28 university demo

The artifact-backed React + TypeScript dashboard has five views: Architecture Explorer, Evidence & Reasoning, Failure Investigator, Experiment Analytics, and Reproducibility. It runs without loading a model or requiring a GPU. See [dashboard setup and demo script](dashboard/README.md) and the [submission checklist](paper/submission_checklist.md).

After frontend dependencies are installed from the committed lockfile:

```sh
npm --prefix dashboard/frontend run build
python dashboard/backend/server.py
```

Open http://127.0.0.1:8765. ULVO users may prefix the Python command with `ulvo run . --`. The same real 18 development examples and their exact images are packaged for both contributors; no final-holdout data or model weights are bundled. Live instrumentation and causal interventions are outside this dashboard foundation.

### Mode A — ULVO (primary lab machine)

```powershell
cd E:\Workspace\Active\CrossModalLab
ulvo project inspect .
ulvo run . -- python -m unittest discover -s tests -v
ulvo run . -- python scripts\check_portability.py
```

Uses `.ulvo/manifest.toml` + `requirements.lock` (CUDA PyTorch). Do not create a project-local `.venv` here.

### Mode B — Standard Python (no ULVO)

```bash
git clone https://github.com/Voidwalker-AT/CrossModalLab.git
cd CrossModalLab
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements-core.txt
python -m unittest discover -s tests -v
python scripts/check_portability.py
```

Symbolic tests and controlled-image generation are **CPU-only**.  
GPU VLM work needs torch/bitsandbytes — see `docs/DEPENDENCIES.md`. Missing CUDA raises an explicit capability error (no silent model/quantization fallback).

### Path overrides

| Variable | Purpose |
|---|---|
| `CROSSMODAL_HF_HOME` / `HF_HOME` | Hugging Face cache root |
| `CROSSMODAL_MODEL_SNAPSHOT_DIR` | Explicit pinned snapshot path |
| `CROSSMODAL_DATASETS_DIR` / `CROSSMODAL_ARTIFACTS_DIR` | Dataset / artifact roots |

Template: `configs/local.env.example`. On the primary machine, leave overrides unset to reuse existing `E:/cache/huggingface` weights.

Further reading: `docs/CONTRIBUTING.md`, `docs/DEPENDENCIES.md`, `docs/ASSET_MANIFEST.md`, `docs/ulvo_cuda_env_recovery.md`.

---

## 5. Repository Layout

```text
CrossModalLab/
├── AGENTS.md                 # Agent operating rules (GRAVITAS)
├── README.md
├── .gitignore
├── pyproject.toml
├── requirements.lock         # CUDA/full lock (Mode A)
├── requirements-core.txt     # Portable CPU core (Mode B)
├── requirements-gpu.txt      # Optional GPU layer (Mode B)
├── .ulvo/manifest.toml       # Optional ULVO connector
├── configs/
│   ├── experiment_config.json
│   ├── local.env.example
│   ├── phase_1c/ … phase_2e/ # Frozen manifests & split specs
│   └── phase_0d2/            # Behavioral benchmark manifest
├── docs/                     # Specs, collaboration, dependency, novelty
├── src/
│   ├── config/               # Portable paths + experiment config
│   ├── symbolic/             # Chain-vs-fork solver & generator
│   ├── evaluation/           # Behavioral harness & candidate scoring
│   ├── instrumentation/      # Passive capture, recovery, telemetry
│   ├── causal/               # Intervention design (CPU validators)
│   ├── probes/               # Linear probes & confounding analysis
│   ├── models/               # SmolVLM2 + Qwen loaders
│   └── training/             # Loss/schema utilities (design-only)
├── scripts/                  # Benchmarks, acquisition, audits, smoke tests
├── tests/                    # Unit / portability tests
├── datasets/                 # Controlled + external benchmark subsets
└── artifacts/                # Evaluations, reports, instrumentation (small JSON)
```

---

## 6. What Has Been Run (summary)

| Phase | Focus | Outcome |
|---|---|---|
| 0 | Spec, literature, 500M pilot | Foundation |
| 0D.2 | 36-case controlled behavioral pilot | Balanced-accuracy / abstention gaps |
| 1B–1E | Cross-modal dataset + confound controls | 240-family contrast-v2 set |
| 1F | Linear probes | Exploratory decode (selection-biased) |
| 2A–2B | Causal design + exploratory patching | Protocol frozen; limited pilot |
| 2C | 500M cleanup, Qwen NF4, paired eval | Shared fork failures on both models |
| 2D | Format / evidence-insufficiency audit | Text-only abstention OK; cross-modal forks fail |
| 2E | Dataset & QLoRA feasibility design | No training executed |

Full reports live under `artifacts/reports/` and `docs/`.

---

## 7. Research Integrity

1. **Probe decodability ≠ causal usage**  
2. **Probe failure ≠ feature absence**  
3. **Activation patching ≠ unique circuit**  
4. **Attention weights ≠ causal explanation**  
5. **No hallucinated citations** (inspection depth disclosed)  
6. **Synthetic demos quarantined** (`is_synthetic_demo: true`)  
7. **Historical artifacts preserved** — new runs write new files  

---

## 8. Contributing

- `main` is the integration branch; use feature branches + PRs  
- No force-push; no credentials; no model weights in Git  
- ULVO optional; standard Python supported  
- See `docs/CONTRIBUTING.md`

---

## 9. License & Citation

Model weights: Apache-2.0 (Hugging Face model cards).  
External datasets: see `docs/ASSET_MANIFEST.md` for licenses.  
Cite this repository via GitHub URL and the commit hash of the release you used.
