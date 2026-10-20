# CrossModalLab

**Academic Project Title:**  
*Mechanistic Analysis of Cross-Modal Relational Reasoning and Evidence Insufficiency in Vision-Language Transformers*

[![Research Status](https://img.shields.io/badge/Status-Phase%200C%20(Model%20Acquisition%20%26%20Benchmark%20Verification)-blue.svg)]()
[![Hardware Bound](https://img.shields.io/badge/Hardware-4GB%20VRAM%20%7C%2016GB%20RAM-orange.svg)]()
[![Methodology](https://img.shields.io/badge/Methodology-Mechanistic%20Interpretability-purple.svg)]()

---

## 1. Project Overview & Scientific Motivation

Compact Vision-Language Models (VLMs) demonstrate remarkable perceptual recognition on isolated objects and standard visual question answering tasks. However, their capacity for **compositional relational reasoning** across multimodal premises remains fragile and prone to failure. When tasked with chaining an observed visual relationship (Premise 1) with an asserted textual relationship (Premise 2) to infer an unstated cross-entity relationship (Query), models frequently fail even when they recognize both premises independently. Furthermore, when presented with relational graphs containing missing evidentiary links (such as underdetermined forks where $A$ and $C$ are related to $B$, but their mutual relation is unconstrained), models frequently hallucinate definite directional conclusions rather than identifying that the evidence is insufficient.

**CrossModalLab** investigates the internal representations, attention routing, and computational sub-modules responsible for these failure modes. By decomposing model behavior using representation probing, causal activation patching, and counterfactual interventions, the project investigates three core research questions:
1. **RQ1 (Locus of Relational Composition):** In which Transformer layer(s) and computational sub-modules (self-attention, cross-attention if present, or feed-forward MLPs) is the transitive relationship $R(A, C)$ formed or lost?
2. **RQ2 (Mechanistic Failure Mode Attribution):** What is the causal breakdown when multi-premise reasoning fails? Is failure driven by perceptual errors, textual comprehension deficits, cross-modal binding collapse, or late-stage language prior overrides?
3. **RQ3 (Representation of Epistemic Insufficiency):** Do internal hidden representations encode the topological difference between complete chains and underdetermined forks? If so, why does the decoding/readout head fail to map this state to an explicit abstention token?

> [!NOTE]
> All proposed failure mechanisms and causal attributions in this repository represent open, falsifiable scientific hypotheses currently under active empirical investigation. No mechanism is asserted as established until verified by causal interventions on local target models.

---

## 2. Formal Setting: Chain-Versus-Fork Relational Reasoning

We formalize relational reasoning over a strict, transitive, asymmetric directional relation $R$ (e.g., strictly $\text{left-of}$ on distinct objects):
* **Forward Chain (Entailed $\top$):** $R(A, B) \land R(B, C) \models R(A, C)$.
* **Reverse Chain (Contradicted $\bot$):** $R(B, A) \land R(C, B) \models \neg R(A, C)$.
* **Underdetermined Fork (Insufficient Evidence $?$):** $R(A, B) \land R(C, B) \not\models R(A, C)$ and $\not\models \neg R(A, C)$.

A critical contribution of this experimental setting is the formal separation of **Contradicted** (provably false) from **Underdetermined** (insufficient evidence) relational queries, evaluating whether compact VLMs exhibit calibrated abstention or default to forced-choice hallucination.

---

## 3. Hardware Constraints & Candidate Models

The empirical scope of this repository is strictly bound by local laboratory hardware:
* **GPU:** NVIDIA GeForce RTX 2050 Laptop GPU (4096 MiB VRAM ceiling, 50W TGP, Compute Capability 8.6).
* **Host RAM:** 16 GB total (~5.2 GB available).
* **Storage Allocation:** Local workspace isolated to drive `E:\Workspace\Active\CrossModalLab`. Drive `C:` is quarantined due to an ongoing external storage audit.
* **Pinned Primary Research Model:** `HuggingFaceTB/SmolVLM2-2.2B-Instruct` @ revision `482adb537c021c86670beed01cd58990d01e72e4` with 4-bit NF4 (bitsandbytes). See `configs/experiment_config.json`. Legacy pilot artifacts also reference `SmolVLM-500M-Instruct`; that is not the current pinned experiment identity.
* **What is established:** The symbolic chain-vs-fork engine, controlled benchmark ground truth, and early behavioral pilot/audit records in `artifacts/`.  
* **What is not established:** Causal circuit claims, unique failure-mechanism attributions, or dashboard-based interventions (open hypotheses / future phases).

---

## 4. Quick Start (Two Supported Modes)

CrossModalLab runs in two modes. **ULVO is optional.** Both modes use the same scientific source code, the same pinned model revision, and the same experiment configuration (`configs/experiment_config.json`).

### Mode A — ULVO-managed (primary developer machine)

```powershell
cd E:\Workspace\Active\CrossModalLab
ulvo project inspect .
ulvo run . -- python -m unittest discover -s tests -v
ulvo run . -- python scripts\check_portability.py
```

Uses `.ulvo/manifest.toml` + `requirements.lock` (CUDA-enabled PyTorch). Do not create a project-local `.venv` on this machine.

### Mode B — Standard Python (any contributor; no ULVO required)

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

CPU-only symbolic tests and controlled-image generation need **no GPU**.  
GPU VLM experiments additionally need torch/bitsandbytes — see `docs/DEPENDENCIES.md`. Missing CUDA produces an explicit capability error; the code does **not** silently switch models or quantization.

### Configure paths

| Variable | Purpose |
|---|---|
| `CROSSMODAL_HF_HOME` / `HF_HOME` | Hugging Face cache root |
| `CROSSMODAL_MODEL_SNAPSHOT_DIR` | Explicit pinned snapshot path |
| `CROSSMODAL_DATASETS_DIR` / `CROSSMODAL_ARTIFACTS_DIR` | Dataset / artifact roots |

Template: `configs/local.env.example`. On the primary machine, leave overrides unset to reuse the existing `E:/cache/huggingface` weights (no redownload).

More detail: `docs/CONTRIBUTING.md`, `docs/DEPENDENCIES.md`, `docs/ASSET_MANIFEST.md`.

---

## 5. Repository Structure

```text
CrossModalLab/
├── AGENTS.md                            # Agent instructions, GRAVITAS protocol, and scientific integrity rules
├── README.md                            # Project overview and research status (this file)
├── .gitignore                           # Python, data science, artifact, and image exclusions
├── pyproject.toml                       # Portable package metadata (core = Pillow only)
├── requirements.lock                    # Authoritative CUDA/full lock (Mode A / ULVO)
├── requirements-core.txt                # Portable CPU core (Mode B)
├── requirements-gpu.txt                 # Optional GPU layer without torch pin (Mode B)
├── configs/
│   ├── experiment_config.json           # Shared pinned model/quantization/seed/prompt config
│   └── local.env.example                # Environment-variable template (copy to .env)
├── docs/                                # Scientific specifications and collaboration docs
│   ├── DEPENDENCIES.md                  # Mode A vs Mode B dependency strategy
│   ├── ASSET_MANIFEST.md                # Dataset/model asset classification & acquisition
│   └── CONTRIBUTING.md                  # Clone, setup, tests, Git workflow
├── src/
│   ├── config/                          # Portable paths + shared experiment configuration
│   ├── symbolic/                        # Formal graph reachability engine
│   └── evaluation/                      # Candidate likelihood scoring
├── scripts/check_portability.py         # CPU-safe portability smoke check
└── artifacts/                           # Versioned research outputs and telemetry
│   ├── evaluations/                     # Initial pilot test results (initial_pilot_15_results.json)
│   ├── reports/                         # Formatted markdown evaluation reports
│   └── telemetry/                       # GPU and memory profiling records (hardware_profiling_run.json)
├── datasets/                            # Ground-truth benchmarks and realistic evaluation sets
│   ├── controlled/                      # 180 controlled procedural samples (images, manifests, jsonl)
│   ├── real_world/                      # GQA (30 images) and COCO val2017 (30 images) candidate subsets
│   ├── benchmarks/                      # R-Bench, HallusionBench (19 stimuli), and MoHoBench subsets
│   └── metadata/                        # Model and master dataset acquisition manifests
├── docs/                                # Scientific specifications and dashboard documentation
│   ├── research_spec.md                 # Scientific problem, formal logic, RQs, and 6 competing hypotheses
│   ├── literature_review.md             # Systematic review of foundational and 2025–2026 prior art
│   ├── novelty_matrix.md                # Fine-grained collision audit and differentiation grid
│   ├── experiment_protocol.md           # Chain-vs-fork experimental protocol and control batteries
│   ├── mechanistic_methods.md           # Probing, causal interventions, failure taxonomy, and memory rules
│   ├── dashboard_spec.md                # Interactive Transformer Research Laboratory Dashboard specification
│   ├── artifact_schemas.md              # Validated JSON schemas and synthetic demo isolation rules
│   └── dashboard_roadmap.md             # 5-stage dashboard implementation roadmap (D1–D5)
├── src/                                 # Ground-truth symbolic reasoning engine
│   └── symbolic/                        # Formal graph reachability, linear extension counting & benchmark generator
├── scripts/                             # Preflight, acquisition, profiling, and evaluation scripts
│   ├── acquire_real_world_datasets.py   # Pipeline acquiring GQA, COCO, HallusionBench, MoHoBench subsets
│   ├── download_primary_model.py        # Checksum-verified acquisition of SmolVLM-500M-Instruct
│   ├── generate_controlled_benchmark.py # Procedural 180-sample benchmark and Pillow image renderer
│   ├── profile_model_gpu.py             # Hardware profiling and 2-pass repeatability verification
│   ├── run_initial_pilot.py             # Initial 15-case evaluation pilot on SmolVLM-500M-Instruct
│   ├── validate_artifact_schemas.py     # JSON Schema Draft 2020-12 validator for artifact schemas
│   └── verify_all_datasets_and_images.py# Integrity and readability checker for 259 benchmark images
└── tests/                               # Comprehensive unit tests (47 tests, 100% passing)
```

---

## 6. Research Integrity & Scientific Standards

CrossModalLab adheres to the following core scientific principles:
1. **Probe Decodability $\neq$ Causal Usage:** Probes measure information accessibility in latent representations to an external classifier, not that downstream layers causally rely on those features to produce an output.
2. **Probe Failure $\neq$ Feature Absence:** Probes failing to decode an attribute linearly does not prove the feature is absent from the network (it may be non-linear or distributed).
3. **Causal Non-Uniqueness:** Causal claims require intervention evidence (activation patching), but patching a component alone does not prove it is the unique necessary and sufficient mechanism without exhaustive path patching and circuit ablation.
4. **No Attention Conflation:** Attention weights are routing distributions, not causal explanations. The dashboard and documentation never claim attention maps prove causal influence.
5. **No Hallucinated Citations:** All referenced literature is verified against published conference proceedings or arXiv preprints with explicit disclosure of inspection depth.
6. **Synthetic Demo Quarantine:** All demonstration artifacts are tagged `is_synthetic_demo: true` and excluded from empirical statistics.
