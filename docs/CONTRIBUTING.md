# Contributing to CrossModalLab

CrossModalLab supports **two execution modes** with the same scientific source code.

| Mode | Environment | Typical user |
|---|---|---|
| **A — ULVO-managed** | `ulvo run . -- python ...` using `.ulvo/manifest.toml` + `requirements.lock` | Primary developer (Windows 11, RTX 2050) |
| **B — Standard Python** | `venv` / `uv` / `pip` + `requirements-core.txt` (+ optional GPU layer) | Any other contributor |

ULVO is **optional**. No scientific script, test, or library may import ULVO APIs.

## 1. Clone

```bash
git clone https://github.com/Voidwalker-AT/CrossModalLab.git
cd CrossModalLab
```

## 2. Environment setup

### Mode A (ULVO)

```powershell
ulvo project inspect .
ulvo run . -- python -c "import sys; print(sys.version)"
```

Do not create a project-local `.venv` on the primary machine.

### Mode B (no ULVO)

```bash
python3.11 -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-core.txt
```

GPU extras: see `docs/DEPENDENCIES.md`.

## 3. Symbolic tests (CPU, no CUDA)

```bash
# Mode A
ulvo run . -- python -m unittest discover -s tests -v

# Mode B
python -m unittest discover -s tests -v
```

## 4. Lightweight smoke tests

```bash
python scripts/check_portability.py          # paths, config, no-ULVO import check
python scripts/preflight_hardware_check.py   # optional GPU/CUDA report
```

## 5. GPU availability

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python scripts/preflight_hardware_check.py
```

If CUDA/bitsandbytes/bfloat16 are missing, GPU experiment scripts must raise an explicit capability error. They must not silently change model or quantization.

## 6. Model and dataset paths

Defaults are repository-relative. Overrides:

- `CROSSMODAL_HF_HOME` / `HF_HOME`
- `CROSSMODAL_MODEL_SNAPSHOT_DIR`
- `CROSSMODAL_DATASETS_DIR`
- `CROSSMODAL_ARTIFACTS_DIR`

Template: `configs/local.env.example` (copy to `.env`, which is gitignored).

On the primary machine, leave overrides unset to reuse `E:/cache/huggingface` without redownloading.

## 7. Running experiments

```bash
# Mode A example
ulvo run . -- python scripts\profile_smolvlm2_quantized.py

# Mode B example (GPU stack installed)
python scripts/profile_smolvlm2_quantized.py
```

Shared pins live in `configs/experiment_config.json`:

- Model: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- Revision: `482adb537c021c86670beed01cd58990d01e72e4`
- Quantization: 4-bit NF4 (bitsandbytes)

Do not change these for “convenience” on weak hardware; use CPU symbolic workflows instead or document a non-comparable run clearly.

## 8. Reproducibility checklist

- Record Python version, package source (ULVO vs venv), and `torch.__version__` / CUDA availability.
- Keep the same seed and prompt/candidate strings from `configs/experiment_config.json`.
- Write new artifacts under `artifacts/evaluations/` with unique names; do not overwrite peers’ files.
- Label synthetic demos `is_synthetic_demo: true` (never mix into empirical aggregates).

## 9. Git workflow

- `main` = stable integration branch.
- Create a feature branch per change: `git checkout -b feature/<short-name>`.
- Open a pull request for reviewed changes.
- **No force pushes.** **No committing credentials, `.env`, model weights, or large image trees.**
- Do not commit `.venv/`, `__pycache__/`, or HF caches (already in `.gitignore`).

```bash
git checkout -b feature/portable-paths
# ... edit, test ...
git add <files>
git commit -m "Describe why the change is needed"
git push -u origin feature/portable-paths
# open PR on GitHub
```

## 10. Scope boundaries

- Do not begin Phase 1 circuit tracing or dashboard implementation in a portability PR.
- Do not modify benchmark ground truth JSONL by hand.
- Do not fabricate results or citations.
- Model downloads and destructive moves require explicit authorization.
