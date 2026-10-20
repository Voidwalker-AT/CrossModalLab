# Dependency Strategy — CrossModalLab

This document defines how **Mode A (ULVO-managed)** and **Mode B (standard Python)**
share one scientific codebase without forcing ULVO or CUDA onto every contributor.

## Principles

1. **One scientific implementation.** No ULVO imports in `src/`, `scripts/`, or `tests/`.
2. **`requirements.lock` is authoritative for the primary GPU lab machine** and is preserved as-is.
3. **Portable layers never invent versions.** Pins are copied from `requirements.lock` only when the package is pure-Python or broadly portable.
4. **CUDA is optional at the packaging layer, mandatory for the pinned VLM experiment.** Symbolic tests and controlled-image generation must run without torch.

## Files

| File | Audience | Contents |
|---|---|---|
| `requirements.lock` | Mode A (ULVO) / primary GPU lab | Full CUDA stack including `torch==2.6.0+cu124` via `--extra-index-url https://download.pytorch.org/whl/cu124` |
| `requirements-core.txt` | Mode B (any OS, CPU OK) | Pillow + jsonschema + psutil only |
| `requirements-gpu.txt` | Mode B (GPU optional layer) | transformers, bitsandbytes, accelerate, etc. **No torch line** |
| `pyproject.toml` | Both | Package metadata, portable `pillow` core, optional extras |

## Mode A — ULVO-managed (primary developer)

```powershell
cd E:\Workspace\Active\CrossModalLab
ulvo project inspect .
ulvo run . -- python -m unittest discover -s tests
ulvo run . -- python scripts\preflight_hardware_check.py
```

ULVO resolves CPython ≥3.11 and installs from `requirements.lock`. Do **not** create a project-local `.venv` on this machine.

## Mode B — Standard Python (any contributor)

### B1. CPU-only symbolic / dataset generation (no GPU required)

```bash
git clone https://github.com/Voidwalker-AT/CrossModalLab.git
cd CrossModalLab
python3.11 -m venv .venv          # or: uv venv --python 3.11
source .venv/bin/activate         # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements-core.txt
python -m unittest discover -s tests -v
python scripts/generate_controlled_benchmark.py --help 2>/dev/null || python -c "from src.symbolic import BenchmarkGenerator; print('symbolic ok')"
```

### B2. GPU VLM experiments (optional; hardware-gated)

Supported combinations (inferred from the primary lock; not universally certified):

| OS | Python | Torch | CUDA index | Notes |
|---|---|---|---|---|
| Windows 11 | 3.11 | 2.6.0+cu124 | `https://download.pytorch.org/whl/cu124` | Primary lab configuration |
| Linux | 3.11 | 2.6.0+cu124 | same | Expected to work; **NOT TESTED** in this repository audit |
| Any | 3.11 | CPU wheels | `https://download.pytorch.org/whl/cpu` | Symbolic OK; quantized SmolVLM2 **will not** match protocol |

```bash
python -m pip install -r requirements-core.txt
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-gpu.txt
python scripts/preflight_hardware_check.py
python scripts/check_portability.py
```

If CUDA, bitsandbytes, or bfloat16 are missing, GPU experiment entrypoints must fail with an explicit `CapabilityError` (see `src/config/experiment.py`). They must **not** silently switch models or drop quantization.

## Platform-specific caveats

- **`torch==2.6.0+cu124`** is a CUDA-local version and is **not installable** from PyPI alone. CPU contributors must never be handed this pin as a default.
- **bitsandbytes** GPU kernels are platform-specific; CPU-only installs may import but will not accelerate quantized inference.
- **Windows path length / symlink warnings**: the primary machine sets `HF_HUB_DISABLE_SYMLINKS_WARNING=1`; this is optional elsewhere.
- **Drive `C:` quarantine** applies only to the primary Windows machine. Other contributors may use normal OS caches.

## Reproducibility rules

- Do not regenerate `requirements.lock` casually.
- Do not change `configs/experiment_config.json` model revision (`482adb537c021c86670beed01cd58990d01e72e4`) without a documented protocol amendment.
- Record the environment source (`ulvo` vs `venv`/`uv`) and package versions in run metadata when publishing new evaluation artifacts.
- Identical inputs + identical config should yield identical symbolic outputs; GPU floating-point nondeterminism must be disclosed, not hidden.
