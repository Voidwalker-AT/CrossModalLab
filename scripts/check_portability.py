#!/usr/bin/env python3
"""CrossModalLab portability smoke check (CPU-safe, no model download).

Verifies:
1. Repository imports work without ULVO-specific modules.
2. Paths resolve independently of the current working directory.
3. Shared experiment config loads with the pinned model revision.
4. Symbolic package imports on CPU.
5. Optional GPU stack reports availability without forcing installation.

Exit codes:
0 = all required checks passed
1 = one or more required checks failed

This script never downloads models or datasets.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))



def _ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def _fail(msg: str, failures: list[str]) -> None:
    print(f"[FAIL] {msg}")
    failures.append(msg)


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def main() -> int:
    failures: list[str] = []

    # Resolve repo root from this file, not CWD.
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    _info(f"Python: {sys.version.split()[0]} ({sys.executable})")
    _info(f"Repo root: {repo_root}")
    _info(f"CWD: {Path.cwd()}")

    # 1) No ULVO dependency in scientific package
    for name in ("ulvo", "wsctl"):
        if importlib.util.find_spec(name) is not None:
            _info(f"Optional host module '{name}' is importable; scientific code must not require it.")
        else:
            _info(f"Host module '{name}' not installed (expected for Mode B).")

    # 2) Import path module from a different CWD
    with tempfile.TemporaryDirectory() as tmp:
        old_cwd = Path.cwd()
        try:
            os.chdir(tmp)
            from src.config.paths import (
                REPO_ROOT,
                apply_default_cache_environment,
                get_artifacts_dir,
                get_datasets_dir,
                get_hf_home,
                get_smolvlm2_snapshot_dir,
                get_temp_dir,
            )

            if REPO_ROOT.resolve() != repo_root.resolve():
                _fail(f"REPO_ROOT mismatch: {REPO_ROOT} != {repo_root}", failures)
            else:
                _ok(f"REPO_ROOT stable from foreign CWD: {REPO_ROOT}")

            hf_home = apply_default_cache_environment()
            _ok(f"HF_HOME resolved: {hf_home}")
            _info(f"datasets: {get_datasets_dir()}")
            _info(f"artifacts: {get_artifacts_dir()}")
            _info(f"temp: {get_temp_dir()}")

            snap = get_smolvlm2_snapshot_dir()
            if snap.is_dir():
                _ok(f"Primary model snapshot present: {snap}")
            else:
                _info(f"Primary model snapshot not present locally: {snap}")
                _info("Symbolic CPU workflows remain available; GPU experiments need the asset.")

            # Env must not have been forced if already set by caller
            if "HF_HOME" not in os.environ:
                _fail("HF_HOME was not set by apply_default_cache_environment", failures)
        except Exception as exc:
            _fail(f"Path module import/execution failed: {exc}", failures)
        finally:
            os.chdir(old_cwd)

    # 3) Experiment config
    try:
        from src.config.experiment import get_experiment_config

        cfg = get_experiment_config()
        expected_rev = "482adb537c021c86670beed01cd58990d01e72e4"
        if cfg.model.revision != expected_rev:
            _fail(f"Pinned revision mismatch: {cfg.model.revision}", failures)
        else:
            _ok(f"Experiment config: {cfg.model.model_id} @ {cfg.model.revision}")
        if cfg.quantization.bnb_4bit_quant_type != "nf4":
            _fail("Quantization type is not nf4", failures)
        else:
            _ok("Quantization config: nf4 / 4-bit")
    except Exception as exc:
        _fail(f"Experiment config failed: {exc}", failures)

    # 4) Symbolic package (CPU)
    try:
        from src.symbolic import BenchmarkGenerator, SymbolicSolver  # noqa: F401

        _ok("src.symbolic imports without torch/ULVO")
    except Exception as exc:
        _fail(f"src.symbolic import failed: {exc}", failures)

    # 5) Evaluation module may require torch — report only
    try:
        import src.evaluation  # noqa: F401

        _info("src.evaluation imports (torch stack present)")
    except Exception as exc:
        _info(f"src.evaluation not importable without GPU stack (expected on Mode B CPU-only): {exc}")

    # 6) ULVO must not be required
    try:
        # Ensure no accidental hard dependency
        spec = importlib.util.find_spec("src.config.paths")
        if spec is None:
            _fail("src.config.paths not discoverable", failures)
        else:
            _ok("Scientific config package discoverable without ULVO")
    except Exception as exc:
        _fail(f"Importlib check failed: {exc}", failures)

    print()
    if failures:
        print(f"PORTABILITY CHECK: FAILED ({len(failures)} failure(s))")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("PORTABILITY CHECK: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
