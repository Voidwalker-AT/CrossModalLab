"""Portable path and cache resolution for CrossModalLab.

Design goals:
- Resolve all repository-relative paths from this file (not the CWD).
- Prefer environment-variable overrides.
- Preserve the primary developer's existing ``E:/cache/huggingface`` model
  layout when that path exists.
- Never require ULVO, Windows drive letters, or a specific username.
- Never download models or create cache directories implicitly.

Environment variables:
- ``CROSSMODAL_HF_HOME``: Hugging Face cache root (overrides detection).
- ``HF_HOME``: Standard Hugging Face cache root (honored if set).
- ``CROSSMODAL_MODEL_SNAPSHOT_DIR``: Explicit SmolVLM2 snapshot directory.
- ``CROSSMODAL_ARTIFACTS_DIR``: Artifact output root.
- ``CROSSMODAL_DATASETS_DIR``: Dataset root.
- ``CROSSMODAL_TEMP_DIR``: Temporary directory override (C: quarantine helper).
- ``CROSSMODAL_LEGACY_E_CACHE``: ``1``/``0`` to force enable/disable the
  legacy ``E:/cache`` preference on the primary machine.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# src/config/paths.py -> src/config -> src -> repository root
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Canonical primary research model (pinned; do not change casually).
PRIMARY_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
PRIMARY_MODEL_REVISION = "482adb537c021c86670beed01cd58990d01e72e4"

# Historical early-pilot model (kept for provenance of existing artifacts).
LEGACY_MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
LEGACY_MODEL_REVISION = "a7da5b986cb59b408707209984f360a5f4ad7e47"

_LEGACY_E_HF_HOME = Path("E:/cache/huggingface")
_LEGACY_E_TEMP = Path("E:/cache/temp")


def _env_path(name: str) -> Optional[Path]:
    raw = os.environ.get(name)
    if not raw:
        return None
    return Path(raw).expanduser()


def _truthy(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _legacy_e_cache_allowed() -> bool:
    force = _truthy("CROSSMODAL_LEGACY_E_CACHE")
    if force is not None:
        return force
    # Auto-prefer legacy E: cache only when it already exists on this machine.
    return _LEGACY_E_HF_HOME.is_dir()


def get_repo_root() -> Path:
    """Return the repository root resolved from this module's location."""
    return REPO_ROOT


def get_hf_home() -> Path:
    """Resolve the Hugging Face cache root without creating directories."""
    explicit = _env_path("CROSSMODAL_HF_HOME") or _env_path("HF_HOME")
    if explicit is not None:
        return explicit

    if _legacy_e_cache_allowed() and _LEGACY_E_HF_HOME.is_dir():
        return _LEGACY_E_HF_HOME

    return Path.home() / ".cache" / "huggingface"


def get_datasets_dir() -> Path:
    return _env_path("CROSSMODAL_DATASETS_DIR") or (REPO_ROOT / "datasets")


def get_artifacts_dir() -> Path:
    return _env_path("CROSSMODAL_ARTIFACTS_DIR") or (REPO_ROOT / "artifacts")


def get_temp_dir() -> Path:
    """Resolve a temporary directory.

    On the primary machine this preserves the historical C:-quarantine
    pattern under ``E:/cache/temp`` when that tree exists (or when the
    legacy preference is forced). Elsewhere, fall back to the OS temp dir.
    """
    explicit = _env_path("CROSSMODAL_TEMP_DIR")
    if explicit is not None:
        return explicit

    if _legacy_e_cache_allowed() and (_LEGACY_E_TEMP.is_dir() or _LEGACY_E_HF_HOME.is_dir()):
        return _LEGACY_E_TEMP

    return Path(os.environ.get("TMP") or os.environ.get("TEMP") or Path.home() / "tmp")


def get_smolvlm2_snapshot_dir() -> Path:
    """Resolve the pinned SmolVLM2 snapshot directory.

    Resolution order:
    1. ``CROSSMODAL_MODEL_SNAPSHOT_DIR``
    2. ``<hf_home>/models--HuggingFaceTB--SmolVLM2-2.2B-Instruct/snapshots/<rev>``
       (layout used by the primary machine cache)
    3. ``<hf_home>/hub/models--.../snapshots/<rev>`` (standard HF hub layout)
    4. ``<hf_home>/models--.../snapshots/<rev>`` as a non-existent expected path
       (callers should check ``.exists()`` and fail with a clear message).
    """
    explicit = _env_path("CROSSMODAL_MODEL_SNAPSHOT_DIR")
    if explicit is not None:
        return explicit

    repo_name = f"models--{PRIMARY_MODEL_ID.replace('/', '--')}"
    hf_home = get_hf_home()
    candidates = [
        hf_home / repo_name / "snapshots" / PRIMARY_MODEL_REVISION,
        hf_home / "hub" / repo_name / "snapshots" / PRIMARY_MODEL_REVISION,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def apply_default_cache_environment() -> Path:
    """Apply non-destructive cache environment defaults.

    Existing environment variables are never overwritten. This preserves
    primary-machine behavior (E: cache quarantine) while remaining portable.
    Returns the resolved Hugging Face cache root.
    """
    hf_home = get_hf_home()
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    temp_dir = get_temp_dir()
    # Only redirect TEMP/TMP when an explicit or legacy-project temp path is used.
    if "CROSSMODAL_TEMP_DIR" in os.environ or (_legacy_e_cache_allowed() and temp_dir == _LEGACY_E_TEMP):
        os.environ.setdefault("TEMP", str(temp_dir))
        os.environ.setdefault("TMP", str(temp_dir))
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Never fail solely because temp redirection is unavailable.
            pass

    return hf_home


def ensure_repo_on_sys_path() -> Path:
    """Insert REPO_ROOT at the front of ``sys.path`` if missing."""
    import sys

    root_str = str(REPO_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return REPO_ROOT
