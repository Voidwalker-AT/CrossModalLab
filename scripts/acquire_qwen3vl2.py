#!/usr/bin/env python3
"""Download pinned Qwen3-VL-2B-Instruct into the CrossModalLab HF cache (E:)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "E:/cache/huggingface")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.paths import apply_default_cache_environment, get_hf_home
from src.instrumentation.capture_integrity import atomic_write_json

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def main() -> int:
    apply_default_cache_environment()
    cache_dir = get_hf_home()
    print(f"HF_HOME={cache_dir}")
    print(f"Downloading {MODEL_ID} @ {REVISION}")
    from huggingface_hub import snapshot_download

    t0 = time.perf_counter()
    path = snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        cache_dir=str(cache_dir),
        local_files_only=False,
    )
    dur = time.perf_counter() - t0
    snap = Path(path)
    files = []
    total = 0
    import hashlib

    for f in sorted(snap.rglob("*")):
        if f.is_file():
            sz = f.stat().st_size
            total += sz
            h = hashlib.sha256(f.read_bytes()).hexdigest() if sz < 50 * 1024 * 1024 else "skipped_large"
            if f.suffix in {".safetensors"}:
                # hash large file in chunks
                hh = hashlib.sha256()
                with f.open("rb") as fh:
                    while chunk := fh.read(8 * 1024 * 1024):
                        hh.update(chunk)
                h = hh.hexdigest()
            files.append({"name": f.name, "size_bytes": sz, "sha256": h})

    required = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "chat_template.json",
        "generation_config.json",
    ]
    missing = [r for r in required if not any(x["name"] == r for x in files)]
    manifest = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "license": "apache-2.0",
        "cache_dir": str(cache_dir),
        "snapshot_path": str(snap),
        "download_duration_sec": round(dur, 2),
        "total_bytes": total,
        "total_mib": round(total / (1024 * 1024), 2),
        "n_files": len(files),
        "files": files,
        "missing_required": missing,
        "status": "COMPLETED" if not missing else "INCOMPLETE",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = REPO_ROOT / "datasets/metadata/qwen3vl2_acquisition_manifest.json"
    atomic_write_json(out, manifest)
    print(f"status={manifest['status']} total_mib={manifest['total_mib']} dur={dur:.1f}s missing={missing}")
    print(f"manifest={out}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
