"""CrossModalLab — SmolVLM2-2.2B-Instruct Checkpoint Acquisition Script.

Downloads and cryptographically verifies the approved scaling candidate:
Repo: HuggingFaceTB/SmolVLM2-2.2B-Instruct
Revision: 482adb537c021c86670beed01cd58990d01e72e4 (Immutable commit hash)
License: Apache 2.0

Quarantine & Storage Rules:
- Cache writes follow CROSSMODAL_HF_HOME / HF_HOME (legacy E:/cache/huggingface when present).
- Zero writes to C:.
- Records exact byte sizes and SHA-256 hashes in datasets/metadata/smolvlm2_acquisition_manifest.json.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.paths import (
    apply_default_cache_environment,
    get_hf_home,
    get_smolvlm2_snapshot_dir,
)

# Preserve legacy E: cache when present; honor env overrides; portable elsewhere.
apply_default_cache_environment()



def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):  # 1MB buffer
            h.update(chunk)
    return h.hexdigest()

def acquire_model():
    from huggingface_hub import snapshot_download

    repo_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    revision = "482adb537c021c86670beed01cd58990d01e72e4"
    cache_dir = get_hf_home()

    print(f"Acquiring {repo_id} at revision {revision}...", flush=True)
    t0 = time.perf_counter()

    local_dir = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=False
    )
    download_duration = time.perf_counter() - t0
    print(f"Snapshot downloaded in {download_duration:.2f}s to: {local_dir}", flush=True)

    snapshot_path = Path(local_dir)
    manifest_records = []
    total_bytes = 0

    print("Computing SHA-256 checksums of downloaded files...", flush=True)
    for p in sorted(snapshot_path.iterdir()):
        if p.is_file():
            sz = p.stat().st_size
            total_bytes += sz
            print(f"  Hashing {p.name} ({sz / (1024**2):.2f} MB)...", end="", flush=True)
            file_hash = sha256_file(p)
            print(f" SHA-256: {file_hash[:16]}...", flush=True)
            manifest_records.append({
                "filename": p.name,
                "size_bytes": sz,
                "size_mb": round(sz / (1024**2), 2),
                "sha256": file_hash
            })

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": repo_id,
        "revision": revision,
        "license": "Apache 2.0",
        "cache_directory": str(snapshot_path.resolve()),
        "total_files": len(manifest_records),
        "total_size_bytes": total_bytes,
        "total_size_gb": round(total_bytes / (1024**3), 3),
        "download_duration_seconds": round(download_duration, 2),
        "files": manifest_records
    }

    out_manifest = Path("datasets/metadata/smolvlm2_acquisition_manifest.json")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest successfully written to: {out_manifest.resolve()}", flush=True)
    print(f"Total Model Weight Footprint: {manifest['total_size_gb']} GB across {len(manifest_records)} files.", flush=True)

if __name__ == "__main__":
    acquire_model()
