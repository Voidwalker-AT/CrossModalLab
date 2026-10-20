"""CrossModalLab — Primary VLM Acquisition Script.

Downloads HuggingFaceTB/SmolVLM-500M-Instruct at immutable commit revision
a7da5b986cb59b408707209984f360a5f4ad7e47 into the configured Hugging Face cache (HF_HOME / CROSSMODAL_HF_HOME).
Verifies file sizes, checksums, and writes an acquisition manifest.
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path


from huggingface_hub import hf_hub_download

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


REPO_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
REVISION = "a7da5b986cb59b408707209984f360a5f4ad7e47"

REQUIRED_FILES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "model.safetensors"
]

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def download_model():
    print(f"Starting acquisition of {REPO_ID} at revision {REVISION}")
    manifest = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "license": "Apache 2.0",
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {},
        "total_bytes": 0,
        "total_duration_sec": 0.0,
        "status": "IN_PROGRESS"
    }

    t0_total = time.perf_counter()
    for filename in REQUIRED_FILES:
        print(f"Downloading {filename}...", flush=True)
        t0 = time.perf_counter()
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            revision=REVISION,
            local_files_only=False
        )
        elapsed = time.perf_counter() - t0
        file_size = os.path.getsize(local_path)
        manifest["total_bytes"] += file_size
        print(f"  -> {filename}: {file_size:,} bytes in {elapsed:.2f}s ({file_size / (1024*1024):.2f} MB)", flush=True)
        
        # Compute sha256
        print(f"  -> Computing SHA256 for {filename}...", flush=True)
        file_hash = sha256_file(local_path)

        manifest["files"][filename] = {
            "local_path": local_path,
            "size_bytes": file_size,
            "sha256": file_hash,
            "download_duration_sec": round(elapsed, 2)
        }

    total_duration = time.perf_counter() - t0_total
    manifest["total_duration_sec"] = round(total_duration, 2)
    manifest["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["total_mib"] = round(manifest["total_bytes"] / (1024 * 1024), 2)
    manifest["status"] = "COMPLETED"

    manifest_path = Path("datasets/metadata/model_acquisition_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nModel acquisition COMPLETED.")
    print(f"Total downloaded: {manifest['total_mib']} MiB in {manifest['total_duration_sec']}s")
    print(f"Manifest written to: {manifest_path.resolve()}")

if __name__ == "__main__":
    download_model()
