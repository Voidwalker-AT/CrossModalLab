"""Verify an extracted presentation payload using only the standard library."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["schema"] != "crossmodallab_submission_v1" or manifest["final_holdout_included"] is not False:
        raise ValueError("unsupported or unsafe package identity")
    expected = set(manifest["files"]) | {"PACKAGE_MANIFEST.json"}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
              and "__pycache__" not in p.parts}
    if expected != actual:
        raise ValueError("package inventory mismatch")
    for name, identity in manifest["files"].items():
        parts = PurePosixPath(name)
        path = (root / name).resolve()
        if parts.is_absolute() or ".." in parts.parts or not path.is_relative_to(root):
            raise ValueError("unsafe payload path")
        raw = path.read_bytes()
        if len(raw) != identity["bytes"] or hashlib.sha256(raw).hexdigest() != identity["sha256"]:
            raise ValueError("payload integrity mismatch: " + name)
    return {"status": "PASS", "files": len(manifest["files"]),
            "source_commit": manifest["source_commit"], "partner_machine": "PENDING"}


if __name__ == "__main__":
    print(json.dumps(verify(sys.argv[1] if len(sys.argv) > 1 else ".")))
