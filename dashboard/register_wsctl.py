"""Register the reviewed project entry, preserving every unrelated registry byte.

Requires explicit --apply and an authorized registry path. Backup stays local.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def registration_bytes(raw, entry):
    bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    before = json.loads(text)
    existing = [p for p in before["projects"] if p["id"].lower() == entry["id"].lower()]
    if existing:
        if len(existing) == 1 and existing[0] == entry:
            return raw, before
        raise ValueError("Existing CrossModalLab entry differs; inspect before any replacement")
    match = re.search(r'"projects"\s*:\s*\[', text)
    if not match:
        raise ValueError("projects array not found")
    start = match.end() - 1
    projects, length = json.JSONDecoder().raw_decode(text[start:])
    end = start + length - 1
    prefix = text[:end].rstrip()
    inserted = "\n".join("    " + line for line in json.dumps(entry, indent=2, ensure_ascii=False).splitlines())
    newline = "\r\n" if "\r\n" in text else "\n"
    addition = ("," if projects else "") + "\n" + inserted + "\n  "
    candidate = prefix + addition.replace("\n", newline) + text[end:]
    after = json.loads(candidate)
    assert after["projects"][:-1] == before["projects"]
    assert {k: v for k, v in after.items() if k != "projects"} == {k: v for k, v in before.items() if k != "projects"}
    return (b"\xef\xbb\xbf" if bom else b"") + candidate.encode("utf-8"), before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    registry = args.registry.resolve()
    entry = json.loads((ROOT / "dashboard/wsctl.project.json").read_text())
    if (registry.parent / entry["path"]).resolve() != ROOT:
        raise ValueError("reviewed entry does not resolve to this checkout")
    raw = registry.read_bytes()
    candidate, before = registration_bytes(raw, entry)
    if candidate == raw:
        print("CrossModalLab already registered identically; no write")
        return
    if not args.apply:
        print(json.dumps({"mode": "PREVIEW", "entry": entry, "preserved_projects": len(before["projects"])}))
        return
    folder = ROOT / ".cache/wsctl"
    folder.mkdir(parents=True, exist_ok=True)
    backup = folder / ("workspace_registry_before_" + uuid4().hex + ".json")
    backup.write_bytes(raw)
    assert backup.read_bytes() == raw
    if registry.read_bytes() != raw:
        raise RuntimeError("Registry changed during review; refusing stale write")
    registry.write_bytes(candidate)
    assert registry.read_bytes() == candidate
    result = {"status": "PASS", "backup": str(backup), "before_sha256": hashlib.sha256(raw).hexdigest(),
              "after_sha256": hashlib.sha256(candidate).hexdigest(), "preserved_projects": len(before["projects"]),
              "added_project": entry["id"], "other_settings_unchanged": True}
    (folder / "registration_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
