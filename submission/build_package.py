"""Create a bounded presentation archive from explicit source/evidence allowlists.

Run after the frontend build. No dependencies are installed and no model or full
dataset is opened. Each execution creates a new archive; existing files survive.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
RUN = "artifacts/training/phase_2i_b/runs/56e277e421e442b08d33b107fc934494"
FIXED = ["src/__init__.py", "src/training/__init__.py", "src/training/schema.py",
         "src/training/answer_parser.py", "configs/experiment_config.json",
         "artifacts/training/phase_2i_a/phase_2i_a_report_v1.json",
         *[RUN + "/" + f for f in ("run.json", "baseline.json", "evaluation_step_0005.json", "train_metrics_append.jsonl")],
         "artifacts/paired/phase_2c_c/paired_evaluations_v1.jsonl",
         "artifacts/probes/phase_1f/probe_reliability_summary_v1.json",
         "artifacts/probes/phase_1f/features_meta_v1.jsonl",
         "artifacts/causal/phase_2b/intervention_records_v1.jsonl",
         "artifacts/causal/phase_2b/recovery_audit_v1.json",
         "artifacts/training/phase_2i_b_pilot_20261023/acceptance.json",
         "artifacts/training/phase_2i_b_recovery/gpu_1b71a5485dff4fd1a0e8a95a442f789d.json",
         "artifacts/training/phase_2i_b_continuation_20261023/resume_validation_74fc874e242448e1b58350a72be19091.json"]
TREES = {"src/config": {".py"}, "src/symbolic": {".py"},
         "dashboard/backend": {".py"}, "dashboard/assets": {".json", ".png"},
         "dashboard/frontend/dist": {".html", ".js", ".css", ".svg"},
         "dashboard/frontend/src": {".tsx", ".ts", ".css"},
         "paper": {".md", ".html", ".json", ".csv", ".png", ".svg", ".py"},
         "submission/2026-10-28": {".md", ".json"}}
FRONTEND = ["package.json", "package-lock.json", "tsconfig.json", "vite.config.ts", "index.html", ".npmrc"]
SECRET_PATTERNS = [rb"GOCSPX-[A-Za-z0-9_-]{20,}", rb"AIza[0-9A-Za-z_-]{35}",
                   rb"gh[pousr]_[A-Za-z0-9]{30,}", rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
                   rb'"client_secret"\s*:\s*"[^"\s]{8,}"']


def payload_paths(root=ROOT):
    root = Path(root).resolve()
    names = set(FIXED)
    names.update("dashboard/frontend/" + f for f in FRONTEND)
    names.update(["dashboard/README.md", "dashboard/SECURITY_AUDIT.md", "dashboard/DEPENDENCY_PLAN.md",
                  "submission/build_package.py", "submission/verify_package.py", "submission/smoke_package.py"])
    for directory, suffixes in TREES.items():
        for p in (root / directory).rglob("*"):
            if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts:
                names.add(p.relative_to(root).as_posix())
    # Screenshots are newly observed presentation evidence, never old run artifacts.
    names.update(p.relative_to(root).as_posix() for p in (root / "artifacts/reports").glob("dashboard_submission_*.png"))
    names.update(p.relative_to(root).as_posix() for p in (root / "artifacts/reports").glob("submission_verification_*.md"))
    for name in sorted(names):
        p = (root / name).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            raise ValueError("missing or escaping allowlisted payload: " + name)
        if any(part in {".git", ".cache", "node_modules", "checkpoints", "datasets"} for part in Path(name).parts):
            raise ValueError("excluded storage category: " + name)
        yield name, p


def build(root=ROOT):
    root = Path(root).resolve()
    if not (root / "dashboard/frontend/dist/index.html").is_file():
        raise ValueError("build the approved production frontend first")
    payload = {}
    files = {}
    for name, path in payload_paths(root):
        raw = path.read_bytes()
        if any(re.search(pattern, raw) for pattern in SECRET_PATTERNS):
            raise ValueError("credential-pattern match; aborting package: " + name)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("unexpected large presentation file: " + name)
        payload[name] = raw
        files[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    start = ("CrossModalLab - October 28 university demonstration\n\n"
             "From this extracted directory run:\n"
             "  python -B dashboard/backend/server.py --port 8765\n"
             "Then open http://127.0.0.1:8765 . Stop with Ctrl+C.\n\n"
             "Only ordinary Python 3.11+ is required for this prebuilt dashboard.\n"
             "No GPU, model, ULVO, pip, npm or internet is required to run it.\n"
             "Read submission/2026-10-28/README.md for the demo script and limitations.\n"
             "Read paper/manuscript/draft.html in a browser for the preliminary paper.\n"
             "Verify: python -B submission/verify_package.py .\n"
             "Local smoke check: python -B submission/smoke_package.py .\n"
             "Physical second-machine verification remains pending.\n").encode()
    payload["START_HERE.txt"] = start
    files["START_HERE.txt"] = {"bytes": len(start), "sha256": hashlib.sha256(start).hexdigest()}
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    manifest = {"schema": "crossmodallab_submission_v1", "source_commit": git("rev-parse", "HEAD"),
                "tracked_changes_present": bool(git("diff", "HEAD", "--name-only")),
                "milestone": "2026-10-28 university demonstration and preliminary paper",
                "no_new_inference_or_training": True, "final_holdout_included": False,
                "runtime": "ordinary Python 3.11+; prebuilt frontend; no GPU/ULVO/npm needed",
                "partner_machine_verification": "PENDING", "files": files}
    out = root / ".cache/submission"
    out.mkdir(parents=True, exist_ok=True)
    archive = out / ("CrossModalLab_Oct28_" + manifest["source_commit"][:8] + "_" + uuid4().hex[:8] + ".zip")
    with ZipFile(archive, "x", compression=ZIP_DEFLATED) as z:
        for name, raw in payload.items():
            z.writestr(name, raw)
        z.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    return archive


if __name__ == "__main__":
    archive = build()
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}))
