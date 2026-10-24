"""Launch the artifact dashboard using ordinary Python; WSCTL/ULVO are optional."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_fingerprint(root=ROOT):
    frontend = root / "dashboard/frontend"
    paths = [frontend / n for n in ("package.json", "package-lock.json", "tsconfig.json", "vite.config.ts", "index.html")]
    paths += sorted((frontend / "src").rglob("*"))
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def ensure_frontend(root=ROOT):
    frontend = root / "dashboard/frontend"
    stamp = root / ".cache/dashboard-build.json"
    fingerprint = build_fingerprint(root)
    index = frontend / "dist/index.html"
    assets = list((frontend / "dist/assets").glob("*.js"))
    references = re.findall(r'(?:src|href)=["\x27]/(assets/[^"\x27]+)', index.read_text()) if index.is_file() else []
    referenced_assets_present = all((frontend / "dist" / name).is_file() for name in references)
    if index.is_file() and assets and referenced_assets_present and stamp.is_file():
        saved = json.loads(stamp.read_text())
        if saved.get("input_sha256") == fingerprint:
            print("Production frontend is current.", flush=True)
            return
    node, npm = shutil.which("node"), shutil.which("npm")
    if not node or not npm:
        raise RuntimeError("Frontend needs a build. Existing Node/npm are required; nothing was installed.")
    version = subprocess.check_output([node, "--version"], text=True).strip()
    major, minor, patch = map(int, version.lstrip("v").split(".")[:3])
    if not ((major == 20 and (minor, patch) >= (19, 0)) or (major == 22 and (minor, patch) >= (12, 0)) or major > 22):
        raise RuntimeError("Vite requires Node ^20.19.0 or >=22.12.0; found " + version)
    expected = json.loads((frontend / "package.json").read_text())
    for package, pinned in {**expected["dependencies"], **expected["devDependencies"]}.items():
        installed = frontend / "node_modules" / package / "package.json"
        if not installed.is_file() or json.loads(installed.read_text())["version"] != pinned:
            raise RuntimeError("Missing/mismatched locked frontend dependency: " + package +
                               ". Follow dashboard/README.md; automatic installation is prohibited.")
    env = os.environ.copy()
    cache = root / ".cache"
    temp = cache / "dashboard-temp"
    temp.mkdir(parents=True, exist_ok=True)
    env.update(TEMP=str(temp), TMP=str(temp), npm_config_cache=str(cache / "npm"), npm_config_ignore_scripts="true")
    print("Building production frontend with existing locked dependencies...", flush=True)
    subprocess.run([npm, "--prefix", str(frontend), "run", "build"], cwd=root, env=env, check=True)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps({"input_sha256": fingerprint}) + "\n")


def port_owner(port):
    if os.name != "nt":
        return "owner inspection: use your system's listener/process tool"
    script = (f"Get-NetTCPConnection -State Listen -LocalPort {int(port)} -ErrorAction SilentlyContinue | "
              "ForEach-Object { $p=Get-Process -Id $_.OwningProcess; "
              "[PSCustomObject]@{PID=$p.Id;Name=$p.ProcessName} } | ConvertTo-Json -Compress")
    result = subprocess.run(["pwsh", "-NoProfile", "-Command", script], capture_output=True, text=True)
    return result.stdout.strip() or "owner unavailable"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.environ.get("CROSSMODAL_DASHBOARD_PORT", "8765")))
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", args.port))
    except OSError:
        print(f"Port {args.port} is occupied: {port_owner(args.port)}. No process was stopped.\n"
              "Use: python scripts/launch_dashboard.py --port 8766", file=sys.stderr)
        return 2
    try:
        ensure_frontend()
        from dashboard.backend.server import make_server
        with make_server(args.port) as server:
            print(f"CrossModalLab: http://127.0.0.1:{server.server_port} (read-only; no GPU/model; Ctrl+C to stop)", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                print("\nCrossModalLab stopped cleanly.", flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print("Dashboard startup failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
