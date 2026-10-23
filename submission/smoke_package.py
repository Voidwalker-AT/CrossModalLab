"""Local package startup check; physical second-machine evidence remains pending."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen


def smoke(root):
    root = Path(root).resolve()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [sys.executable, "-B", "dashboard/backend/server.py", "--port", str(port)]
    proc = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        base = "http://127.0.0.1:" + str(port)
        for _ in range(80):
            if proc.poll() is not None:
                raise RuntimeError("server exited: " + proc.stderr.read().decode(errors="replace"))
            try:
                with urlopen(base + "/api/health", timeout=1) as response:
                    health = json.load(response)
                break
            except OSError:
                time.sleep(.1)
        else:
            raise RuntimeError("packaged startup timed out")
        assert health["read_only"] is True and health["gpu_loaded"] is False
        with urlopen(base + "/api/workspace") as response:
            workspace = json.load(response)
        assert len(workspace["cases"]) == 18 and workspace["holdout_used"] is False
        from urllib.parse import quote
        for case in workspace["cases"]:
            with urlopen(base + "/api/image/" + quote(case["sample_id"])) as response:
                assert response.read().startswith(b"\x89PNG\r\n\x1a\n")
        with urlopen(base) as response:
            assert b'<div id="root">' in response.read()
        return {"status": "PASS", "python": sys.version, "command": command,
                "cases": 18, "images": 18, "health": health,
                "partner_machine_verification": "PENDING; local rehearsal only"}
    finally:
        proc.terminate()  # Only the child created here; never an existing listener.
        proc.communicate(timeout=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    print(json.dumps(smoke(parser.parse_args().root)))
