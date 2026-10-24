"""Serve the built dashboard and allowlisted artifact API on loopback only."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dashboard.backend.artifacts import ArtifactStore


def make_server(port=8765, *, store=None):
    store = store or ArtifactStore()
    static = ROOT / "dashboard/frontend/dist"
    class Handler(BaseHTTPRequestHandler):
        def send(self, code, data, content_type="application/json; charset=utf-8"):
            raw = json.dumps(data, allow_nan=False, ensure_ascii=False).encode() if not isinstance(data, bytes) else data
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.headers.get("Host", "").split(":")[0] not in {"127.0.0.1", "localhost"}:
                return self.send(403, {"error": "loopback host required"})
            path = unquote(urlsplit(self.path).path)
            try:
                if path == "/api/health":
                    return self.send(200, {"status": "ok", "read_only": True, "gpu_loaded": False})
                if path == "/api/workspace":
                    return self.send(200, store.workspace())
                if path == "/api/provenance":
                    return self.send(200, store.provenance())
                if path.startswith("/api/comparison/"):
                    from dashboard.backend.exports import comparison_svg
                    sample = path[len("/api/comparison/"):]
                    if not sample.endswith(".svg") or sample[:-4] not in store.cases:
                        return self.send(404, {"error": "comparison outside development allowlist"})
                    return self.send(200, comparison_svg(store, sample[:-4]), "image/svg+xml; charset=utf-8")
                if path.startswith("/api/image/"):
                    return self.send(200, store.image(path[len('/api/image/'):]).read_bytes(), "image/png")
                if path.startswith("/api/"):
                    return self.send(404, {"error": "unknown evidence endpoint"})
                target = (static / (path.lstrip("/") or "index.html")).resolve()
                if not target.is_relative_to(static.resolve()):
                    return self.send(404, {"error": "not found"})
                if not target.is_file():
                    return self.send(404, {"error": "frontend not built or resource unavailable; see dashboard/README.md"})
                return self.send(200, target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            except KeyError:
                return self.send(404, {"error": "case is outside development allowlist"})
            except (ValueError, FileNotFoundError) as exc:
                return self.send(422, {"error": str(exc)})

        def readonly(self):
            self.send(405, {"error": "read-only dashboard; experiment and dataset mutation prohibited"})
        do_POST = do_PUT = do_PATCH = do_DELETE = readonly

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    with make_server(args.port) as server:
        print(f"CrossModalLab: http://127.0.0.1:{server.server_port} (read-only, no GPU)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
