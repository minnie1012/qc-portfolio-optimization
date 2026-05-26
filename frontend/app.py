"""Zero-dependency HTTP server for the QC Portfolio Optimization dashboard.

Run:
    python frontend/app.py
Then open http://localhost:5050
"""

from __future__ import annotations

import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = Path(__file__).resolve().parent
TEMPLATES = FRONTEND / "templates"
STATIC = FRONTEND / "static"
INSTANCES_DIR = ROOT / "data" / "instances"
RAW_RESULTS_DIR = ROOT / "results" / "raw"
WARM_START_DIR = ROOT / "results" / "warm_start"

PORT = 5050

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}

INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _instance_size_bucket(instance_id: str) -> str:
    for tag in ("tiny", "small", "medium", "large", "n7_gap"):
        if instance_id.startswith(tag):
            return tag
    return "other"


def _instance_summary(path: Path) -> dict:
    data = _load_json(path)
    return {
        "instance_id": data["instance_id"],
        "N": data["N"],
        "K": data["K"],
        "q": data["q"],
        "tickers": data["asset_tickers"],
        "bucket": _instance_size_bucket(data["instance_id"]),
        "date_range": data.get("date_range"),
    }


def _iter_result_files():
    for d in (RAW_RESULTS_DIR, WARM_START_DIR):
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            yield f


def _result_summary(data: dict) -> dict:
    return {
        "algorithm": data["algorithm"],
        "instance_id": data["instance_id"],
        "seed": data.get("seed"),
        "objective_value": data.get("objective_value"),
        "feasible": data.get("feasible"),
        "approx_ratio": data.get("approx_ratio"),
        "wall_time_seconds": data.get("wall_time_seconds"),
        "qubit_count": data.get("qubit_count"),
        "circuit_depth": data.get("circuit_depth"),
        "two_qubit_gate_count": data.get("two_qubit_gate_count"),
        "total_gate_count": data.get("total_gate_count"),
        "optimizer_iters": data.get("optimizer_iters"),
        "shots": data.get("shots"),
        "backend": data.get("backend"),
        "hyperparameters": data.get("hyperparameters", {}),
    }


def list_instances() -> list[dict]:
    items: list[dict] = []
    for f in sorted(INSTANCES_DIR.glob("*.json")):
        try:
            items.append(_instance_summary(f))
        except Exception:
            continue
    return items


def list_results() -> list[dict]:
    items: list[dict] = []
    for f in _iter_result_files():
        try:
            items.append(_result_summary(_load_json(f)))
        except Exception:
            continue
    return items


def get_instance(instance_id: str) -> dict | None:
    if not INSTANCE_ID_RE.match(instance_id):
        return None
    path = INSTANCES_DIR / f"{instance_id}.json"
    return _load_json(path) if path.exists() else None


def get_result(algorithm: str, instance_id: str) -> dict | None:
    if not INSTANCE_ID_RE.match(instance_id) or not INSTANCE_ID_RE.match(algorithm):
        return None
    for d in (RAW_RESULTS_DIR, WARM_START_DIR):
        for f in d.glob(f"{algorithm}__{instance_id}__*.json"):
            return _load_json(f)
    return None


def results_for(instance_id: str) -> list[dict]:
    if not INSTANCE_ID_RE.match(instance_id):
        return []
    items: list[dict] = []
    for f in _iter_result_files():
        if f"__{instance_id}__" not in f.name:
            continue
        try:
            items.append(_load_json(f))
        except Exception:
            continue
    return items


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter logs
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_404(self) -> None:
        self._send_json({"error": "not found"}, status=404)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._send_404()
            return
        body = path.read_bytes()
        ctype = CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _render_index(self) -> None:
        # Trivial template: strip `{{ url_for(...) }}` shims since we serve static
        # via the same origin under /static/.
        html = (TEMPLATES / "index.html").read_text()
        html = re.sub(
            r"\{\{\s*url_for\('static',\s*filename='([^']+)'\)\s*\}\}",
            r"/static/\1",
            html,
        )
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            return self._render_index()

        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            target = (STATIC / rel).resolve()
            try:
                target.relative_to(STATIC.resolve())
            except ValueError:
                return self._send_404()
            return self._send_file(target)

        if path == "/api/instances":
            return self._send_json(list_instances())

        if path == "/api/results":
            return self._send_json(list_results())

        m = re.match(r"^/api/instance/([A-Za-z0-9_]+)$", path)
        if m:
            data = get_instance(m.group(1))
            return self._send_json(data) if data else self._send_404()

        m = re.match(r"^/api/results_for/([A-Za-z0-9_]+)$", path)
        if m:
            return self._send_json(results_for(m.group(1)))

        m = re.match(r"^/api/result/([A-Za-z0-9_]+)/([A-Za-z0-9_]+)$", path)
        if m:
            data = get_result(m.group(1), m.group(2))
            return self._send_json(data) if data else self._send_404()

        self._send_404()


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Dashboard ready at http://127.0.0.1:{PORT}")
    print(f"  instances : {INSTANCES_DIR}")
    print(f"  results   : {RAW_RESULTS_DIR}, {WARM_START_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
