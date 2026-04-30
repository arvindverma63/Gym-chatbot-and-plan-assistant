"""HTTP server for the compact gym chatbot and recommendation API."""

from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from assistant_core import FitPaxAssistant


HOST = "127.0.0.1"
PORT = 8000

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_HTML = STATIC_DIR / "index.html"
STYLES_CSS = STATIC_DIR / "styles.css"
APP_JS = STATIC_DIR / "app.js"

assistant = FitPaxAssistant(BASE_DIR)


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status_code: int, content_type: str, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_file(self, path: Path) -> None:
        if not path.exists():
            self._send_json(404, {"error": "not_found"})
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_text(200, "text/html", _read_text_file(INDEX_HTML))
            return
        if parsed.path == "/styles.css":
            self._send_text(200, "text/css", _read_text_file(STYLES_CSS))
            return
        if parsed.path == "/app.js":
            self._send_text(200, "application/javascript", _read_text_file(APP_JS))
            return
        if parsed.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if parsed.path == "/options":
            self._send_json(200, assistant.form_options())
            return
        if parsed.path.startswith("/exercise-gif/"):
            exercise_id = parsed.path.rsplit("/", 1)[-1]
            gif_path = assistant.resolve_exercise_gif(exercise_id)
            if gif_path:
                self._send_file(gif_path)
            else:
                self._send_json(404, {"error": "gif_not_found"})
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/chat", "/recommend", "/retrain", "/feedback"}:
            self._send_json(404, {"error": "not_found"})
            return

        if self.path == "/retrain":
            self._send_json(200, assistant.retrain())
            return

        if self.path == "/feedback":
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            try:
                payload = json.loads(raw.decode("utf-8"))
                result = assistant.record_feedback(payload)
                status = 200 if result.get("ok", False) else 400
                self._send_json(status, result)
            except json.JSONDecodeError:
                self._send_json(400, {"ok": False, "error": "invalid_json"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            if self.path == "/recommend":
                result = assistant.recommend(payload)
            else:
                result = assistant.chat(payload)

            status = 200 if result.get("ok", False) else 400
            self._send_json(status, result)
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "invalid_json"})


def main() -> None:
    if not INDEX_HTML.exists():
        raise FileNotFoundError("Missing static/index.html")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Gym AI server running at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
