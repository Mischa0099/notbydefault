"""Tiny notes API (stdlib only). Notes live in memory — no database chosen yet."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

NOTES: list[dict] = []


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/notes":
            self._send(200, NOTES)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/notes":
            return self._send(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length", 0))
        note = {"id": len(NOTES) + 1, "text": json.loads(self.rfile.read(length) or "{}").get("text", "")}
        NOTES.append(note)
        self._send(201, note)


if __name__ == "__main__":
    print("Serving on http://localhost:8000")
    HTTPServer(("localhost", 8000), Handler).serve_forever()
