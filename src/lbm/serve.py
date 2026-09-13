"""HTTP server that runs :class:`lbm.policy.LBMPolicy` for sim eval clients."""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from lbm.policy_codec import decode_obs, encode_array

_log = logging.getLogger(__name__)


class PolicyHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, policy: Any):
        self.policy = policy
        self.policy_lock = threading.Lock()
        super().__init__(server_address, PolicyRequestHandler)


class PolicyRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        _log.info("%s - " + fmt, self.address_string(), *args)

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        try:
            if path in ("/", "/health"):
                self._send_json(200, {"ok": True})
                return
            if path == "/metadata":
                meta = self.server.policy.metadata() if hasattr(self.server.policy, "metadata") else {}
                self._send_json(200, meta)
                return
            self._send_json(404, {"error": f"unknown path {path}"})
        except Exception as exc:
            _log.exception("GET %s failed", path)
            self._send_json(500, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"invalid json: {exc}"})
            return
        if path == "/reset":
            with self.server.policy_lock:
                self.server.policy.reset()
            self._send_json(200, {"ok": True})
            return
        if path != "/infer":
            self._send_json(404, {"error": f"unknown path {path}"})
            return
        try:
            obs = decode_obs(payload)
            # One policy owns one temporal context; concurrent HTTP requests
            # must not mutate its observation buffers while inference is using them.
            with self.server.policy_lock:
                actions = self.server.policy.infer(obs)
            self._send_json(200, {"actions": encode_array(actions)})
        except Exception as exc:
            _log.exception("infer failed")
            self._send_json(500, {"error": str(exc)})


def serve_policy(policy: Any, *, host: str = "0.0.0.0", port: int = 8000) -> None:
    server = PolicyHTTPServer((host, port), policy)
    _log.info("LBM policy server on http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log.info("shutting down")
    finally:
        server.server_close()
