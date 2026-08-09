# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm-service/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Strict V2 ingress and attempt control for a local vLLM-Ascend runtime."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import threading
from urllib.parse import urlsplit

import requests

from .attempts import AttemptKey, AttemptLedger, AttemptResult

logger = logging.getLogger("vllm_sidecar.agent")

_INFERENCE_PATHS = {
    "/v1/chat/completions",
    "/v1/completions",
}
_READ_ONLY_PATHS = {"/v1/models"}
_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class AgentRuntime:
    def __init__(
        self,
        listen_address: str,
        upstream_url: str,
        max_attempt_records: int = 8192,
        terminal_ttl_seconds: float = 60.0,
        connect_timeout_seconds: float = 1.0,
    ) -> None:
        host, port = _split_address(listen_address)
        if connect_timeout_seconds <= 0:
            raise ValueError("connect timeout must be positive")
        self._listen_address = listen_address
        self._upstream_url = upstream_url.rstrip("/")
        self._connect_timeout = connect_timeout_seconds
        self._ledger = AttemptLedger(max_attempt_records, terminal_ttl_seconds)
        self._server = ThreadingHTTPServer((host, port), self._handler_type())
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None
        self._reaper_stop = threading.Event()
        self._reaper_thread: threading.Thread | None = None

    @property
    def listen_address(self) -> str:
        host, port = self._server.server_address[:2]
        return f"{host}:{port}"

    @property
    def ledger(self) -> AttemptLedger:
        return self._ledger

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="vllm-provider-agent",
            daemon=True,
        )
        self._reaper_thread = threading.Thread(
            target=self._reap_loop,
            name="vllm-provider-attempt-reaper",
            daemon=True,
        )
        self._thread.start()
        self._reaper_thread.start()

    def stop(self) -> None:
        self._ledger.fence()
        self._reaper_stop.set()
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._reaper_thread is not None:
            self._reaper_thread.join(timeout=5.0)
        self._thread = None
        self._reaper_thread = None

    def activate(self, incarnation_id: str) -> None:
        self._ledger.activate(incarnation_id)

    def fence(self, reason: str = "ADMISSION_REASON_ENGINE_DRAINING") -> None:
        self._ledger.fence(reason)

    def _reap_loop(self) -> None:
        while not self._reaper_stop.wait(0.05):
            self._ledger.reap_expired()

    def _handler_type(self):
        runtime = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                runtime._handle(self)

            def do_POST(self) -> None:
                runtime._handle(self)

            def do_DELETE(self) -> None:
                runtime._handle(self)

            def log_message(self, message: str, *args) -> None:
                logger.debug("agent ingress: " + message, *args)

        return Handler

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        path = urlsplit(handler.path).path
        if path == "/livez":
            self._write_json(handler, 200, {"status": "alive"})
            return
        if path in ("/health", "/readyz"):
            if self._ledger.accepting():
                self._write_json(handler, 200, {"status": "ready"})
            else:
                self._write_json(handler, 503, {"status": "fenced"})
            return
        if path == "/v1/internal/attempt/query":
            self._attempt_control(handler, cancel=False)
            return
        if path == "/v1/internal/attempt/cancel":
            self._attempt_control(handler, cancel=True)
            return
        if path in _INFERENCE_PATHS:
            self._inference(handler, path)
            return
        if path not in _READ_ONLY_PATHS:
            self._write_json(handler, 404, {"error": "unsupported Agent path"})
            return
        if handler.command != "GET":
            self._write_json(handler, 405, {"error": "method not allowed"})
            return
        if not self._ledger.accepting():
            self._write_json(handler, 503, {"error": "agent is fenced"})
            return
        self._proxy(handler, path, None, None)

    def _attempt_control(
        self, handler: BaseHTTPRequestHandler, cancel: bool
    ) -> None:
        body = self._read_json(handler)
        if body is None:
            return
        request_uid = body.get("request_uid")
        attempt_seq = body.get("attempt_seq")
        incarnation_id = body.get("incarnation_id")
        if (
            not isinstance(request_uid, str)
            or not request_uid
            or type(attempt_seq) is not int
            or attempt_seq < 0
        ):
            self._write_json(handler, 400, {"error": "invalid attempt identity"})
            return
        if incarnation_id != self._ledger.incarnation_id():
            self._write_json(handler, 409, {"error": "stale incarnation"})
            return
        if cancel:
            result = self._ledger.cancel(request_uid, attempt_seq)
        else:
            result = self._ledger.query(request_uid, attempt_seq)
        self._write_json(
            handler,
            200,
            _attempt_json(
                result,
                AttemptKey(request_uid, attempt_seq),
                self._ledger.incarnation_id(),
            ),
        )

    def _inference(self, handler: BaseHTTPRequestHandler, path: str) -> None:
        if handler.command != "POST":
            self._write_json(handler, 405, {"error": "method not allowed"})
            return
        request_uid = handler.headers.get("X-Request-UID", "")
        try:
            attempt_seq = int(handler.headers.get("X-Attempt-Seq", ""))
            remaining_ms = int(handler.headers.get("X-Remaining-Deadline-Ms", ""))
        except ValueError:
            self._write_json(handler, 400, {"error": "invalid V2 attempt headers"})
            return
        result = self._ledger.begin(request_uid, attempt_seq, remaining_ms)
        if not result.accepted:
            status = 409 if result.replayed else 503
            self._write_json(
                handler,
                status,
                _attempt_json(
                    result,
                    AttemptKey(request_uid, attempt_seq),
                    self._ledger.incarnation_id(),
                ),
            )
            return
        key = AttemptKey(request_uid, attempt_seq)
        self._proxy(handler, path, key, remaining_ms)

    def _proxy(
        self,
        handler: BaseHTTPRequestHandler,
        path: str,
        key: AttemptKey | None,
        remaining_ms: int | None,
    ) -> None:
        raw_body = self._read_body(handler)
        if raw_body is None:
            if key is not None:
                self._ledger.finish(
                    key,
                    "ATTEMPT_LIFECYCLE_STATE_FAILED",
                    "ADMISSION_REASON_INVALID_REQUEST",
                )
            return
        body = _inject_request_id(raw_body, path, key)
        headers = {
            name: value
            for name, value in handler.headers.items()
            if name.lower() not in _HOP_HEADERS
            and name.lower() not in ("host", "content-length")
        }
        if body:
            headers["Content-Length"] = str(len(body))
        headers["Accept-Encoding"] = "identity"
        read_timeout = max(0.001, (remaining_ms or 30000) / 1000.0)
        try:
            response = requests.request(
                handler.command,
                self._upstream_url + handler.path,
                headers=headers,
                data=body if body else None,
                stream=True,
                timeout=(self._connect_timeout, read_timeout),
                allow_redirects=False,
            )
        except requests.RequestException as error:
            if key is not None:
                self._ledger.reap_expired()
                current = self._ledger.query(key.request_uid, key.attempt_seq)
                if current.state == "ATTEMPT_LIFECYCLE_STATE_EXPIRED":
                    self._write_json(
                        handler,
                        504,
                        _attempt_json(
                            current, key, self._ledger.incarnation_id()
                        ),
                    )
                    return
                if current.state in (
                    "ATTEMPT_LIFECYCLE_STATE_CANCELLED",
                    "ATTEMPT_LIFECYCLE_STATE_CANCELLED_BEFORE_CREATE",
                ):
                    self._write_json(
                        handler,
                        409,
                        _attempt_json(
                            current, key, self._ledger.incarnation_id()
                        ),
                    )
                    return
                self._ledger.finish(
                    key,
                    "ATTEMPT_LIFECYCLE_STATE_FAILED",
                    "ADMISSION_REASON_INTERNAL_ERROR",
                )
            self._write_json(handler, 502, {"error": f"upstream failed: {error}"})
            return

        if key is not None and not self._ledger.attach(key, response):
            response.close()
            self._write_json(handler, 409, {"error": "attempt was cancelled"})
            return
        try:
            handler.send_response(response.status_code)
            for name, value in response.headers.items():
                if name.lower() not in _HOP_HEADERS and name.lower() not in (
                    "content-length",
                    "content-encoding",
                ):
                    handler.send_header(name, value)
            handler.send_header("Connection", "close")
            handler.end_headers()
            for chunk in response.iter_content(chunk_size=16384):
                if not chunk:
                    continue
                handler.wfile.write(chunk)
                handler.wfile.flush()
            if key is not None:
                if 200 <= response.status_code < 300:
                    self._ledger.finish(
                        key,
                        "ATTEMPT_LIFECYCLE_STATE_DONE",
                        "ADMISSION_REASON_ATTEMPT_TERMINAL",
                    )
                else:
                    self._ledger.finish(
                        key,
                        "ATTEMPT_LIFECYCLE_STATE_FAILED",
                        "ADMISSION_REASON_INTERNAL_ERROR",
                    )
        except (BrokenPipeError, ConnectionResetError, requests.RequestException):
            if key is not None:
                self._ledger.cancel(key.request_uid, key.attempt_seq)
        finally:
            response.close()
            handler.close_connection = True

    @staticmethod
    def _read_body(handler: BaseHTTPRequestHandler) -> bytes | None:
        try:
            length = int(handler.headers.get("Content-Length", "0"))
        except ValueError:
            AgentRuntime._write_json(handler, 400, {"error": "invalid content length"})
            return None
        if length < 0 or length > 64 * 1024 * 1024:
            AgentRuntime._write_json(handler, 413, {"error": "request body too large"})
            return None
        return handler.rfile.read(length) if length else b""

    @staticmethod
    def _read_json(handler: BaseHTTPRequestHandler) -> dict | None:
        raw = AgentRuntime._read_body(handler)
        if raw is None:
            return None
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            AgentRuntime._write_json(handler, 400, {"error": "invalid json"})
            return None
        if not isinstance(body, dict):
            AgentRuntime._write_json(handler, 400, {"error": "json must be an object"})
            return None
        return body

    @staticmethod
    def _write_json(
        handler: BaseHTTPRequestHandler, status: int, body: dict
    ) -> None:
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(payload)
        handler.close_connection = True


def _split_address(address: str) -> tuple[str, int]:
    host, separator, port_text = address.rpartition(":")
    if not separator or not host:
        raise ValueError("agent listen address must be host:port")
    port = int(port_text)
    if port < 0 or port > 65535:
        raise ValueError("agent listen port is out of range")
    return host, port


def _attempt_json(
    result: AttemptResult,
    key: AttemptKey | None = None,
    incarnation_id: str = "",
) -> dict:
    payload = {
        "accepted": result.accepted,
        "replayed": result.replayed,
        "state": result.state,
        "reason": result.reason,
    }
    if key is not None:
        payload.update(
            {
                "request_uid": key.request_uid,
                "attempt_seq": key.attempt_seq,
                "incarnation_id": incarnation_id,
            }
        )
    return payload


def _inject_request_id(
    body: bytes, path: str, key: AttemptKey | None
) -> bytes:
    if key is None or path not in _INFERENCE_PATHS:
        return body
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body
    if not isinstance(payload, dict):
        return body
    payload["request_id"] = f"xllm-{key.request_uid}-{key.attempt_seq}"
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")
