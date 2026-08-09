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

import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import math
import socket
import threading
from urllib.parse import urlsplit

from scripts.logger import logger

from .attempts import AttemptKey, AttemptLedger, AttemptResult

_INFERENCE_PATHS = {
    "/v1/chat/completions",
    "/v1/completions",
}
_READ_ONLY_PATHS = {"/v1/models"}
_MAX_CONTROL_BODY_BYTES = 64 * 1024
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
_AGENT_ONLY_HEADERS = {
    "x-attempt-seq",
    "x-incarnation-id",
    "x-internal-token",
    "x-remaining-deadline-ms",
    "x-request-uid",
}


def _positive_finite(value: object) -> bool:
    if type(value) not in (int, float) or value <= 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def valid_internal_token(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 4096
        and value.isascii()
        and all("!" <= character <= "~" for character in value)
    )


class _CancelableHttpConnection:
    def __init__(self, connection: http.client.HTTPConnection) -> None:
        self._connection = connection
        self._lock = threading.Lock()
        self._cancelled = False

    @property
    def connection(self) -> http.client.HTTPConnection:
        return self._connection

    def request(
        self, method: str, path: str, body: bytes | None, headers: dict[str, str]
    ) -> bool:
        with self._lock:
            if self._cancelled:
                return False
            self._connection.request(method, path, body=body, headers=headers)
            return True

    def close(self) -> None:
        with self._lock:
            self._cancelled = True
            connection_socket = self._connection.sock
            if connection_socket is not None:
                try:
                    connection_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            self._connection.close()


class _ByteCapacity:
    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._used = 0
        self._lock = threading.Lock()

    def try_reserve(self, size: int) -> bool:
        with self._lock:
            if size > self._capacity - self._used:
                return False
            self._used += size
            return True

    def release(self, size: int) -> None:
        with self._lock:
            self._used -= size


class AgentRuntime:
    def __init__(
        self,
        listen_address: str,
        upstream_url: str,
        max_attempt_records: int = 8192,
        max_cancel_fences: int = 8192,
        terminal_ttl_seconds: float = 60.0,
        negative_fence_ttl_seconds: float = 60.0,
        max_inflight_requests: int = 256,
        max_request_body_bytes: int = 8 * 1024 * 1024,
        inflight_body_capacity_bytes: int = 64 * 1024 * 1024,
        connect_timeout_seconds: float = 1.0,
        ingress_timeout_seconds: float = 5.0,
        internal_token: str = "",
    ) -> None:
        host, port = split_address(listen_address)
        if (
            type(max_inflight_requests) is not int
            or max_inflight_requests <= 0
            or type(max_request_body_bytes) is not int
            or max_request_body_bytes <= 0
            or type(inflight_body_capacity_bytes) is not int
            or inflight_body_capacity_bytes < max_request_body_bytes
            or not _positive_finite(connect_timeout_seconds)
            or not _positive_finite(ingress_timeout_seconds)
            or not valid_internal_token(internal_token)
        ):
            raise ValueError(
                "Agent limits and timeouts must be positive and "
                "internal_token must be a non-empty printable ASCII secret"
            )
        self._listen_address = listen_address
        upstream = urlsplit(upstream_url)
        if (
            upstream.scheme not in ("http", "https")
            or upstream.hostname is None
            or upstream.username is not None
            or upstream.password is not None
            or upstream.query
            or upstream.fragment
        ):
            raise ValueError("upstream URL must be an HTTP(S) base URL")
        self._upstream_scheme = upstream.scheme
        self._upstream_host = upstream.hostname
        self._upstream_port = upstream.port or (
            443 if upstream.scheme == "https" else 80
        )
        self._upstream_path = upstream.path.rstrip("/")
        self._connect_timeout = connect_timeout_seconds
        self._ingress_timeout = ingress_timeout_seconds
        self._internal_token = internal_token
        self._inflight = threading.BoundedSemaphore(max_inflight_requests)
        self._max_request_body_bytes = max_request_body_bytes
        self._body_capacity = _ByteCapacity(inflight_body_capacity_bytes)
        self._ledger = AttemptLedger(
            max_attempt_records,
            terminal_ttl_seconds,
            max_cancel_fences,
            negative_fence_ttl_seconds,
        )
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
        if self._thread is not None:
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

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        runtime = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(runtime._ingress_timeout)

            def do_GET(self) -> None:
                runtime._handle(self)

            def do_POST(self) -> None:
                runtime._handle(self)

            def do_DELETE(self) -> None:
                runtime._handle(self)

            def log_message(self, message: str, *arguments: object) -> None:
                logger.debug("agent ingress: " + message, *arguments)

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
        if not hmac.compare_digest(
            handler.headers.get("X-Internal-Token", ""), self._internal_token
        ):
            self._write_json(handler, 401, {"error": "unauthorized"})
            return
        if path == "/v1/internal/attempt/query":
            if handler.command != "POST":
                self._write_json(handler, 405, {"error": "method not allowed"})
                return
            self._attempt_control(handler, cancel=False)
            return
        if path == "/v1/internal/attempt/cancel":
            if handler.command != "POST":
                self._write_json(handler, 405, {"error": "method not allowed"})
                return
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
        if not self._inflight.acquire(blocking=False):
            self._write_json(handler, 503, {"error": "Agent concurrency limit"})
            return
        try:
            self._proxy(handler, path, None, None, None, 0)
        finally:
            self._inflight.release()

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
        if cancel:
            result = self._ledger.cancel(
                request_uid, attempt_seq, incarnation_id
            )
        else:
            result = self._ledger.query(request_uid, attempt_seq, incarnation_id)
        if result.reason == "ADMISSION_REASON_INVALID_REQUEST":
            self._write_json(handler, 400, {"error": "invalid attempt identity"})
            return
        if result.reason == "ADMISSION_REASON_STALE_INCARNATION":
            self._write_json(handler, 409, {"error": "stale incarnation"})
            return
        if result.reason == "ADMISSION_REASON_CANCEL_FENCE_CAPACITY":
            self._write_json(
                handler,
                503,
                _attempt_json(
                    result,
                    AttemptKey(request_uid, attempt_seq),
                    incarnation_id,
                ),
            )
            return
        self._write_json(
            handler,
            200,
            _attempt_json(
                result,
                AttemptKey(request_uid, attempt_seq),
                incarnation_id,
            ),
        )

    def _inference(self, handler: BaseHTTPRequestHandler, path: str) -> None:
        if handler.command != "POST":
            self._write_json(handler, 405, {"error": "method not allowed"})
            return
        if not self._inflight.acquire(blocking=False):
            self._write_json(handler, 503, {"error": "Agent concurrency limit"})
            return
        try:
            request_uid = handler.headers.get("X-Request-UID", "")
            incarnation_id = handler.headers.get("X-Incarnation-ID", "")
            try:
                attempt_seq = int(handler.headers.get("X-Attempt-Seq", ""))
                remaining_ms = int(
                    handler.headers.get("X-Remaining-Deadline-Ms", "")
                )
            except ValueError:
                self._write_json(
                    handler, 400, {"error": "invalid V2 attempt headers"}
                )
                return
            body_length = self._body_length(
                handler, self._max_request_body_bytes
            )
            if body_length is None:
                return
            if not self._body_capacity.try_reserve(body_length):
                self._write_json(
                    handler, 503, {"error": "Agent body capacity limit"}
                )
                return
            try:
                result = self._ledger.begin(
                    request_uid, attempt_seq, incarnation_id, remaining_ms
                )
                if not result.accepted:
                    if result.reason == "ADMISSION_REASON_INVALID_REQUEST":
                        status = 400
                    elif (
                        result.replayed
                        or result.reason == "ADMISSION_REASON_STALE_INCARNATION"
                    ):
                        status = 409
                    else:
                        status = 503
                    self._write_json(
                        handler,
                        status,
                        _attempt_json(
                            result,
                            AttemptKey(request_uid, attempt_seq),
                            incarnation_id,
                        ),
                    )
                    return
                key = AttemptKey(request_uid, attempt_seq)
                self._proxy(
                    handler,
                    path,
                    key,
                    incarnation_id,
                    remaining_ms,
                    self._max_request_body_bytes,
                )
            finally:
                self._body_capacity.release(body_length)
        finally:
            self._inflight.release()

    def _proxy(
        self,
        handler: BaseHTTPRequestHandler,
        path: str,
        key: AttemptKey | None,
        incarnation_id: str | None,
        remaining_ms: int | None,
        max_body_bytes: int,
    ) -> None:
        raw_body = self._read_body(handler, max_body_bytes)
        if raw_body is None:
            if key is not None:
                self._ledger.finish(
                    key,
                    incarnation_id or "",
                    "ATTEMPT_LIFECYCLE_STATE_FAILED",
                    "ADMISSION_REASON_INVALID_REQUEST",
                )
            return
        body = _inject_request_id(raw_body, path, key)
        if body is None:
            if key is not None:
                self._ledger.finish(
                    key,
                    incarnation_id or "",
                    "ATTEMPT_LIFECYCLE_STATE_FAILED",
                    "ADMISSION_REASON_INVALID_REQUEST",
                )
            self._write_json(handler, 400, {"error": "invalid inference JSON"})
            return
        headers = {
            name: value
            for name, value in handler.headers.items()
            if name.lower() not in _HOP_HEADERS
            and name.lower() not in _AGENT_ONLY_HEADERS
            and name.lower() not in ("host", "content-length")
        }
        if body:
            headers["Content-Length"] = str(len(body))
        headers["Accept-Encoding"] = "identity"
        read_timeout = max(0.001, (remaining_ms or 30000) / 1000.0)
        upstream: _CancelableHttpConnection | None = None
        try:
            connection_type = (
                http.client.HTTPSConnection
                if self._upstream_scheme == "https"
                else http.client.HTTPConnection
            )
            connection = connection_type(
                self._upstream_host,
                self._upstream_port,
                timeout=min(self._connect_timeout, read_timeout),
            )
            connection.connect()
            if connection.sock is not None:
                connection.sock.settimeout(read_timeout)
            upstream = _CancelableHttpConnection(connection)
            if key is not None and not self._ledger.attach(
                key, incarnation_id or "", upstream
            ):
                upstream.close()
                self._write_json(handler, 409, {"error": "attempt was cancelled"})
                return
            if not upstream.request(
                handler.command,
                self._upstream_request_target(handler.path),
                body if body else None,
                headers,
            ):
                self._write_json(handler, 409, {"error": "attempt was cancelled"})
                return
            response = connection.getresponse()
        except (OSError, http.client.HTTPException) as error:
            if upstream is not None:
                upstream.close()
            if key is not None:
                self._ledger.reap_expired()
                current = self._ledger.query(
                    key.request_uid, key.attempt_seq, incarnation_id or ""
                )
                if current.state == "ATTEMPT_LIFECYCLE_STATE_EXPIRED":
                    self._write_json(
                        handler,
                        504,
                        _attempt_json(
                            current, key, incarnation_id or ""
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
                            current, key, incarnation_id or ""
                        ),
                    )
                    return
                self._ledger.finish(
                    key,
                    incarnation_id or "",
                    "ATTEMPT_LIFECYCLE_STATE_FAILED",
                    "ADMISSION_REASON_INTERNAL_ERROR",
                )
            self._write_json(handler, 502, {"error": f"upstream failed: {error}"})
            return
        try:
            handler.send_response(response.status)
            for name, value in response.getheaders():
                if name.lower() not in _HOP_HEADERS and name.lower() not in (
                    "content-length",
                    "content-encoding",
                ):
                    handler.send_header(name, value)
            handler.send_header("Connection", "close")
            handler.end_headers()
            while True:
                chunk = response.read(16384)
                if not chunk:
                    break
                handler.wfile.write(chunk)
                handler.wfile.flush()
            if key is not None:
                if 200 <= response.status < 300:
                    self._ledger.finish(
                        key,
                        incarnation_id or "",
                        "ATTEMPT_LIFECYCLE_STATE_DONE",
                        "ADMISSION_REASON_ATTEMPT_TERMINAL",
                    )
                else:
                    self._ledger.finish(
                        key,
                        incarnation_id or "",
                        "ATTEMPT_LIFECYCLE_STATE_FAILED",
                        "ADMISSION_REASON_INTERNAL_ERROR",
                    )
        except (OSError, ValueError, http.client.HTTPException):
            if key is not None:
                self._ledger.cancel(
                    key.request_uid, key.attempt_seq, incarnation_id or ""
                )
        finally:
            response.close()
            if upstream is not None:
                upstream.close()
            handler.close_connection = True

    def _upstream_request_target(self, request_target: str) -> str:
        parsed = urlsplit(request_target)
        path = parsed.path or "/"
        result = self._upstream_path + path
        if parsed.query:
            result += "?" + parsed.query
        return result

    @staticmethod
    def _body_length(
        handler: BaseHTTPRequestHandler, max_body_bytes: int
    ) -> int | None:
        try:
            length = int(handler.headers.get("Content-Length", "0"))
        except ValueError:
            AgentRuntime._write_json(handler, 400, {"error": "invalid content length"})
            return None
        if length < 0 or length > max_body_bytes:
            AgentRuntime._write_json(handler, 413, {"error": "request body too large"})
            return None
        return length

    @staticmethod
    def _read_body(
        handler: BaseHTTPRequestHandler, max_body_bytes: int
    ) -> bytes | None:
        length = AgentRuntime._body_length(handler, max_body_bytes)
        if length is None:
            return None
        if not length:
            return b""
        try:
            body = handler.rfile.read(length)
        except OSError:
            try:
                AgentRuntime._write_json(
                    handler, 408, {"error": "request body timeout"}
                )
            except OSError:
                handler.close_connection = True
            return None
        if len(body) != length:
            AgentRuntime._write_json(handler, 400, {"error": "incomplete body"})
            return None
        return body

    @staticmethod
    def _read_json(handler: BaseHTTPRequestHandler) -> dict | None:
        raw = AgentRuntime._read_body(handler, _MAX_CONTROL_BODY_BYTES)
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


def split_address(address: str) -> tuple[str, int]:
    host, separator, port_text = address.rpartition(":")
    if (
        not separator
        or not host
        or host != host.strip()
        or any(character.isspace() for character in host)
        or "/" in host
        or ":" in host
        or "[" in host
        or "]" in host
        or not port_text.isascii()
        or not port_text.isdigit()
    ):
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
) -> bytes | None:
    if key is None or path not in _INFERENCE_PATHS:
        return body
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    payload["request_id"] = f"xllm-{key.request_uid}-{key.attempt_seq}"
    return json.dumps(
        payload, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
