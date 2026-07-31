#!/usr/bin/env python3

# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import io
import threading
import unittest
from collections import defaultdict
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from retry_client import RetryPolicy
from retry_client import send_with_typed_stale_retry


class FaultServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, FaultHandler)
        self.counts: dict[str, int] = defaultdict(int)
        self.request_ids: dict[str, list[str]] = defaultdict(list)
        self.request_bodies: dict[str, list[bytes]] = defaultdict(list)


class FaultHandler(BaseHTTPRequestHandler):
    server: FaultServer

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        self.server.counts[self.path] += 1
        self.server.request_ids[self.path].append(
            self.headers.get("x-request-id", "")
        )
        self.server.request_bodies[self.path].append(body)

        if self.path == "/stale-once" and self.server.counts[self.path] == 1:
            self._send_stale()
        elif self.path == "/always-stale":
            self._send_stale()
        elif self.path == "/nonretryable":
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b"unavailable")
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"data: first-token\n\n")
            self.wfile.write(b"data: backend-error\n\n")
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

    def _send_stale(self) -> None:
        self.send_response(503)
        self.send_header("x-llm-d-retryable", "true")
        self.send_header("x-llm-d-error-code", "stale_routing_decision")
        self.send_header("Retry-After", "0")
        self.end_headers()
        self.wfile.write(b"stale")

    def log_message(self, format: str, *args: object) -> None:
        del format, args


class FakeResponse(io.BytesIO):
    def __init__(self, status: int, headers: dict[str, str]) -> None:
        super().__init__(b"response")
        self.status = status
        self.headers = Message()
        for name, value in headers.items():
            self.headers[name] = value


class GatewayRetryClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = FaultServer(("127.0.0.1", 0))
        cls.thread = threading.Thread(
            target=cls.server.serve_forever,
            daemon=True,
        )
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def test_stale_once_retries_with_same_logical_request(self) -> None:
        body = b'{"model":"test","prompt":"hello"}'
        outcome = send_with_typed_stale_retry(
            self.base_url + "/stale-once",
            body,
            {"Content-Type": "application/json"},
            RetryPolicy(max_attempts=2, total_timeout_s=2),
        )
        with outcome.response:
            self.assertEqual(outcome.response.read(), b"ok")

        self.assertEqual(outcome.attempts, 2)
        self.assertFalse(outcome.retry_budget_exhausted)
        request_ids = self.server.request_ids["/stale-once"]
        self.assertEqual(len(request_ids), 2)
        self.assertTrue(request_ids[0])
        self.assertEqual(request_ids[0], request_ids[1])
        self.assertEqual(
            self.server.request_bodies["/stale-once"],
            [body, body],
        )

    def test_retry_budget_is_bounded(self) -> None:
        outcome = send_with_typed_stale_retry(
            self.base_url + "/always-stale",
            b"{}",
            {},
            RetryPolicy(max_attempts=3, total_timeout_s=2),
        )
        self.assertEqual(outcome.response.status, 503)
        self.assertEqual(outcome.attempts, 3)
        self.assertTrue(outcome.retry_budget_exhausted)
        outcome.response.close()

    def test_plain_503_is_not_retried(self) -> None:
        outcome = send_with_typed_stale_retry(
            self.base_url + "/nonretryable",
            b"{}",
            {},
            RetryPolicy(max_attempts=3, total_timeout_s=2),
        )
        self.assertEqual(outcome.response.status, 503)
        self.assertEqual(outcome.attempts, 1)
        self.assertFalse(outcome.retry_budget_exhausted)
        outcome.response.close()

    def test_http_200_stream_is_never_retried(self) -> None:
        outcome = send_with_typed_stale_retry(
            self.base_url + "/stream",
            b"{}",
            {},
            RetryPolicy(max_attempts=3, total_timeout_s=2),
        )
        with outcome.response:
            body = outcome.response.read()
        self.assertIn(b"first-token", body)
        self.assertIn(b"backend-error", body)
        self.assertEqual(outcome.attempts, 1)
        self.assertEqual(self.server.counts["/stream"], 1)

    def test_retry_after_cannot_exceed_total_deadline(self) -> None:
        calls = 0

        def request_fn(request: object, timeout_s: float) -> FakeResponse:
            nonlocal calls
            del request, timeout_s
            calls += 1
            return FakeResponse(
                503,
                {
                    "x-llm-d-retryable": "true",
                    "x-llm-d-error-code": "stale_routing_decision",
                    "Retry-After": "10",
                },
            )

        outcome = send_with_typed_stale_retry(
            "http://gateway.invalid/v1/completions",
            b"{}",
            {},
            RetryPolicy(max_attempts=3, total_timeout_s=1),
            request_fn=request_fn,
            clock=lambda: 100.0,
        )
        self.assertEqual(calls, 1)
        self.assertEqual(outcome.attempts, 1)
        self.assertTrue(outcome.retry_budget_exhausted)
        outcome.response.close()

    def test_internal_routing_headers_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Gateway/EPP"):
            send_with_typed_stale_retry(
                self.base_url + "/ok",
                b"{}",
                {"x-llm-d-prefill-endpoint": "forged:8000"},
                RetryPolicy(),
            )


if __name__ == "__main__":
    unittest.main()
