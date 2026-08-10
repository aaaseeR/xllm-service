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
# ==============================================================================
"""Protocol-speaking vLLM stand-in with a Torch CPU simulated-HBM arena."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import signal
import threading
import time
import uuid
from urllib.parse import urlsplit

import torch

torch.set_num_threads(1)


class SimulatedHbm:
    """Fixed Torch tensor with explicit allocation, zeroing and accounting."""

    def __init__(self, blocks: int, block_bytes: int) -> None:
        self._tensor = torch.zeros(
            (blocks, block_bytes // 4), dtype=torch.int32, device="cpu"
        )
        self._free = list(range(blocks))
        self._owned: dict[str, list[int]] = {}
        self._lock = threading.Lock()
        self._high_watermark = 0
        self._allocations = 0

    def reserve(self, request_id: str, blocks: int) -> bool:
        with self._lock:
            if request_id in self._owned or blocks > len(self._free):
                return False
            selected = [self._free.pop() for _ in range(blocks)]
            marker = abs(hash(request_id)) % 2147483646 + 1
            indices = torch.tensor(selected, dtype=torch.int64)
            self._tensor.index_fill_(0, indices, marker)
            self._owned[request_id] = selected
            self._allocations += 1
            self._high_watermark = max(
                self._high_watermark, self._tensor.shape[0] - len(self._free)
            )
            return True

    def release(self, request_id: str) -> None:
        with self._lock:
            selected = self._owned.pop(request_id)
            indices = torch.tensor(selected, dtype=torch.int64)
            self._tensor.index_fill_(0, indices, 0)
            self._free.extend(selected)

    def snapshot(self) -> dict[str, int | float | bool]:
        with self._lock:
            used = self._tensor.shape[0] - len(self._free)
            return {
                "total_blocks": self._tensor.shape[0],
                "used_blocks": used,
                "active_allocations": len(self._owned),
                "high_watermark_blocks": self._high_watermark,
                "allocation_count": self._allocations,
                "used_ratio": used / self._tensor.shape[0],
                "tensor_zero": bool(torch.count_nonzero(self._tensor).item() == 0),
            }


class RuntimeState:
    def __init__(self, ordinal: int, delay_ms: int) -> None:
        self.ordinal = ordinal
        self.delay_ms = delay_ms
        self.hbm = SimulatedHbm(128, 16 * 1024)
        self._lock = threading.Lock()
        self.running = 0
        self.waiting = 0
        self.completed = 0
        self.failed = 0
        self.ttft_sum = 0.0
        self.tbt_sum = 0.0

    def begin(self) -> None:
        with self._lock:
            self.running += 1

    def finish(self, elapsed: float, ok: bool) -> None:
        with self._lock:
            self.running -= 1
            if ok:
                self.completed += 1
                self.ttft_sum += elapsed
                self.tbt_sum += elapsed
            else:
                self.failed += 1

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            result: dict[str, object] = {
                "ordinal": self.ordinal,
                "running": self.running,
                "waiting": self.waiting,
                "completed": self.completed,
                "failed": self.failed,
                "ttft_sum_seconds": self.ttft_sum,
                "tbt_sum_seconds": self.tbt_sum,
            }
        result["simulated_hbm"] = self.hbm.snapshot()
        return result


class RuntimeServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 1024

    def __init__(self, address: tuple[str, int], state: RuntimeState) -> None:
        super().__init__(address, RuntimeHandler)
        self.state = state


class RuntimeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def runtime(self) -> RuntimeState:
        return self.server.state  # type: ignore[attr-defined,no-any-return]

    def log_message(self, _format: str, *_arguments: object) -> None:
        return

    def _write(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: object) -> None:
        self._write(
            status,
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            "application/json",
        )

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/health":
            self._json(200, {"status": "healthy"})
            return
        if path == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "offline-e2e-model-r1",
                            "object": "model",
                            "owned_by": "xllm-e2e",
                        }
                    ],
                },
            )
            return
        if path == "/metrics":
            snapshot = self.runtime.snapshot()
            hbm = snapshot["simulated_hbm"]
            assert isinstance(hbm, dict)
            lines = [
                f'vllm:num_requests_running{{dp_rank="0"}} {snapshot["running"]}',
                'vllm:num_requests_waiting_for_capacity{dp_rank="0"} 0',
                'vllm:num_requests_waiting_deferred{dp_rank="0"} 0',
                f'vllm:gpu_cache_usage_perc{{dp_rank="0"}} {hbm["used_ratio"]}',
                f'vllm:time_to_first_token_seconds_sum {snapshot["ttft_sum_seconds"]}',
                f'vllm:time_to_first_token_seconds_count {snapshot["completed"]}',
                f'vllm:time_per_output_token_seconds_sum {snapshot["tbt_sum_seconds"]}',
                f'vllm:time_per_output_token_seconds_count {snapshot["completed"]}',
                f'xllm_e2e_simulated_hbm_used_blocks {hbm["used_blocks"]}',
                f'xllm_e2e_simulated_hbm_high_watermark_blocks {hbm["high_watermark_blocks"]}',
            ]
            self._write(200, ("\n".join(lines) + "\n").encode(), "text/plain")
            return
        if path == "/debug/state":
            self._json(200, self.runtime.snapshot())
            return
        self._json(404, {"error": "unsupported path"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path not in ("/v1/completions", "/v1/chat/completions"):
            self._json(404, {"error": "unsupported path"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError):
            self._json(400, {"error": "invalid JSON"})
            return
        request_id = str(payload.get("request_id") or uuid.uuid4().hex)
        if not self.runtime.hbm.reserve(request_id, 2):
            self._json(503, {"error": "simulated HBM capacity exhausted"})
            return
        started = time.monotonic()
        ok = False
        self.runtime.begin()
        try:
            prompt = payload.get("prompt", "")
            delay_ms = self.runtime.delay_ms
            if isinstance(prompt, str) and prompt.startswith(
                "__xllm_e2e_hold_3000ms__"
            ):
                # Keep an actual inference attempt active while the hard-gate
                # harness closes Agent admission through the real lifecycle
                # API.  The marker stays inside this protocol-speaking mock;
                # no production service test hook is involved.
                delay_ms = 3000
            time.sleep(delay_ms / 1000.0)
            text = f"mock-vllm replica={self.runtime.ordinal}"
            self._json(
                200,
                {
                    "id": f"cmpl-{request_id}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": "offline-e2e-model-r1",
                    "choices": [
                        {
                            "index": 0,
                            "text": text,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "total_tokens": 4,
                    },
                },
            )
            ok = True
        finally:
            self.runtime.finish(time.monotonic() - started, ok)
            self.runtime.hbm.release(request_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ordinal", type=int, required=True)
    parser.add_argument("--delay-ms", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = RuntimeServer(
        ("127.0.0.1", args.port), RuntimeState(args.ordinal, args.delay_ms)
    )

    def stop(_signum: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    server.serve_forever()
    server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
