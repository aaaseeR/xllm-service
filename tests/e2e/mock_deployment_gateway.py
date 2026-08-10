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
"""Offline V3 deployment actuator with real sidecar/runtime subprocesses."""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

import requests

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.logger import logger


def wait_until(timeout: float, predicate: Callable[[], object]) -> object:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except (OSError, requests.RequestException, ValueError):
            pass
        time.sleep(0.05)
    raise RuntimeError("timed out waiting for deployed replica")


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2.0)


@dataclass
class Replica:
    ordinal: int
    engine_uid: str
    engine_incarnation: str
    agent_port: int
    runtime_port: int
    runtime: subprocess.Popen[bytes]
    sidecar: subprocess.Popen[bytes]


class GatewayState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._lock = threading.RLock()
        self._replicas: dict[str, Replica] = {}
        self._operations: dict[str, dict[str, object]] = {}
        self._action_counts: dict[str, int] = {}
        self._query_count = 0
        self._terminated_resources: list[dict[str, object]] = []
        self._lost_create_response = False
        self._drain_proofs = 0
        self._heartbeat_forwarded = 0
        self._heartbeat_failed = 0
        self._max_active_replicas = 0

    @property
    def master_key(self) -> str:
        return f"/{self.args.etcd_namespace}/XLLM:SERVICE:MASTER"

    def _etcd_get(self, key: str) -> str | None:
        encoded = base64.b64encode(key.encode()).decode()
        response = requests.post(
            "http://127.0.0.1:2379/v3/kv/range",
            json={"key": encoded},
            timeout=0.5,
        )
        response.raise_for_status()
        values = response.json().get("kvs", [])
        if not values:
            return None
        return base64.b64decode(values[0].get("value", "")).decode()

    def leader_url(self) -> str | None:
        address = self._etcd_get(self.master_key)
        if not address:
            return None
        port = int(address.rsplit(":", 1)[1])
        mapping = {18889: 18888, 18899: 18898}
        return f"http://127.0.0.1:{mapping[port]}" if port in mapping else None

    def forward_heartbeat(
        self,
        body: bytes,
        headers: dict[str, str],
    ) -> tuple[int, bytes]:
        leader = self.leader_url()
        if leader is None:
            with self._lock:
                self._heartbeat_failed += 1
            return 503, b'{"error":"leader unavailable"}'
        try:
            response = requests.post(
                leader + "/v1/internal/heartbeat",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Internal-Token": headers.get("X-Internal-Token", ""),
                },
                timeout=1.0,
            )
        except requests.RequestException as error:
            with self._lock:
                self._heartbeat_failed += 1
            return 503, json.dumps({"error": str(error)}).encode()
        with self._lock:
            if response.status_code == 200:
                self._heartbeat_forwarded += 1
            else:
                self._heartbeat_failed += 1
        return response.status_code, response.content

    @staticmethod
    def _response(
        intent: dict[str, object],
        code: str,
        engine_uid: str,
        engine_incarnation: str,
        lifecycle: str,
        termination_proven: bool = False,
        replayed: bool = False,
        message: str = "offline deployment actuator",
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "code": code,
            "operation_id": intent["operation_id"],
            "action": intent["action"],
            "leader_incarnation": intent["leader_incarnation"],
            "leader_epoch": intent["leader_epoch"],
            "desired_generation": intent["desired_generation"],
            "pool": intent["pool"],
            "engine_uid": engine_uid,
            "engine_incarnation": engine_incarnation,
            "lifecycle": lifecycle,
            "termination_proven": termination_proven,
            "replayed": replayed,
            "message": message,
        }

    def _registration(self, address: str) -> dict[str, object] | None:
        key = f"/{self.args.etcd_namespace}/XLLM:DEFAULT:{address}"
        value = self._etcd_get(key)
        return json.loads(value) if value else None

    def _start_replica(self, intent: dict[str, object]) -> Replica:
        # The Placement ordinal is a globally unique cycle/operation identity,
        # not a compact replica index. The deployment system owns assignment
        # of that intent to concrete runtime slots.
        occupied = {replica.ordinal for replica in self._replicas.values()}
        ordinal = next(index for index in range(64) if index not in occupied)
        agent_port = self.args.agent_base_port + ordinal
        runtime_port = self.args.runtime_base_port + ordinal
        address = f"127.0.0.1:{agent_port}"
        repository = Path(self.args.repository_root)
        runtime_environment = dict(os.environ)
        runtime_environment.update(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
        )
        runtime = subprocess.Popen(
            [
                sys.executable,
                str(repository / "tests/e2e/mock_vllm_runtime.py"),
                f"--port={runtime_port}",
                f"--ordinal={ordinal}",
                f"--delay-ms={self.args.runtime_delay_ms}",
            ],
            cwd=repository,
            env=runtime_environment,
            start_new_session=True,
        )
        try:
            wait_until(
                10,
                lambda: requests.get(
                    f"http://127.0.0.1:{runtime_port}/health", timeout=0.3
                ).status_code
                == 200,
            )
            sidecar = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "vllm_sidecar.sidecar",
                    "--etcd-endpoints=127.0.0.1:2379",
                    f"--etcd-namespace={self.args.etcd_namespace}",
                    f"--vllm-url=http://127.0.0.1:{runtime_port}",
                    f"--metrics-url=http://127.0.0.1:{runtime_port}/metrics",
                    f"--register-addr={address}",
                    f"--agent-listen={address}",
                    f"--provider-config={self.args.provider_config}",
                    f"--xllm-service-url=http://127.0.0.1:{self.args.port}",
                    f"--internal-token={self.args.internal_token}",
                    f"--instance-name=offline-vllm-{ordinal}",
                    "--lease-ttl=3",
                    "--keepalive-interval=0.2",
                    "--heartbeat-interval=0.2",
                    "--health-timeout=2.0",
                    "--agent-ingress-timeout=10",
                    "--log-level=INFO",
                ],
                cwd=repository,
                start_new_session=True,
            )
        except Exception:
            stop_process(runtime)
            raise
        try:
            metadata = wait_until(12, lambda: self._registration(address))
            assert isinstance(metadata, dict)
            incarnation = str(metadata["incarnation_id"])
            wait_until(
                12,
                lambda: requests.get(
                    f"http://{address}/readyz", timeout=0.3
                ).status_code
                == 200,
            )
        except Exception:
            stop_process(sidecar)
            stop_process(runtime)
            raise
        return Replica(
            ordinal=ordinal,
            engine_uid=address,
            engine_incarnation=incarnation,
            agent_port=agent_port,
            runtime_port=runtime_port,
            runtime=runtime,
            sidecar=sidecar,
        )

    def _capture_runtime(self, replica: Replica) -> dict[str, object]:
        try:
            response = requests.get(
                f"http://127.0.0.1:{replica.runtime_port}/debug/state",
                timeout=1.0,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            return {"ordinal": replica.ordinal, "capture_failed": True}

    def _terminate_replica(self, replica: Replica) -> None:
        try:
            drained = (
                requests.get(
                    f"http://127.0.0.1:{replica.agent_port}/readyz", timeout=0.5
                ).status_code
                == 503
            )
        except requests.RequestException:
            drained = False
        if drained:
            self._drain_proofs += 1
        resource = self._capture_runtime(replica)
        resource["drain_admission_closed_before_terminate"] = drained
        self._terminated_resources.append(resource)
        stop_process(replica.sidecar)
        stop_process(replica.runtime)

    def execute(self, intent: dict[str, object]) -> tuple[dict[str, object], bool]:
        operation_id = str(intent["operation_id"])
        with self._lock:
            previous = self._operations.get(operation_id)
            if previous is not None:
                replay = dict(previous)
                replay["replayed"] = True
                return replay, False
            action = str(intent["action"])
            self._action_counts[action] = self._action_counts.get(action, 0) + 1
            if action == "CREATE":
                replica = self._start_replica(intent)
                self._replicas[replica.engine_uid] = replica
                self._max_active_replicas = max(
                    self._max_active_replicas, len(self._replicas)
                )
                response = self._response(
                    intent,
                    "SUCCEEDED",
                    replica.engine_uid,
                    replica.engine_incarnation,
                    "READY",
                    message="strict Agent registered and ready",
                )
                self._operations[operation_id] = response
                lose = not self._lost_create_response
                self._lost_create_response = True
                return response, lose
            if action == "TERMINATE":
                engine_uid = str(intent["engine_uid"])
                replica = self._replicas.pop(engine_uid, None)
                if replica is not None:
                    self._terminate_replica(replica)
                response = self._response(
                    intent,
                    "SUCCEEDED",
                    str(intent["engine_uid"]),
                    str(intent["engine_incarnation"]),
                    "ABSENT",
                    termination_proven=True,
                    message="sidecar lease revoked and runtime terminated",
                )
                self._operations[operation_id] = response
                return response, False
            response = self._response(
                intent,
                "INVALID",
                str(intent.get("engine_uid", "")),
                str(intent.get("engine_incarnation", "")),
                "FAILED",
                message="unsupported action",
            )
            self._operations[operation_id] = response
            return response, False

    def query(self, intent: dict[str, object]) -> dict[str, object]:
        with self._lock:
            self._query_count += 1
            response = self._operations.get(str(intent["operation_id"]))
            if response is not None:
                replay = dict(response)
                replay["replayed"] = True
                return replay
            return self._response(
                intent,
                "NOT_FOUND",
                str(intent.get("engine_uid", "")),
                str(intent.get("engine_incarnation", "")),
                "ABSENT",
                message="operation not found",
            )

    def debug_state(self) -> dict[str, object]:
        with self._lock:
            active_resources = [
                self._capture_runtime(replica)
                for replica in self._replicas.values()
            ]
            operations = [dict(value) for value in self._operations.values()]
            return {
                "active_replicas": len(self._replicas),
                "max_active_replicas": self._max_active_replicas,
                "replicas": [
                    {
                        "ordinal": replica.ordinal,
                        "engine_uid": replica.engine_uid,
                        "engine_incarnation": replica.engine_incarnation,
                        "runtime_pid": replica.runtime.pid,
                        "sidecar_pid": replica.sidecar.pid,
                    }
                    for replica in self._replicas.values()
                ],
                "operations": operations,
                "action_counts": dict(self._action_counts),
                "query_count": self._query_count,
                "lost_create_response_count": int(self._lost_create_response),
                "drain_proofs": self._drain_proofs,
                "heartbeat_forwarded": self._heartbeat_forwarded,
                "heartbeat_failed": self._heartbeat_failed,
                "active_resources": active_resources,
                "terminated_resources": list(self._terminated_resources),
            }

    def close(self) -> None:
        with self._lock:
            for replica in list(self._replicas.values()):
                resource = self._capture_runtime(replica)
                self._terminated_resources.append(resource)
                stop_process(replica.sidecar)
                stop_process(replica.runtime)
            self._replicas.clear()


class GatewayServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128

    def __init__(self, address: tuple[str, int], state: GatewayState) -> None:
        super().__init__(address, GatewayHandler)
        self.state = state


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def gateway(self) -> GatewayState:
        return self.server.state  # type: ignore[attr-defined,no-any-return]

    def log_message(self, message: str, *arguments: object) -> None:
        logger.info("deployment-gateway: " + message, *arguments)

    def _write(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, status: int, payload: object) -> None:
        self._write(status, json.dumps(payload, separators=(",", ":")).encode())

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/health":
            self._json(200, {"status": "healthy"})
        elif path == "/debug/state":
            self._json(200, self.gateway.debug_state())
        else:
            self._json(404, {"error": "unsupported path"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(400, {"error": "invalid content length"})
            return
        body = self.rfile.read(length)
        if path == "/v1/internal/heartbeat":
            status, response = self.gateway.forward_heartbeat(
                body, dict(self.headers.items())
            )
            self._write(status, response)
            return
        if path not in (
            "/v1/internal/placement/execute",
            "/v1/internal/placement/query",
        ):
            self._json(404, {"error": "unsupported path"})
            return
        if self.headers.get("X-Internal-Token") != self.gateway.args.internal_token:
            self._json(401, {"error": "unauthorized"})
            return
        try:
            intent = json.loads(body)
            if not isinstance(intent, dict):
                raise ValueError("intent must be an object")
        except (json.JSONDecodeError, ValueError) as error:
            self._json(400, {"error": str(error)})
            return
        if path.endswith("/query"):
            self._json(200, self.gateway.query(intent))
            return
        try:
            response, lose_response = self.gateway.execute(intent)
        except Exception as error:
            self._json(500, {"error": str(error)})
            return
        if lose_response:
            self.close_connection = True
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
            return
        self._json(200, response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18280)
    parser.add_argument("--etcd-namespace", required=True)
    parser.add_argument("--provider-config", required=True)
    parser.add_argument("--internal-token", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--agent-base-port", type=int, default=18300)
    parser.add_argument("--runtime-base-port", type=int, default=19300)
    parser.add_argument("--runtime-delay-ms", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = GatewayState(args)
    server = GatewayServer(("127.0.0.1", args.port), state)

    def stop(_signum: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever()
    finally:
        state.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
