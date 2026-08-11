#!/usr/bin/env python3
# Copyright 2025-2026 The xLLM Authors.
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
"""Offline multi-process V2/V3 cluster load and fault-injection gate.

This is intentionally not a unit-test harness. It launches the production
xllm-service binary, a real etcd server, protocol-speaking Engine/Agent
processes, concurrent OpenAI HTTP clients, and fault injectors. CPU tensors are
used as bounded simulated HBM/KV capacity so leaks and high-water marks remain
observable before the NPU validation stage.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable
import concurrent.futures
from dataclasses import asdict, dataclass, field
import http.client
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import requests
import torch

torch.set_num_threads(1)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vllm_sidecar.etcd_registry import EtcdError, EtcdGatewayClient
from vllm_sidecar.descriptor import build_provider_descriptor
from scripts.logger import logger


INTERNAL_TOKEN = "offline-e2e-internal-token"
MODEL_REVISION = "offline-e2e-model-r1"
RENDERER_DIGEST = "offline-e2e-renderer-r1"


class GateFailure(RuntimeError):
    pass


def is_expected_abrupt_loss_error(error: str) -> bool:
    """Recognize only transport loss or an exact Runtime-fence proof."""
    transport_tokens = (
        "Connection refused",
        "Connection reset by peer",
        "backend instance is not available",
    )
    if any(token in error for token in transport_tokens):
        return True

    marker = "409 Conflict: "
    if marker not in error:
        return False
    try:
        payload = json.loads(error.split(marker, 1)[1])
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(payload, dict) or set(payload) != {
        "accepted",
        "replayed",
        "state",
        "reason",
        "request_uid",
        "attempt_seq",
        "incarnation_id",
    }:
        return False
    return (
        payload["accepted"] is True
        and payload["replayed"] is True
        and payload["state"] == "ATTEMPT_LIFECYCLE_STATE_CANCELLED"
        and payload["reason"] == "ADMISSION_REASON_INTERNAL_ERROR"
        and isinstance(payload["request_uid"], str)
        and bool(payload["request_uid"])
        and type(payload["attempt_seq"]) is int
        and payload["attempt_seq"] >= 0
        and isinstance(payload["incarnation_id"], str)
        and bool(payload["incarnation_id"])
    )


def wait_until(
    description: str,
    timeout: float,
    predicate: Callable[[], object],
) -> object:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except (OSError, requests.RequestException, EtcdError) as error:
            last_error = error
        time.sleep(0.1)
    detail = f": {last_error}" if last_error is not None else ""
    raise GateFailure(f"timed out waiting for {description}{detail}")


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(0, index)]


class ManagedProcess:
    def __init__(self, name: str, command: list[str], log_dir: Path) -> None:
        self.name = name
        self.command = command
        self.log_path = log_dir / f"{name}.log"
        self._log = self.log_path.open("wb")
        self.process = subprocess.Popen(
            command,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    @property
    def pid(self) -> int:
        return self.process.pid

    def check(self) -> None:
        result = self.process.poll()
        if result is not None:
            raise GateFailure(
                f"process {self.name} exited with {result}; log={self.log_path}"
            )

    def send_signal(self, signum: int) -> None:
        if self.process.poll() is None:
            os.killpg(self.process.pid, signum)

    def stop(self, timeout: float = 5.0) -> None:
        if self.process.poll() is None:
            self.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.send_signal(signal.SIGKILL)
                self.process.wait(timeout=2.0)
        self._log.close()


class TorchHbmArena:
    """Bounded CPU tensor arena with HBM-like block ownership semantics."""

    def __init__(self, total_blocks: int, block_bytes: int) -> None:
        if total_blocks <= 0 or block_bytes <= 0 or block_bytes % 4 != 0:
            raise ValueError("invalid simulated HBM geometry")
        self._tensor = torch.zeros(
            (total_blocks, block_bytes // 4), dtype=torch.int32, device="cpu"
        )
        self._free = list(range(total_blocks))
        self._owned: dict[str, list[int]] = {}
        self._lock = threading.Lock()
        self._high_watermark = 0
        self._checksum = 0

    def reserve(self, request_uid: str, blocks: int) -> bool:
        with self._lock:
            if request_uid in self._owned or blocks > len(self._free):
                return False
            selected = [self._free.pop() for _ in range(blocks)]
            marker = abs(hash(request_uid)) % 2147483646 + 1
            indices = torch.tensor(selected, dtype=torch.int64)
            self._tensor.index_fill_(0, indices, marker)
            self._owned[request_uid] = selected
            used = self._tensor.shape[0] - len(self._free)
            self._high_watermark = max(self._high_watermark, used)
            self._checksum ^= marker
            return True

    def release(self, request_uid: str) -> None:
        with self._lock:
            selected = self._owned.pop(request_uid, None)
            if selected is None:
                raise GateFailure(f"simulated HBM double release: {request_uid}")
            indices = torch.tensor(selected, dtype=torch.int64)
            self._tensor.index_fill_(0, indices, 0)
            self._free.extend(selected)

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            used = self._tensor.shape[0] - len(self._free)
            return {
                "total_blocks": self._tensor.shape[0],
                "used_blocks": used,
                "free_blocks": len(self._free),
                "active_allocations": len(self._owned),
                "high_watermark_blocks": self._high_watermark,
                "used_ratio": used / self._tensor.shape[0],
                "allocation_checksum": self._checksum,
            }

    def assert_converged(self) -> None:
        snapshot = self.snapshot()
        if snapshot["used_blocks"] != 0 or snapshot["active_allocations"] != 0:
            raise GateFailure(f"simulated HBM did not converge: {snapshot}")
        if int(torch.count_nonzero(self._tensor).item()) != 0:
            raise GateFailure("simulated HBM tensor retained nonzero KV data")


def native_descriptor(
    engine_uid: str, incarnation: str, address: str, role: str
) -> dict:
    profile_digest = f"offline-e2e-native-{role.lower()}-profile-r1"
    return {
        "contract_version": 1,
        "identity": {
            "engine_uid": engine_uid,
            "incarnation_id": incarnation,
            "provider_id": "PROVIDER_ID_XLLM_NATIVE",
            "runtime_family": "xllm",
            "runtime_version": "offline-e2e-r1",
            "plugin_version": "builtin-r1",
            "hardware_runtime_version": "torch-cpu-simulated-hbm-r1",
            "protocol_version": 1,
        },
        "endpoint": {
            "control_transport": "brpc",
            "data_transport": "brpc",
            "address": address,
        },
        "serving": {
            "role": role,
            "execution_modes": [
                {
                    "mode": "EXECUTION_MODE_REMOTE_PD",
                    "transfer_mode": "TRANSFER_MODE_LAYERWISE_PUSH",
                    "selection_order": "SELECTION_ORDER_P_FIRST",
                    "binding_stage": "BINDING_STAGE_BEFORE_PREFILL",
                    "p_selection_delegated": False,
                }
            ],
            "api_features": ["completions"],
        },
        "model": {
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": "offline-e2e-tokenizer-r1",
            "chat_template_digest": "offline-e2e-template-r1",
            "quantization": "none",
            "renderer_digest": RENDERER_DIGEST,
        },
        "topology": {
            "soc": "torch-cpu-simulated-npu",
            "device_count": 1,
            "tp": 1,
            "dp": 1,
            "pp": 1,
            "ep": 1,
            "cp": 1,
        },
        "kv": {
            "kv_layout_digest": "offline-e2e-kv-layout-r1",
            "cache_dtype": "bfloat16",
            "block_size": 16,
            "cache_groups": ["full-attention"],
            "head_shard_mapping_digest": "offline-e2e-head-map-r1",
            "connector": "native-simulated-hbm",
            "connector_version": "1",
            "transfer_modes": ["TRANSFER_MODE_LAYERWISE_PUSH"],
            "kv_namespace": "offline-e2e-kv-namespace-r1",
            "hash_version": 1,
            "hash_seed": 1024,
        },
        "scheduler": {
            "scheduler_class": "offline-e2e-native",
            "max_num_seqs": 64,
            "max_num_batched_tokens": 4096,
            "scheduler_policy_digest": "offline-e2e-scheduler-r1",
        },
        "capabilities": [
            "PROVIDER_CAPABILITY_REMOTE_PD_LAYERWISE_PUSH",
            "PROVIDER_CAPABILITY_NATIVE_RESERVATION",
            "PROVIDER_CAPABILITY_ATTEMPT_QUERY",
            "PROVIDER_CAPABILITY_CANCEL_FENCE",
            "PROVIDER_CAPABILITY_ENGINE_LOCAL_DEADLINE",
            "PROVIDER_CAPABILITY_STRUCTURED_ADMISSION",
            "PROVIDER_CAPABILITY_PER_DP_STATE",
            "PROVIDER_CAPABILITY_SELF_FENCING",
            "PROVIDER_CAPABILITY_DRAIN",
        ],
        "profile_digest": profile_digest,
    }


class NativeEngineLease:
    _LEASE_TTL_SECONDS = 5
    _OWNERSHIP_GUARD_SECONDS = 0.5

    def __init__(
        self,
        etcd_endpoint: str,
        namespace: str,
        engine_uid: str,
        incarnation: str,
        address: str,
        role: str,
        master_urls: Callable[[], list[str]],
        hbm: TorchHbmArena,
        on_ownership_lost: Callable[[], None],
    ) -> None:
        # A production Engine owns its etcd connection.  Do not share the
        # harness control client's requests.Session across concurrent Engine
        # lease threads: requests.Session is not thread-safe and a prolonged
        # etcd outage must exercise independent lease recovery.
        self._etcd = EtcdGatewayClient(etcd_endpoint, timeout=0.75)
        self._namespace = namespace
        self.engine_uid = engine_uid
        self.incarnation = incarnation
        self.address = address
        self.role = role
        self._master_urls = master_urls
        self._hbm = hbm
        self._on_ownership_lost = on_ownership_lost
        self._descriptor = native_descriptor(engine_uid, incarnation, address, role)
        prefix = "PREFILL" if role == "ENGINE_ROLE_PREFILL" else "DECODE"
        self._key = f"/{namespace}/XLLM:{prefix}:{engine_uid}"
        self._lease_id: str | None = None
        self._registration_count = 0
        self._last_lease_error = ""
        self._last_confirmed_lease = time.monotonic()
        self._ownership_lost = threading.Event()
        self._state_seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._register()
        self._thread = threading.Thread(
            target=self._run, name=f"lease-{self.engine_uid}", daemon=True
        )
        self._thread.start()

    def _register(self) -> None:
        lease_id = self._etcd.lease_grant(self._LEASE_TTL_SECONDS)
        instance_type = 1 if self.role == "ENGINE_ROLE_PREFILL" else 2
        value = {
            "name": self.engine_uid,
            "rpc_address": self.address,
            "incarnation_id": self.incarnation,
            "register_ts_ms": int(time.time() * 1000),
            "type": instance_type,
            "addrs": ["127.0.0.1"],
            "cluster_ids": [abs(hash(self.engine_uid)) % 100000 + 1],
            "dp_size": 1,
            "kv_split_size": 1,
            "ports": [int(self.address.rsplit(":", 1)[1]) + 1000],
            "provider_id": 1,
            "provider_contract_version": 1,
            "provider_profile_digest": self._descriptor["profile_digest"],
            "backend_type": "xllm",
            "provider_descriptor": self._descriptor,
        }
        self._etcd.put(self._key, json.dumps(value), lease_id)
        self._lease_id = lease_id
        self._registration_count += 1
        self._last_lease_error = ""
        self._last_confirmed_lease = time.monotonic()

    @property
    def membership_key(self) -> str:
        return self._key

    @property
    def registration_count(self) -> int:
        return self._registration_count

    @property
    def last_lease_error(self) -> str:
        return self._last_lease_error

    @property
    def ownership_lost(self) -> bool:
        return self._ownership_lost.is_set()

    @property
    def last_confirmed_lease(self) -> float:
        return self._last_confirmed_lease

    def _heartbeat_body(self) -> dict:
        self._state_seq += 1
        hbm = self._hbm.snapshot()
        return {
            "name": self.engine_uid,
            "incarnation_id": self.incarnation,
            "load_metrics": {
                "waiting_requests_num": 0,
                "gpu_cache_usage_perc": hbm["used_ratio"],
            },
            "latency_metrics": {"recent_max_ttft": 2, "recent_max_tbt": 1},
            "engine_state": {
                "engine_uid": self.engine_uid,
                "incarnation_id": self.incarnation,
                "state_seq": self._state_seq,
                "observed_at_unix_ms": int(time.time() * 1000),
                "lifecycle": "ENGINE_LIFECYCLE_READY",
                "ownership": "ENGINE_OWNERSHIP_OWNED",
                "shallow_health": "HEALTH_STATUS_HEALTHY",
                "deep_health": "HEALTH_STATUS_UNKNOWN",
                "per_dp": [
                    {
                        "dp_rank": 0,
                        "running": hbm["active_allocations"],
                        "waiting_capacity": 0,
                        "waiting_deferred": 0,
                        "kv_used_ratio": hbm["used_ratio"],
                        "kv_free_blocks": hbm["free_blocks"],
                        "admission_credit": max(
                            0, 64 - int(hbm["active_allocations"])
                        ),
                    }
                ],
                "state_quality": "STATE_QUALITY_FULL",
                "provider_id": "PROVIDER_ID_XLLM_NATIVE",
                "profile_digest": self._descriptor["profile_digest"],
                "model_revision": MODEL_REVISION,
                "heartbeat_age_ms_at_publish": 0,
                "state_age_ms_at_publish": 0,
            },
        }

    def _send_heartbeat(self) -> None:
        body = self._heartbeat_body()
        headers = {
            "Content-Type": "application/json",
            "X-Internal-Token": INTERNAL_TOKEN,
        }
        for url in self._master_urls():
            try:
                response = requests.post(
                    url + "/v1/internal/heartbeat",
                    json=body,
                    headers=headers,
                    timeout=0.5,
                )
                if response.status_code == 200:
                    return
            except requests.RequestException:
                continue

    def _run(self) -> None:
        while not self._stop.wait(0.25):
            try:
                if self._lease_id is None or self._etcd.lease_keepalive(
                    self._lease_id
                ) <= 0:
                    self._lease_id = None
                    self._self_fence()
                    break
                self._last_confirmed_lease = time.monotonic()
                self._last_lease_error = ""
                self._send_heartbeat()
            except EtcdError as error:
                self._last_lease_error = str(error)
                ownership_deadline = (
                    self._last_confirmed_lease
                    + self._LEASE_TTL_SECONDS
                    - self._OWNERSHIP_GUARD_SECONDS
                )
                if time.monotonic() >= ownership_deadline:
                    # The old process can no longer prove ownership before an
                    # external observer may expire its lease.  It must fence
                    # itself; only a supervisor restart with a new
                    # incarnation may register again.
                    self._lease_id = None
                    self._self_fence()
                    break
                # Registry is unavailable but the locally proven ownership
                # window is still open, so keep the existing data-plane
                # heartbeat path alive during the bounded grace period.
                self._send_heartbeat()

    def _self_fence(self) -> None:
        if self._ownership_lost.is_set():
            return
        self._ownership_lost.set()
        self._on_ownership_lost()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._lease_id is not None:
            try:
                self._etcd.lease_revoke(self._lease_id)
            except EtcdError:
                pass
            self._lease_id = None


@dataclass
class LoadResult:
    phase: str
    requested: int
    succeeded: int
    failed: int
    routes: dict[str, int]
    error_samples: list[str]
    duration_seconds: float
    requests_per_second: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float


class V2LoadClient:
    def __init__(
        self,
        master_urls: Callable[[], list[str]],
        hbm: TorchHbmArena,
    ) -> None:
        self._master_urls = master_urls
        self._hbm = hbm

    def _url(self) -> str:
        urls = self._master_urls()
        if not urls:
            raise GateFailure("no live xllm-service HTTP endpoint")
        # Client traffic enters through the elected service.  The remaining
        # URLs are only heartbeat/failover candidates, not load-balancing
        # targets while they are followers.
        return urls[0]

    def _one(
        self, index: int, remaining_deadline_ms: int | None = None
    ) -> tuple[bool, float, str]:
        request_uid = f"load-{uuid.uuid4().hex}-{index}"
        if not self._hbm.reserve(request_uid, 2):
            return False, 0.0, "simulated HBM capacity exhausted"
        started = time.monotonic()
        try:
            payload: dict[str, object] = {
                "model": MODEL_REVISION,
                # Exercise routing with non-identical workload keys.  A
                # single repeated prompt legitimately stays on one
                # rendezvous-hash route and cannot validate fan-out.
                "prompt": f"hello distributed service {index}",
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": False,
            }
            if remaining_deadline_ms is not None:
                payload["remaining_deadline_ms"] = remaining_deadline_ms
            response = requests.post(
                self._url() + "/v1/completions",
                json=payload,
                headers={"X-Request-Id": request_uid},
                timeout=8.0,
            )
            latency_ms = (time.monotonic() - started) * 1000.0
            if response.status_code != 200:
                return False, latency_ms, f"HTTP {response.status_code}: {response.text[:200]}"
            payload = response.json()
            text = payload["choices"][0]["text"]
            if not text.startswith("mock-native route=") or "->" not in text:
                return False, latency_ms, f"invalid completion: {text!r}"
            return True, latency_ms, text.removeprefix("mock-native route=")
        except (requests.RequestException, ValueError, KeyError, IndexError) as error:
            return False, (time.monotonic() - started) * 1000.0, str(error)
        finally:
            self._hbm.release(request_uid)

    def _stream_one(self, index: int) -> tuple[bool, float, str]:
        request_uid = f"stream-{uuid.uuid4().hex}-{index}"
        if not self._hbm.reserve(request_uid, 2):
            return False, 0.0, "simulated HBM capacity exhausted"
        started = time.monotonic()
        try:
            with requests.post(
                self._url() + "/v1/completions",
                json={
                    "model": MODEL_REVISION,
                    "prompt": f"hello streaming service {index}",
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "stream": True,
                },
                headers={"X-Request-Id": request_uid},
                timeout=8.0,
                stream=True,
            ) as response:
                if response.status_code != 200:
                    latency_ms = (time.monotonic() - started) * 1000.0
                    return False, latency_ms, f"HTTP {response.status_code}: {response.text[:200]}"
                chunks: list[str] = []
                done = False
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        done = True
                        continue
                    event = json.loads(data)
                    choices = event.get("choices", [])
                    if choices:
                        chunks.append(str(choices[0].get("text", "")))
                text = "".join(chunks)
                latency_ms = (time.monotonic() - started) * 1000.0
                if not done or not text.startswith("mock-native route=") or "->" not in text:
                    return False, latency_ms, f"invalid SSE completion: done={done}, text={text!r}"
                return True, latency_ms, text.removeprefix("mock-native route=")
        except (requests.RequestException, ValueError, KeyError, IndexError) as error:
            return False, (time.monotonic() - started) * 1000.0, str(error)
        finally:
            self._hbm.release(request_uid)

    @staticmethod
    def summarize(
        phase: str,
        started: float,
        results: list[tuple[bool, float, str]],
    ) -> LoadResult:
        duration = time.monotonic() - started
        latencies = [latency for _, latency, _ in results]
        routes: dict[str, int] = {}
        errors: list[str] = []
        succeeded = 0
        for ok, _, detail in results:
            if ok:
                succeeded += 1
                routes[detail] = routes.get(detail, 0) + 1
            elif len(errors) < 10:
                errors.append(detail)
        return LoadResult(
            phase=phase,
            requested=len(results),
            succeeded=succeeded,
            failed=len(results) - succeeded,
            routes=routes,
            error_samples=errors,
            duration_seconds=duration,
            requests_per_second=len(results) / duration,
            latency_mean_ms=statistics.mean(latencies) if latencies else 0.0,
            latency_p50_ms=percentile(latencies, 0.50),
            latency_p95_ms=percentile(latencies, 0.95),
            latency_p99_ms=percentile(latencies, 0.99),
        )

    def run(self, phase: str, requests_count: int, concurrency: int) -> LoadResult:
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="v2-load"
        ) as pool:
            results = list(pool.map(self._one, range(requests_count)))
        return self.summarize(phase, started, results)

    def run_streaming(
        self, phase: str, requests_count: int, concurrency: int
    ) -> LoadResult:
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="v2-stream"
        ) as pool:
            results = list(pool.map(self._stream_one, range(requests_count)))
        return self.summarize(phase, started, results)

    def run_deadline(
        self,
        phase: str,
        requests_count: int,
        concurrency: int,
        remaining_deadline_ms: int,
    ) -> LoadResult:
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="v2-deadline"
        ) as pool:
            futures = [
                pool.submit(self._one, index, remaining_deadline_ms)
                for index in range(requests_count)
            ]
            results = [future.result() for future in futures]
        return self.summarize(phase, started, results)


class V3LoadClient:
    def __init__(self, master_urls: Callable[[], list[str]]) -> None:
        self._master_urls = master_urls

    def _url(self) -> str:
        urls = self._master_urls()
        if not urls:
            raise GateFailure("no live xllm-service HTTP endpoint")
        return urls[0]

    def _one(
        self,
        index: int,
        hold_open: bool = False,
        remaining_deadline_ms: int | None = None,
    ) -> tuple[bool, float, str]:
        request_uid = f"v3-load-{uuid.uuid4().hex}-{index}"
        started = time.monotonic()
        prompt = f"hello autoscaled service {index}"
        if hold_open:
            # Keep the drain-race requests alive long enough for the test to
            # observe Runtime ownership, pause heartbeat propagation, and
            # deliver the lifecycle command even on a loaded CI host.
            hold_ms = 700 if remaining_deadline_ms is not None else 1500
            prompt = f"__xllm_e2e_hold_{hold_ms}ms__ {prompt}"
        try:
            payload: dict[str, object] = {
                "model": MODEL_REVISION,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": False,
            }
            if remaining_deadline_ms is not None:
                payload["remaining_deadline_ms"] = remaining_deadline_ms
            response = requests.post(
                self._url() + "/v1/completions",
                json=payload,
                headers={"X-Request-Id": request_uid},
                timeout=10.0,
            )
            latency_ms = (time.monotonic() - started) * 1000.0
            if response.status_code != 200:
                return (
                    False,
                    latency_ms,
                    f"HTTP {response.status_code}: {response.text[:1024]}",
                )
            text = response.json()["choices"][0]["text"]
            if not text.startswith("mock-vllm replica="):
                return False, latency_ms, f"invalid completion: {text!r}"
            return True, latency_ms, text.removeprefix("mock-vllm replica=")
        except (requests.RequestException, ValueError, KeyError, IndexError) as error:
            return False, (time.monotonic() - started) * 1000.0, str(error)

    @staticmethod
    def summarize(
        phase: str,
        started: float,
        results: list[tuple[bool, float, str]],
    ) -> LoadResult:
        duration = time.monotonic() - started
        requests_count = len(results)
        latencies = [latency for _, latency, _ in results]
        routes: dict[str, int] = {}
        errors: list[str] = []
        succeeded = 0
        for ok, _, detail in results:
            if ok:
                succeeded += 1
                routes[detail] = routes.get(detail, 0) + 1
            elif len(errors) < 10:
                errors.append(detail)
        return LoadResult(
            phase=phase,
            requested=requests_count,
            succeeded=succeeded,
            failed=requests_count - succeeded,
            routes=routes,
            error_samples=errors,
            duration_seconds=duration,
            requests_per_second=requests_count / duration,
            latency_mean_ms=statistics.mean(latencies) if latencies else 0.0,
            latency_p50_ms=percentile(latencies, 0.50),
            latency_p95_ms=percentile(latencies, 0.95),
            latency_p99_ms=percentile(latencies, 0.99),
        )

    def run(self, phase: str, requests_count: int, concurrency: int) -> LoadResult:
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="v3-load"
        ) as pool:
            results = list(pool.map(self._one, range(requests_count)))
        return self.summarize(phase, started, results)

    def run_streaming(
        self, phase: str, requests_count: int, concurrency: int
    ) -> LoadResult:
        def stream_one(index: int) -> tuple[bool, float, str]:
            request_uid = f"v3-stream-{uuid.uuid4().hex}-{index}"
            started = time.monotonic()
            try:
                with requests.post(
                    self._url() + "/v1/completions",
                    json={
                        "model": MODEL_REVISION,
                        "prompt": f"hello autoscaled streaming service {index}",
                        "max_tokens": 1,
                        "temperature": 0.0,
                        "stream": True,
                    },
                    headers={"X-Request-Id": request_uid},
                    timeout=10.0,
                    stream=True,
                ) as response:
                    if response.status_code != 200:
                        latency_ms = (time.monotonic() - started) * 1000.0
                        return False, latency_ms, f"HTTP {response.status_code}: {response.text[:200]}"
                    chunks: list[str] = []
                    done = False
                    for line in response.iter_lines(decode_unicode=True):
                        if not line or not line.startswith("data:"):
                            continue
                        data = line.removeprefix("data:").strip()
                        if data == "[DONE]":
                            done = True
                            continue
                        event = json.loads(data)
                        choices = event.get("choices", [])
                        if choices:
                            chunks.append(str(choices[0].get("text", "")))
                    text = "".join(chunks)
                    latency_ms = (time.monotonic() - started) * 1000.0
                    if not done or not text.startswith("mock-vllm replica="):
                        return False, latency_ms, f"invalid SSE completion: done={done}, text={text!r}"
                    return True, latency_ms, text.removeprefix("mock-vllm replica=")
            except (requests.RequestException, ValueError, KeyError, IndexError) as error:
                return False, (time.monotonic() - started) * 1000.0, str(error)

        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="v3-stream"
        ) as pool:
            results = list(pool.map(stream_one, range(requests_count)))
        return self.summarize(phase, started, results)

    def run_deadline(
        self,
        phase: str,
        requests_count: int,
        concurrency: int,
        remaining_deadline_ms: int,
    ) -> LoadResult:
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="v3-deadline"
        ) as pool:
            futures = [
                pool.submit(self._one, index, True, remaining_deadline_ms)
                for index in range(requests_count)
            ]
            results = [future.result() for future in futures]
        return self.summarize(phase, started, results)

    def run_paced(
        self,
        phase: str,
        max_requests: int,
        interval_seconds: float,
        stop_predicate: Callable[[], bool],
    ) -> LoadResult:
        started = time.monotonic()
        results: list[tuple[bool, float, str]] = []
        for index in range(max_requests):
            results.append(self._one(index))
            if index >= 4 and stop_predicate():
                break
            time.sleep(interval_seconds)
        return self.summarize(phase, started, results)


@dataclass
class GateReport:
    run_id: str
    mode: str
    started_at_unix_ms: int
    topology: dict[str, object]
    phases: list[dict[str, object]] = field(default_factory=list)
    faults: list[dict[str, object]] = field(default_factory=list)
    resource_evidence: dict[str, object] = field(default_factory=dict)
    metric_evidence: dict[str, object] = field(default_factory=dict)
    passed: bool = False


class ClusterGate:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        self.root = Path(args.artifact_dir) / self.run_id
        self.root.mkdir(parents=True)
        self.log_dir = self.root / "logs"
        self.log_dir.mkdir()
        self.runtime_dir = Path(tempfile.mkdtemp(prefix="xllm-e2e-"))
        self.namespace = f"offline-e2e-{self.run_id}"
        self.etcd = EtcdGatewayClient("127.0.0.1:2379", timeout=0.5)
        self._cached_master_address: str | None = None
        self.processes: list[ManagedProcess] = []
        self.masters: dict[str, ManagedProcess] = {}
        self.engine_processes: dict[str, ManagedProcess] = {}
        self.engine_leases: dict[str, NativeEngineLease] = {}
        self.hbm = TorchHbmArena(total_blocks=128, block_bytes=64 * 1024)
        topology: dict[str, object] = {
            "etcd": 1,
            "xllm_service_masters": 2,
            "torch_version": torch.__version__,
        }
        if args.scenario == "v2":
            topology.update(
                {
                    "native_prefill_engines": 2,
                    "native_decode_engines": 2,
                    "simulated_hbm_bytes": 128 * 64 * 1024,
                }
            )
        else:
            topology.update(
                {
                    "deployment_gateways": 1,
                    "strict_vllm_agent_replicas": "autoscaled 1..3",
                    "simulated_hbm_bytes_per_replica": 128 * 16 * 1024,
                }
            )
        self.report = GateReport(
            run_id=self.run_id,
            mode=args.mode,
            started_at_unix_ms=int(time.time() * 1000),
            topology=topology,
        )

    @property
    def build_dir(self) -> Path:
        return Path(self.args.build_dir)

    def _start(self, name: str, command: list[str]) -> ManagedProcess:
        process = ManagedProcess(name, command, self.log_dir)
        self.processes.append(process)
        return process

    def _master_key(self) -> str:
        return f"/{self.namespace}/XLLM:SERVICE:MASTER"

    def master_rpc_address(self) -> str | None:
        try:
            address = self.etcd.get(self._master_key()) or None
            if address is not None:
                self._cached_master_address = address
            return address
        except EtcdError:
            return None

    @staticmethod
    def _etcd_value_with_revision(key: str) -> tuple[str, int]:
        response = requests.post(
            "http://127.0.0.1:2379/v3/kv/range",
            json={"key": base64.b64encode(key.encode()).decode()},
            timeout=1.0,
        )
        response.raise_for_status()
        values = response.json().get("kvs", [])
        if len(values) != 1:
            raise GateFailure(f"expected one etcd value for {key}: {values}")
        value = base64.b64decode(values[0].get("value", "")).decode()
        revision = int(values[0].get("mod_revision", 0))
        if not value or revision <= 0:
            raise GateFailure(f"invalid etcd identity value for {key}")
        return value, revision

    @staticmethod
    def _etcd_lease_ttl_for_key(key: str) -> int:
        response = requests.post(
            "http://127.0.0.1:2379/v3/kv/range",
            json={"key": base64.b64encode(key.encode()).decode()},
            timeout=1.0,
        )
        response.raise_for_status()
        values = response.json().get("kvs", [])
        if len(values) != 1 or not values[0].get("lease"):
            return 0
        ttl_response = requests.post(
            "http://127.0.0.1:2379/v3/lease/timetolive",
            json={"ID": values[0]["lease"]},
            timeout=1.0,
        )
        ttl_response.raise_for_status()
        return int(ttl_response.json().get("TTL") or 0)

    def master_identity(self) -> tuple[str, int]:
        master_prefix = f"/{self.namespace}/"
        _, address_revision = self._etcd_value_with_revision(
            master_prefix + "XLLM:SERVICE:MASTER"
        )
        incarnation, incarnation_revision = self._etcd_value_with_revision(
            master_prefix + "XLLM:STATE:MASTER_INCARNATION"
        )
        if address_revision != incarnation_revision:
            raise GateFailure(
                "master address and incarnation were not atomically published"
            )
        return incarnation, address_revision

    def master_urls(self) -> list[str]:
        address = self.master_rpc_address()
        ordered: list[str] = []
        if address:
            rpc_port = int(address.rsplit(":", 1)[1])
            if rpc_port == 18889:
                ordered.append("http://127.0.0.1:18888")
            elif rpc_port == 18899:
                ordered.append("http://127.0.0.1:18898")
        for url in ("http://127.0.0.1:18888", "http://127.0.0.1:18898"):
            name = "master-a" if url.endswith("18888") else "master-b"
            process = self.masters.get(name)
            if process is not None and process.process.poll() is None and url not in ordered:
                ordered.append(url)
        return ordered

    def cached_master_urls(self) -> list[str]:
        """Return data-plane endpoints without a synchronous etcd lookup."""
        address = self._cached_master_address
        ordered: list[str] = []
        if address:
            rpc_port = int(address.rsplit(":", 1)[1])
            if rpc_port == 18889:
                ordered.append("http://127.0.0.1:18888")
            elif rpc_port == 18899:
                ordered.append("http://127.0.0.1:18898")
        for url in ("http://127.0.0.1:18888", "http://127.0.0.1:18898"):
            name = "master-a" if url.endswith("18888") else "master-b"
            process = self.masters.get(name)
            if (
                process is not None
                and process.process.poll() is None
                and url not in ordered
            ):
                ordered.append(url)
        return ordered

    def leader_url(self) -> str | None:
        urls = self.master_urls()
        return urls[0] if urls else None

    def write_tokenizer(self) -> Path:
        path = self.runtime_dir / "tokenizer"
        path.mkdir()
        (path / "tokenizer.json").write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "truncation": None,
                    "padding": None,
                    "added_tokens": [],
                    "normalizer": None,
                    "pre_tokenizer": {"type": "Whitespace"},
                    "post_processor": None,
                    "decoder": None,
                    "model": {
                        "type": "WordLevel",
                        "vocab": {
                            "[UNK]": 0,
                            "hello": 1,
                            "distributed": 2,
                            "service": 3,
                        },
                        "unk_token": "[UNK]",
                    },
                }
            ),
            encoding="utf-8",
        )
        (path / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_class": "PreTrainedTokenizerFast",
                    "add_bos_token": False,
                    "add_eos_token": False,
                    "bos_token": "[UNK]",
                    "eos_token": "[UNK]",
                    "chat_template": (
                        "{% for message in messages %}"
                        "{{ message['content'] }}"
                        "{% endfor %}"
                    ),
                }
            ),
            encoding="utf-8",
        )
        (path / "config.json").write_text(
            json.dumps({"model_type": "offline_e2e"}), encoding="utf-8"
        )
        return path

    def start_etcd(self) -> ManagedProcess:
        data_dir = self.runtime_dir / "etcd"
        process = self._start(
            "etcd",
            [
                "etcd",
                "--name=offline-e2e",
                f"--data-dir={data_dir}",
                "--listen-client-urls=http://127.0.0.1:2379",
                "--advertise-client-urls=http://127.0.0.1:2379",
                "--listen-peer-urls=http://127.0.0.1:2380",
                "--initial-advertise-peer-urls=http://127.0.0.1:2380",
                "--initial-cluster=offline-e2e=http://127.0.0.1:2380",
                "--initial-cluster-state=new",
            ],
        )
        wait_until("etcd gateway", 15, lambda: self.etcd.lease_grant(2))
        return process

    def start_masters(
        self, tokenizer: Path, placement_config: Path | None = None
    ) -> None:
        binary = self.build_dir / "xllm_service/xllm_master_serving"
        if not binary.is_file():
            raise GateFailure(f"missing production service binary: {binary}")
        common = [
            str(binary),
            "--etcd_addr=127.0.0.1:2379",
            f"--etcd_namespace={self.namespace}",
            "--load_balance_policy=RR",
            "--kv_route_mode=SHADOW",
            "--readiness_check_interval_s=1",
            "--connect_timeout_ms=500",
            "--vllm_http_timeout_ms=2500",
            "--request_watchdog_interval_ms=20",
            "--engine_state_soft_ttl_ms=500",
            "--engine_state_hard_ttl_ms=2000",
            "--engine_heartbeat_hard_ttl_ms=2000",
            "--engine_direct_evidence_ttl_ms=1000",
            "--output_gap_timeout_ms=1000",
            "--output_gap_query_timeout_ms=100",
            "--p_first_event_retry_ub_ms=700",
            "--first_event_dispatch_margin_ms=200",
            "--flow_max_queued_requests=32",
            "--flow_max_dispatched_contexts=8",
            "--flow_max_model_queued_requests=32",
            "--flow_max_model_dispatched_contexts=8",
            "--observability_export_interval_ms=20",
            "--observability_snapshot_interval_ms=1000",
            "--observability_build_id=offline-e2e",
            f"--tokenizer_path={tokenizer}",
            f"--native_renderer_digest={RENDERER_DIGEST}",
            f"--internal_api_token={INTERNAL_TOKEN}",
        ]
        if placement_config is not None:
            common.extend(
                [
                    f"--placement_config_path={placement_config}",
                    "--placement_mode_override=3",
                ]
            )
        self.masters["master-a"] = self._start(
            "master-a",
            common + ["--http_server_port=18888", "--rpc_server_port=18889"],
        )
        wait_until("first service election", 20, self.master_rpc_address)
        self.masters["master-b"] = self._start(
            "master-b",
            common + ["--http_server_port=18898", "--rpc_server_port=18899"],
        )
        wait_until(
            "both service live endpoints",
            20,
            lambda: all(
                requests.get(url + "/livez", timeout=0.5).status_code == 200
                for url in ("http://127.0.0.1:18888", "http://127.0.0.1:18898")
            ),
        )

    def write_v3_configs(self) -> tuple[Path, Path, str]:
        provider = {
            "runtime": {
                "runtime_version": "offline-vllm-r1",
                "plugin_version": "offline-ascend-plugin-r1",
                "hardware_runtime_version": "torch-cpu-simulated-hbm-r1",
                "fate_bound_mode": "same_restart_unit",
                "raw_ingress_isolated": True,
            },
            "model": {
                "model_revision": MODEL_REVISION,
                "tokenizer_revision": "offline-e2e-tokenizer-r1",
                "chat_template_digest": "offline-e2e-template-r1",
                "quantization": "none",
                "renderer_digest": RENDERER_DIGEST,
            },
            "topology": {
                "soc": "torch-cpu-simulated-npu",
                "device_count": 1,
                "tp": 1,
                "dp": 1,
                "pp": 1,
                "ep": 1,
                "cp": 1,
            },
            "kv": {
                "kv_layout_digest": "offline-e2e-kv-layout-r1",
                "cache_dtype": "bfloat16",
                "block_size": 16,
                "cache_groups": ["full-attention"],
                "head_shard_mapping_digest": "offline-e2e-head-map-r1",
                "connector": "none",
                "connector_version": "1",
            },
            "scheduler": {
                "scheduler_class": "vllm-v1-offline",
                "max_num_seqs": 64,
                "max_num_batched_tokens": 4096,
                "scheduler_policy_digest": "offline-e2e-scheduler-r1",
            },
            "api_features": ["chat_completions", "completions", "models"],
        }
        descriptor = build_provider_descriptor(
            provider,
            "127.0.0.1:18300",
            "profile-only-incarnation",
            "127.0.0.1:18300",
        )
        profile_digest = descriptor["profile_digest"]
        provider_path = self.runtime_dir / "vllm-provider.json"
        provider_path.write_text(json.dumps(provider, indent=2), encoding="utf-8")

        placement = {
            "schema_version": 3,
            "loop_interval_ms": 100,
            "controller": {
                "mode": "ENFORCED",
                "max_pools": 1,
                "max_desired_snapshot_bytes": 262144,
                "max_operation_snapshot_bytes": 1048576,
                "max_devices": 3,
                "max_new_operations_per_cycle": 4,
                "max_actuator_actions_per_cycle": 8,
                "planner": {
                    "scale_up_hold_ms": 100,
                    "scale_down_stabilization_ms": 500,
                    "cooldown_ms": 500,
                    "economic_horizon_ms": 3600000,
                    "min_scale_down_samples": 1,
                    "max_scale_up_step": 2,
                    "max_scale_down_step": 1,
                    "queue_high_watermark": 1000.0,
                    "queue_low_watermark": 100.0,
                    "admission_reject_high_watermark": 0.5,
                    "admission_reject_low_watermark": 0.1,
                    "kv_high_watermark": 0.95,
                    "kv_low_watermark": 0.9,
                },
                "reconcile": {
                    "max_operations_per_cycle": 8,
                    "max_operations_per_pool": 64,
                    "max_create_per_cycle": 2,
                    "max_drain_per_cycle": 1,
                    "terminal_visibility_grace_ms": 500,
                },
            },
            "executor": {
                "max_records": 256,
                "max_message_bytes": 512,
                "operation_timeout_ms": 15000,
                "terminal_retention_ms": 5000,
                "max_terminal_compactions_per_cycle": 64,
            },
            "observation": {
                "max_models": 1,
                "bucket_count": 20,
                "bucket_width_ms": 100,
                "max_latency_samples_per_bucket": 4096,
                "forecast_horizon_ms": 2000,
                "forecast_headroom": 1.0,
            },
            "input_builder": {"max_pools": 1, "max_members": 64},
            "transports": {
                "native_timeout_ms": 1000,
                "native_max_channels": 64,
                "vllm_timeout_ms": 1000,
                "vllm_max_channels": 64,
                "vllm_max_response_bytes": 65536,
                "vllm_internal_api_token": INTERNAL_TOKEN,
            },
            "deployment": {
                "address": "127.0.0.1:18280",
                "timeout_ms": 12000,
                "max_response_bytes": 65536,
                "internal_api_token": INTERNAL_TOKEN,
            },
            "pools": [
                {
                    "provider": "VLLM_ASCEND",
                    "model_revision": MODEL_REVISION,
                    "role": "AGGREGATED",
                    "profile_digest": profile_digest,
                    "devices_per_replica": 1,
                    "instance_cost_per_hour": 1.0,
                    "load_warmup_p99_ms": 100,
                    "prefill_tokens_per_second_under_slo": 0.0,
                    "decode_tokens_per_second_under_slo": 0.0,
                    "requests_per_second_under_slo": 10.0,
                    "target_utilization": 0.8,
                    "min_replicas": 1,
                    "max_replicas": 3,
                    "failure_headroom_replicas": 0,
                    "ttft_slo_ms": 10000.0,
                    "tpot_slo_ms": 10000.0,
                    "priority": 100,
                    "slo_risk_score": 1.0,
                    "config_digest": "offline-v3-config-r1",
                    "external": {
                        "queue_depth": 0.0,
                        "kv_used_ratio": 0.0,
                        "full_cache_loss_cost": 0.0,
                        "confirmed_store_coverage": 1.0,
                        "out_of_distribution": False,
                    },
                }
            ],
        }
        placement_path = self.runtime_dir / "placement-v3.json"
        placement_path.write_text(json.dumps(placement, indent=2), encoding="utf-8")
        return provider_path, placement_path, str(profile_digest)

    def start_deployment_gateway(self, provider_config: Path) -> ManagedProcess:
        gateway = self._start(
            "deployment-gateway",
            [
                sys.executable,
                str(REPOSITORY_ROOT / "tests/e2e/mock_deployment_gateway.py"),
                "--port=18280",
                f"--etcd-namespace={self.namespace}",
                f"--provider-config={provider_config}",
                f"--internal-token={INTERNAL_TOKEN}",
                f"--repository-root={REPOSITORY_ROOT}",
            ],
        )
        wait_until(
            "deployment gateway",
            10,
            lambda: requests.get(
                "http://127.0.0.1:18280/health", timeout=0.5
            ).status_code
            == 200,
        )
        return gateway

    @staticmethod
    def gateway_state() -> dict[str, object]:
        response = requests.get("http://127.0.0.1:18280/debug/state", timeout=8.0)
        response.raise_for_status()
        return response.json()

    def _start_native_engine(
        self,
        label: str,
        incarnation: str,
        port: int,
        role: str,
        generation: int = 1,
    ) -> None:
        binary = self.build_dir / "tests/e2e/xllm_service_mock_native_engine"
        if not binary.is_file():
            raise GateFailure(f"missing mock Native Engine binary: {binary}")
        address = f"127.0.0.1:{port}"
        # Native InstanceMgr uses the registered instance name as the bRPC
        # dial target. Production Native Engine identities therefore use their
        # stable, dialable address; labels remain harness-only.
        engine_uid = address
        process = self._start(
            f"engine-{label}-r{generation}",
            [
                str(binary),
                f"--listen_address={address}",
                f"--engine_uid={engine_uid}",
                f"--incarnation_id={incarnation}",
                "--worker_threads=8",
                "--queue_capacity=2048",
                "--completion_delay_ms=100",
            ],
        )
        self.engine_processes[label] = process
        wait_until(
            f"Native Engine {label} generation {generation}",
            10,
            lambda p=process: p.process.poll() is None
            and requests.get(
                f"http://127.0.0.1:{port}/health", timeout=0.5
            ).status_code
            == 200,
        )
        lease = NativeEngineLease(
            "127.0.0.1:2379",
            self.namespace,
            engine_uid,
            incarnation,
            address,
            role,
            self.cached_master_urls,
            self.hbm,
            lambda p=process: p.send_signal(signal.SIGTERM),
        )
        lease.start()
        self.engine_leases[label] = lease

    def start_native_engines(self) -> None:
        definitions = [
            ("p-0", "p-0-inc-r1", 19000, "ENGINE_ROLE_PREFILL"),
            ("p-1", "p-1-inc-r1", 19001, "ENGINE_ROLE_PREFILL"),
            ("d-0", "d-0-inc-r1", 19100, "ENGINE_ROLE_DECODE"),
            ("d-1", "d-1-inc-r1", 19101, "ENGINE_ROLE_DECODE"),
        ]
        for label, incarnation, port, role in definitions:
            self._start_native_engine(label, incarnation, port, role)

        wait_until(
            "strict V2 registry, state, links, and service readiness",
            35,
            lambda: self.leader_url()
            and requests.get(self.leader_url() + "/readyz", timeout=0.5).status_code
            == 200,
        )

    def assert_phase(
        self,
        result: LoadResult,
        require_all: bool = True,
        minimum_routes: int = 2,
    ) -> None:
        self.report.phases.append(asdict(result))
        if require_all and result.failed:
            raise GateFailure(
                f"phase {result.phase} had {result.failed} failures: "
                f"{result.error_samples}"
            )
        if require_all and len(result.routes) < minimum_routes:
            raise GateFailure(
                f"phase {result.phase} exercised {len(result.routes)} routes, "
                f"required {minimum_routes}: {result.routes}"
            )

    def assert_bounded_abrupt_loss(
        self,
        result: LoadResult,
        max_failures: int,
        minimum_routes: int,
        recovery_duration_seconds: float,
        max_duration_seconds: float,
    ) -> None:
        self.report.phases.append(asdict(result))
        if result.succeeded == 0 or len(result.routes) < minimum_routes:
            raise GateFailure(
                f"phase {result.phase} lost serving progress: {result}"
            )
        if result.failed > max_failures:
            raise GateFailure(
                f"phase {result.phase} exceeded abrupt-loss failure bound "
                f"{max_failures}: {result.error_samples}"
            )
        if recovery_duration_seconds > max_duration_seconds:
            raise GateFailure(
                f"phase {result.phase} exceeded recovery-window bound "
                f"{max_duration_seconds}s: {recovery_duration_seconds}s"
            )
        if any(
            not is_expected_abrupt_loss_error(error)
            for error in result.error_samples
        ):
            raise GateFailure(
                f"phase {result.phase} returned an unexpected failure: "
                f"{result.error_samples}"
            )

    def stop_engine(self, label: str) -> None:
        lease = self.engine_leases.pop(label)
        process = self.engine_processes.pop(label)
        lease.stop()
        process.stop()

    def crash_engine(self, label: str) -> None:
        lease = self.engine_leases.pop(label)
        process = self.engine_processes.pop(label)
        process.send_signal(signal.SIGKILL)
        process.process.wait(timeout=3.0)
        lease.stop()

    def restart_native_engines_after_ownership_loss(self) -> dict[str, str]:
        restarted: dict[str, str] = {}
        for label in sorted(self.engine_leases):
            old_lease = self.engine_leases[label]
            if not old_lease.ownership_lost:
                continue
            role = old_lease.role
            port = int(old_lease.address.rsplit(":", 1)[1])
            old_lease.stop()
            self.engine_processes[label].stop()
            wait_until(
                f"old membership removal for {label}",
                10,
                lambda key=old_lease.membership_key: self.etcd.get(key) is None,
            )
            incarnation = f"{label}-inc-r2"
            self._start_native_engine(
                label, incarnation, port, role, generation=2
            )
            restarted[label] = incarnation
        return restarted

    def kill_leader(self) -> tuple[str, str]:
        old_address = self.master_rpc_address()
        if old_address is None:
            raise GateFailure("cannot identify current service leader")
        if old_address.endswith(":18889"):
            name = "master-a"
        elif old_address.endswith(":18899"):
            name = "master-b"
        else:
            raise GateFailure(f"unexpected master address: {old_address}")
        self.masters[name].send_signal(signal.SIGKILL)
        self.masters[name].process.wait(timeout=3.0)
        new_address = wait_until(
            "follower promotion",
            15,
            lambda: (
                address
                if (address := self.master_rpc_address())
                and address != old_address
                else None
            ),
        )
        return old_address, str(new_address)

    def collect_metrics(self) -> dict[str, object]:
        leader = self.leader_url()
        if leader is None:
            raise GateFailure("no leader for metrics collection")
        text = requests.get(leader + "/metrics", timeout=2.0).text
        evidence: dict[str, object] = {}
        for token in (
            "xllm_service_v2_output_sequence_total",
            'outcome="buffered_gap"',
            'outcome="duplicate"',
            "xllm_service_v2_active_requests",
            "xllm_service_v2_dispatched_requests",
            "xllm_service_v2_queued_requests",
        ):
            evidence[token] = token in text
        metrics_path = self.root / "metrics.prom"
        metrics_path.write_text(text, encoding="utf-8")
        if not all(evidence.values()):
            raise GateFailure(f"required V2 metric evidence missing: {evidence}")
        return evidence

    def scrape_v3_metrics(self, filename: str) -> str:
        leader = self.leader_url()
        if leader is None:
            raise GateFailure("no leader for V3 metrics collection")
        text = requests.get(leader + "/metrics", timeout=2.0).text
        (self.root / filename).write_text(text, encoding="utf-8")
        return text

    def metric_sample(
        self,
        metric_name: str,
        label_fragment: str = "",
    ) -> float:
        leader = self.leader_url()
        if leader is None:
            raise GateFailure(f"no leader while reading metric {metric_name}")
        text = requests.get(leader + "/metrics", timeout=1.0).text
        for line in text.splitlines():
            if not line.startswith(metric_name):
                continue
            if label_fragment and label_fragment not in line:
                continue
            try:
                return float(line.rsplit(" ", 1)[-1])
            except ValueError:
                continue
        return 0.0

    def wait_v3_runtime_resources_released(self, description: str) -> None:
        def released() -> dict[str, object] | None:
            state = self.gateway_state()
            resources = state["active_resources"]
            if all(
                int(resource["running"]) == 0
                and int(resource["simulated_hbm"]["used_blocks"]) == 0
                and int(resource["simulated_hbm"]["active_allocations"])
                == 0
                for resource in resources
            ):
                return state
            return None

        wait_until(
            description,
            6,
            released,
        )

    def run_v3_deadline_and_disconnect(
        self, client: V3LoadClient
    ) -> str:
        deadline_before = self.metric_sample(
            "xllm_service_v2_request_failure_total",
            'reason="EVENT_REASON_DEADLINE_EXCEEDED"',
        )
        deadline = client.run_deadline(
            "v3-inflight-deadline-cancel",
            requests_count=3,
            concurrency=3,
            remaining_deadline_ms=500,
        )
        self.report.phases.append(asdict(deadline))
        deadline_errors = " ".join(deadline.error_samples).lower()
        if (
            deadline.succeeded != 0
            or deadline.failed != 3
            or deadline.duration_seconds > 2.0
            or not any(
                token in deadline_errors
                for token in ("deadline", "expired", "504")
            )
        ):
            raise GateFailure(
                f"V3 in-flight deadline was not bounded and classified: {deadline}"
            )
        wait_until(
            "V3 deadline failure metric",
            3,
            lambda: self.metric_sample(
                "xllm_service_v2_request_failure_total",
                'reason="EVENT_REASON_DEADLINE_EXCEEDED"',
            )
            > deadline_before,
        )
        self.wait_v3_runtime_resources_released(
            "V3 deadline runtime and simulated HBM release"
        )

        disconnect_before = self.metric_sample(
            "xllm_service_v2_request_failure_total",
            'reason="EVENT_REASON_CANCELLED"',
        )
        leader = self.leader_url()
        if leader is None:
            raise GateFailure("no leader for V3 client disconnect")
        port = int(leader.rsplit(":", 1)[1])
        request_uid = f"v3-disconnect-{uuid.uuid4().hex}"
        body = json.dumps(
            {
                "model": MODEL_REVISION,
                "prompt": "__xllm_e2e_hold_3000ms__ disconnect",
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": True,
                "remaining_deadline_ms": 5000,
            },
            separators=(",", ":"),
        ).encode()
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
        connection.request(
            "POST",
            "/v1/completions",
            body=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "X-Request-Id": request_uid,
            },
        )
        try:
            wait_until(
                "V3 disconnected request reaches a runtime",
                2,
                lambda: any(
                    int(resource["running"]) > 0
                    for resource in self.gateway_state()["active_resources"]
                ),
            )
        finally:
            connection.close()
        wait_until(
            "V3 client disconnect cancellation metric",
            3,
            lambda: self.metric_sample(
                "xllm_service_v2_request_failure_total",
                'reason="EVENT_REASON_CANCELLED"',
            )
            > disconnect_before,
        )
        wait_until(
            "V3 disconnected Service request release",
            2,
            lambda: self.metric_sample("xllm_service_v2_active_requests") == 0,
        )
        self.wait_v3_runtime_resources_released(
            "V3 disconnected runtime and simulated HBM release"
        )
        recovery = client.run("v3-after-deadline-disconnect", 60, 24)
        self.assert_phase(recovery, minimum_routes=3)
        metrics = self.scrape_v3_metrics("metrics-deadline-disconnect.prom")
        self.report.faults.append(
            {
                "fault": "v3_deadline_and_client_disconnect",
                "deadline_failures": deadline.failed,
                "deadline_duration_seconds": deadline.duration_seconds,
                "disconnect_request_uid": request_uid,
                "post_cancel_success": recovery.succeeded,
                "runtime_and_simulated_hbm_released": True,
            }
        )
        return metrics

    def execute_agent_lifecycle(
        self,
        replica: dict[str, object],
        action: str,
        desired_generation: int,
    ) -> dict[str, object]:
        leader_incarnation, leader_epoch = self.master_identity()
        command = {
            "schema_version": 1,
            "operation_id": f"offline-e2e-{action.lower()}-{uuid.uuid4().hex}",
            "leader_incarnation": leader_incarnation,
            "leader_epoch": leader_epoch,
            "desired_generation": desired_generation,
            "engine_uid": str(replica["engine_uid"]),
            "engine_incarnation": str(replica["engine_incarnation"]),
            "action": action,
        }
        response = requests.post(
            f"http://{replica['engine_uid']}/v1/internal/lifecycle/execute",
            json=command,
            headers={"X-Internal-Token": INTERNAL_TOKEN},
            timeout=2.0,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise GateFailure(f"invalid Agent lifecycle response: {payload}")
        return payload

    @staticmethod
    def set_gateway_heartbeat_forwarding(enabled: bool) -> dict[str, object]:
        action = "resume-heartbeats" if enabled else "pause-heartbeats"
        response = requests.post(
            f"http://127.0.0.1:18280/debug/{action}",
            headers={"X-Internal-Token": INTERNAL_TOKEN},
            timeout=2.0,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise GateFailure(f"invalid heartbeat forwarding response: {payload}")
        return payload

    def v3_drain_retry_observed(self) -> str | None:
        leader = self.leader_url()
        if leader is None:
            return None
        text = requests.get(leader + "/metrics", timeout=1.0).text
        for line in text.splitlines():
            if (
                line.startswith("xllm_service_v2_attempt_retries_total{")
                and 'outcome="VLLM_DRAIN_RESELECTED"' in line
                and float(line.rsplit(" ", 1)[-1]) >= 1.0
            ):
                return text
        return None

    def collect_v3_metrics(self, historical: list[str]) -> dict[str, object]:
        current = self.scrape_v3_metrics("metrics.prom")
        text = "\n".join(historical + [current])
        evidence: dict[str, object] = {}
        for token in (
            "xllm_service_v3_placement_leader",
            "xllm_service_v3_placement_cycles_total",
            "xllm_service_v3_placement_observations_total",
            "xllm_service_v3_placement_recommendations_total",
            'action="SCALE_UP"',
            'action="SCALE_DOWN"',
            "xllm_service_v3_placement_operations_total",
            'outcome="ADDED"',
            'outcome="DRIVEN"',
            "xllm_service_v3_placement_operation_duration_milliseconds",
            "xllm_service_v2_attempt_retries_total",
            'outcome="VLLM_DRAIN_RESELECTED"',
            'reason="EVENT_REASON_CANCELLED"',
            'reason="EVENT_REASON_DEADLINE_EXCEEDED"',
        ):
            evidence[token] = token in text
        if not all(evidence.values()):
            raise GateFailure(f"required V3 metric evidence missing: {evidence}")
        return evidence

    def run_v3_drain_race(self, client: V3LoadClient) -> str:
        """Force select-before-drain and require transparent bounded retry."""

        hold_started = time.monotonic()
        hold_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="v3-drain-hold"
        )
        hold_futures = [
            hold_pool.submit(client._one, index, True) for index in range(3)
        ]
        race_pool: concurrent.futures.ThreadPoolExecutor | None = None
        heartbeat_forwarding_paused = False
        target: dict[str, object] | None = None
        drain_cancelled = False
        try:
            state = wait_until(
                "an active VLLM attempt on every replica",
                5,
                lambda: (
                    snapshot
                    if len(snapshot := self.gateway_state()["active_resources"])
                    == 3
                    and all(int(resource["running"]) > 0 for resource in snapshot)
                    else None
                ),
            )
            assert isinstance(state, list)
            replicas = sorted(
                self.gateway_state()["replicas"], key=lambda item: item["ordinal"]
            )
            target = replicas[0]
            paused = self.set_gateway_heartbeat_forwarding(False)
            heartbeat_forwarding_paused = True
            if paused.get("heartbeat_forwarding_enabled") is not False:
                raise GateFailure(f"failed to pause Agent heartbeats: {paused}")
            begin = self.execute_agent_lifecycle(
                target,
                "PROVIDER_LIFECYCLE_ACTION_BEGIN_DRAIN",
                desired_generation=1,
            )
            if (
                begin.get("code") != "PROVIDER_LIFECYCLE_CODE_IN_PROGRESS"
                or begin.get("lifecycle") != "ENGINE_LIFECYCLE_DRAINING"
            ):
                raise GateFailure(
                    f"Agent did not enter a cancellable drain: {begin}"
                )

            race_started = time.monotonic()
            race_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=12, thread_name_prefix="v3-drain-race"
            )
            race_futures = [
                race_pool.submit(client._one, index) for index in range(12)
            ]
            retry_metrics = wait_until(
                "select-before-drain bounded retry metric",
                2,
                self.v3_drain_retry_observed,
            )
            assert isinstance(retry_metrics, str)
            cancel = self.execute_agent_lifecycle(
                target,
                "PROVIDER_LIFECYCLE_ACTION_CANCEL_DRAIN",
                desired_generation=2,
            )
            if (
                cancel.get("code") != "PROVIDER_LIFECYCLE_CODE_SUCCEEDED"
                or cancel.get("lifecycle") != "ENGINE_LIFECYCLE_READY"
            ):
                raise GateFailure(f"Agent drain cancellation failed: {cancel}")
            drain_cancelled = True
            resumed = self.set_gateway_heartbeat_forwarding(True)
            heartbeat_forwarding_paused = False
            if resumed.get("heartbeat_forwarding_enabled") is not True:
                raise GateFailure(f"failed to resume Agent heartbeats: {resumed}")

            race_result = V3LoadClient.summarize(
                "v3-select-before-drain-retry",
                race_started,
                [future.result(timeout=10) for future in race_futures],
            )
            self.assert_phase(race_result, minimum_routes=2)
            hold_result = V3LoadClient.summarize(
                "v3-inflight-survives-drain-cancel",
                hold_started,
                [future.result(timeout=10) for future in hold_futures],
            )
            self.assert_phase(hold_result, minimum_routes=3)
            (self.root / "metrics-drain-race.prom").write_text(
                retry_metrics, encoding="utf-8"
            )
            self.report.faults.append(
                {
                    "fault": "select_before_agent_drain",
                    "target": target["engine_uid"],
                    "begin_code": begin["code"],
                    "cancel_code": cancel["code"],
                    "heartbeat_propagation_paused": True,
                    "transparent_retries": 1,
                    "request_failures": race_result.failed,
                    "inflight_failures": hold_result.failed,
                }
            )
            return retry_metrics
        finally:
            if target is not None and not drain_cancelled:
                try:
                    self.execute_agent_lifecycle(
                        target,
                        "PROVIDER_LIFECYCLE_ACTION_CANCEL_DRAIN",
                        desired_generation=2,
                    )
                except (GateFailure, requests.RequestException):
                    pass
            if heartbeat_forwarding_paused:
                try:
                    self.set_gateway_heartbeat_forwarding(True)
                except (GateFailure, requests.RequestException):
                    pass
            if race_pool is not None:
                race_pool.shutdown(wait=True)
            hold_pool.shutdown(wait=True)

    def etcd_prefix(self, prefix: str) -> dict[str, str]:
        if not prefix:
            raise ValueError("etcd prefix must not be empty")
        prefix_bytes = prefix.encode()
        range_end = prefix_bytes[:-1] + bytes([prefix_bytes[-1] + 1])
        response = requests.post(
            "http://127.0.0.1:2379/v3/kv/range",
            json={
                "key": base64.b64encode(prefix_bytes).decode(),
                "range_end": base64.b64encode(range_end).decode(),
            },
            timeout=2.0,
        )
        response.raise_for_status()
        return {
            base64.b64decode(item["key"]).decode(): base64.b64decode(
                item.get("value", "")
            ).decode()
            for item in response.json().get("kvs", [])
        }

    def run_v2(self) -> None:
        tokenizer = self.write_tokenizer()
        etcd_process = self.start_etcd()
        self.start_masters(tokenizer)
        self.start_native_engines()
        client = V2LoadClient(self.cached_master_urls, self.hbm)

        baseline_count = 120 if self.args.mode == "smoke" else 1200
        concurrency = 24
        self.assert_phase(client.run("v2-baseline", baseline_count, concurrency))

        self.assert_phase(
            client.run_streaming(
                "v2-openai-sse",
                16 if self.args.mode == "smoke" else 160,
                8,
            )
        )

        inflight_deadline = client.run_deadline(
            "v2-inflight-deadline-cancel",
            requests_count=1,
            concurrency=1,
            remaining_deadline_ms=50,
        )
        self.report.phases.append(asdict(inflight_deadline))
        inflight_deadline_errors = " ".join(
            inflight_deadline.error_samples
        ).upper()
        if (
            inflight_deadline.succeeded != 0
            or inflight_deadline.failed != 1
            or inflight_deadline.duration_seconds > 0.25
            or "DEADLINE" not in inflight_deadline_errors
        ):
            raise GateFailure(
                "in-flight V2 deadline was not cancelled and classified: "
                f"{inflight_deadline}"
            )
        self.hbm.assert_converged()
        self.assert_phase(
            client.run("v2-after-inflight-deadline", 16, 8),
            require_all=True,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            active = pool.submit(client.run, "v2-deadline-work-ahead", 8, 8)
            time.sleep(0.03)
            deadline = client.run_deadline(
                "v2-unsatisfiable-deadline",
                8 if self.args.mode == "smoke" else 64,
                8,
                25,
            )
            self.assert_phase(active.result(timeout=10))
        self.report.phases.append(asdict(deadline))
        deadline_errors = " ".join(deadline.error_samples).upper()
        if deadline.succeeded or "DEADLINE" not in deadline_errors:
            raise GateFailure(
                "work-ahead deadline load did not fail closed with a deadline "
                f"classification: {deadline}"
            )

        overload = client.run(
            "v2-bounded-overload",
            96 if self.args.mode == "smoke" else 960,
            64,
        )
        self.report.phases.append(asdict(overload))
        overload_errors = " ".join(overload.error_samples).upper()
        if overload.succeeded == 0 or overload.failed == 0 or (
            "QUEUE" not in overload_errors and "CAPACITY" not in overload_errors
        ):
            raise GateFailure(
                "bounded overload did not preserve both progress and structured "
                f"backpressure: {overload}"
            )
        recovery_started = time.monotonic()
        wait_until(
            "V2 observation/readiness recovery after overload",
            3,
            lambda: self.leader_url()
            and requests.get(
                self.leader_url() + "/readyz", timeout=0.5
            ).status_code
            == 200,
        )
        readiness_recovery_seconds = time.monotonic() - recovery_started
        self.assert_phase(client.run("v2-overload-recovery", 32, 8))
        self.report.faults.append(
            {
                "fault": "bounded_overload_and_observation_recovery",
                "progress": overload.succeeded,
                "structured_backpressure": overload.failed,
                "readiness_recovery_seconds": readiness_recovery_seconds,
                "post_recovery_success": 32,
            }
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            inflight = pool.submit(
                client.run,
                "v2-inflight-prefill-sigkill",
                64 if self.args.mode == "smoke" else 640,
                32,
            )
            time.sleep(0.03)
            self.crash_engine("p-1")
        wait_until(
            "failed Prefill membership removal",
            10,
            lambda: self.etcd.get(
                f"/{self.namespace}/XLLM:PREFILL:127.0.0.1:19001"
            )
            is None,
        )
        self.assert_phase(inflight.result(timeout=30), require_all=True)
        after_engine_fault = client.run(
            "v2-after-prefill-loss", baseline_count // 2, concurrency
        )
        self.assert_phase(after_engine_fault, require_all=True)
        self.report.faults.append(
            {
                "fault": "inflight_prefill_sigkill_and_lease_loss",
                "target": "p-1",
                "inflight_success": inflight.result().succeeded,
                "post_fault_success": after_engine_fault.succeeded,
            }
        )

        etcd_process.send_signal(signal.SIGSTOP)
        try:
            during_etcd_stall = client.run(
                "v2-etcd-short-stall-data-plane",
                60 if self.args.mode == "smoke" else 120,
                concurrency,
            )
        finally:
            etcd_process.send_signal(signal.SIGCONT)
        self.assert_phase(during_etcd_stall, require_all=True)
        if during_etcd_stall.duration_seconds >= 2.0:
            raise GateFailure(
                "short etcd outage accidentally crossed the ownership "
                f"window: {during_etcd_stall.duration_seconds:.3f}s"
            )
        self.report.faults.append(
            {
                "fault": "etcd_short_sigstop_within_ownership_grace",
                "duration_seconds": during_etcd_stall.duration_seconds,
                "data_plane_success": during_etcd_stall.succeeded,
            }
        )

        # Readiness remains true throughout a bounded Registry outage and is
        # therefore not proof that either the Service or Engine ownership
        # lease has actually been refreshed. Starting the long outage against
        # the residual TTL from the short outage makes the nominal grace phase
        # cross the lease boundary nondeterministically. Require fresh
        # ownership evidence before measuring the independent long outage.
        short_recovery_started = time.monotonic()
        wait_until(
            "Engine lease refresh after short etcd recovery",
            5,
            lambda: all(
                lease.last_confirmed_lease > short_recovery_started
                for lease in self.engine_leases.values()
            ),
        )
        master_key = f"/{self.namespace}/XLLM:SERVICE:MASTER"
        wait_until(
            "Service leader lease refresh after short etcd recovery",
            5,
            lambda: self._etcd_lease_ttl_for_key(master_key) >= 2,
        )
        wait_until(
            "service readiness after short etcd recovery",
            10,
            lambda: self.leader_url()
            and requests.get(
                self.leader_url() + "/readyz", timeout=0.5
            ).status_code
            == 200,
        )

        long_outage_started = time.monotonic()
        fail_closed_status = 0
        direct_failure_result: tuple[bool, float, str] = (True, 0.0, "")
        fail_closed_result: tuple[bool, float, str] = (True, 0.0, "")
        etcd_process.send_signal(signal.SIGSTOP)
        try:
            grace_load = client.run(
                "v2-etcd-long-stall-grace-data-plane",
                max(30, baseline_count // 4),
                concurrency,
            )
            remaining = 5.25 - (time.monotonic() - long_outage_started)
            if remaining > 0:
                time.sleep(remaining)
            wait_until(
                "Engine self-fencing after lease ownership loss",
                3,
                lambda: all(
                    lease.ownership_lost
                    for lease in self.engine_leases.values()
                ),
            )
            direct_failure_result = client._one(-2)
            if direct_failure_result[0]:
                raise GateFailure(
                    "request executed after all Engine processes self-fenced"
                )
            def fail_closed_readiness_status() -> int:
                non_ready_statuses: list[int] = []
                for service_url in self.cached_master_urls():
                    try:
                        status = requests.get(
                            service_url + "/readyz", timeout=0.5
                        ).status_code
                    except requests.RequestException:
                        continue
                    if status == 200:
                        return 0
                    non_ready_statuses.append(status)
                return non_ready_statuses[0] if non_ready_statuses else 0

            fail_closed_status = int(
                wait_until(
                    "Service fail-closed after Engine ownership expiry",
                    # Engine ownership and the Service leader lease are
                    # independent clocks. The Engine must self-fence first;
                    # admission then closes no later than the Service lease or
                    # STATE_BLIND grace. Keep this wait above the configured
                    # lease boundary without weakening the 503 assertion.
                    8,
                    fail_closed_readiness_status,
                )
            )
            fail_closed_result = client._one(-1)
        finally:
            etcd_process.send_signal(signal.SIGCONT)
        self.assert_phase(grace_load, require_all=True)
        if fail_closed_result[0] or "SERVICE_NOT_READY" not in fail_closed_result[2]:
            raise GateFailure(
                "long etcd outage did not fail closed with a structured "
                f"response: readyz={fail_closed_status}, "
                f"request={fail_closed_result}"
            )

        wait_until("master re-election after long etcd outage", 15, self.master_rpc_address)
        restarted = self.restart_native_engines_after_ownership_loss()
        if set(restarted) != set(self.engine_leases):
            raise GateFailure(
                "not every surviving Engine restarted after ownership "
                f"loss: restarted={restarted}"
            )

        def engine_memberships_recovered() -> bool:
            return all(
                self.etcd.get(lease.membership_key) is not None
                for lease in self.engine_leases.values()
            )

        wait_until(
            "surviving Engine lease re-registration",
            15,
            engine_memberships_recovered,
        )
        wait_until(
            "service readiness after etcd recovery",
            35,
            lambda: self.leader_url()
            and requests.get(self.leader_url() + "/readyz", timeout=0.5).status_code
            == 200,
        )
        self.report.faults.append(
            {
                "fault": "etcd_long_outage_self_fence_and_reincarnation",
                "duration_seconds": time.monotonic() - long_outage_started,
                "direct_failure_request": direct_failure_result[2],
                "fail_closed_readyz_status": fail_closed_status,
                "fail_closed_request": fail_closed_result[2],
                "restarted_incarnations": restarted,
            }
        )

        old_leader, new_leader = self.kill_leader()
        wait_until(
            "promoted service readiness",
            20,
            lambda: self.leader_url()
            and requests.get(self.leader_url() + "/readyz", timeout=0.5).status_code
            == 200,
        )
        after_failover = client.run(
            "v2-after-master-failover", baseline_count, concurrency
        )
        self.assert_phase(after_failover, require_all=True)
        self.report.faults.append(
            {
                "fault": "service_leader_sigkill",
                "old_leader": old_leader,
                "new_leader": new_leader,
                "post_failover_success": after_failover.succeeded,
            }
        )

        self.hbm.assert_converged()
        self.report.resource_evidence = self.hbm.snapshot()
        self.report.metric_evidence = self.collect_metrics()

    def run_v3(self) -> None:
        tokenizer = self.write_tokenizer()
        provider_config, placement_config, profile_digest = self.write_v3_configs()
        self.start_etcd()
        self.start_deployment_gateway(provider_config)
        self.start_masters(tokenizer, placement_config)

        wait_until(
            "initial V3 Agent replica",
            35,
            lambda: (
                state
                if int((state := self.gateway_state())["active_replicas"]) == 1
                else None
            ),
        )
        wait_until(
            "lost CREATE response resolved by query",
            20,
            lambda: (
                state
                if int((state := self.gateway_state())["query_count"]) >= 1
                else None
            ),
        )
        wait_until(
            "strict aggregated service readiness",
            25,
            lambda: self.leader_url()
            and requests.get(self.leader_url() + "/readyz", timeout=0.5).status_code
            == 200,
        )

        client = V3LoadClient(self.cached_master_urls)
        baseline = client.run("v3-min-replica-baseline", 1, 1)
        self.assert_phase(baseline, minimum_routes=1)
        self.assert_phase(
            client.run_streaming(
                "v3-openai-sse",
                8 if self.args.mode == "smoke" else 80,
                8,
            ),
            minimum_routes=1,
        )

        high_count = 180 if self.args.mode == "smoke" else 1800
        concurrency = 32 if self.args.mode == "smoke" else 64
        steady_concurrency = 32
        # A strict at-most-once attempt cannot be transparently replayed after
        # SIGKILL when the Agent that owned its ledger is gone.  Keep the
        # abrupt-loss wave at three requests per physical replica plus one
        # stale-routing race, so the four-failure hard bound is a deterministic
        # blast-radius assertion rather than an accidental function of the
        # unrelated steady-state load concurrency.
        abrupt_fault_concurrency = 9
        high_load = client.run("v3-high-load-scale-up-signal", high_count, concurrency)
        if self.args.mode == "smoke":
            self.assert_phase(high_load, minimum_routes=1)
        else:
            self.report.phases.append(asdict(high_load))
            high_load_errors = " ".join(high_load.error_samples).upper()
            if (
                high_load.succeeded < 180
                or high_load.failed == 0
                or "QUEUE_CAPACITY_EXHAUSTED" not in high_load_errors
            ):
                raise GateFailure(
                    "V3 pre-scale pressure did not preserve progress and "
                    f"structured backpressure: {high_load}"
                )
            self.report.faults.append(
                {
                    "fault": "v3_pre_scale_bounded_overload",
                    "progress": high_load.succeeded,
                    "structured_backpressure": high_load.failed,
                }
            )
        wait_until(
            "V3 scale-up to three strict Agent replicas",
            35,
            lambda: (
                state
                if int((state := self.gateway_state())["active_replicas"]) == 3
                else None
            ),
        )
        time.sleep(0.5)
        post_scale = client.run(
            "v3-after-scale-up",
            180 if self.args.mode == "smoke" else 1800,
            32,
        )
        self.assert_phase(post_scale, minimum_routes=2)

        cancellation_metrics = self.run_v3_deadline_and_disconnect(client)

        scale_up_state = self.gateway_state()
        if (
            int(scale_up_state["active_replicas"]) != 3
            or int(scale_up_state["max_active_replicas"]) != 3
        ):
            raise GateFailure(
                f"V3 scale-up did not converge within max replicas: {scale_up_state}"
            )
        drain_retry_metrics = self.run_v3_drain_race(client)

        def restart_component(
            component: str,
        ) -> tuple[requests.Response, float]:
            started = time.monotonic()
            response = requests.post(
                f"http://127.0.0.1:18280/debug/restart-{component}",
                headers={"X-Internal-Token": INTERNAL_TOKEN},
                timeout=20.0,
            )
            return response, time.monotonic() - started

        for component in ("agent", "runtime"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fault = pool.submit(restart_component, component)
                during_restart = client.run(
                    f"v3-{component}-only-sigkill-restart",
                    600 if self.args.mode == "smoke" else 2400,
                    abrupt_fault_concurrency,
                )
                fault_response, recovery_duration_seconds = fault.result(
                    timeout=25.0
                )
            fault_response.raise_for_status()
            self.assert_bounded_abrupt_loss(
                during_restart,
                max_failures=4,
                minimum_routes=2,
                recovery_duration_seconds=recovery_duration_seconds,
                max_duration_seconds=5.0,
            )
            # Agent readiness changes before its next EngineState heartbeat is
            # visible at Service.  Allow several 200ms publish periods, then
            # require the recovered route to carry real traffic below.
            time.sleep(1.0)
            recovery = client.run(
                f"v3-after-{component}-only-restart",
                120 if self.args.mode == "smoke" else 1200,
                steady_concurrency,
            )
            self.assert_phase(recovery, minimum_routes=3)
            component_state = self.gateway_state()
            if int(component_state["max_active_replicas"]) > 3:
                raise GateFailure(
                    f"V3 {component} restart exceeded device capacity: "
                    f"{component_state}"
                )
            if component == "runtime" and int(
                component_state["recovered_create_reconciliations"]
            ) < 1:
                raise GateFailure(
                    "Runtime-only restart did not exercise deployment "
                    f"inventory reconciliation: {component_state}"
                )
            self.report.faults.append(
                {
                    "fault": f"vllm_{component}_only_sigkill_restart",
                    "restarted_replica": fault_response.json(),
                    "during_fault_success": during_restart.succeeded,
                    "bounded_transport_failures": during_restart.failed,
                    "recovery_duration_seconds": recovery_duration_seconds,
                    "post_restart_success": recovery.succeeded,
                    "max_active_replicas": component_state[
                        "max_active_replicas"
                    ],
                    "recovered_create_reconciliations": component_state[
                        "recovered_create_reconciliations"
                    ],
                }
            )

        replacement_started = time.monotonic()
        crashed = requests.post(
            "http://127.0.0.1:18280/debug/crash-replica",
            headers={"X-Internal-Token": INTERNAL_TOKEN},
            timeout=3.0,
        )
        crashed.raise_for_status()
        crashed_replica = crashed.json()

        def wait_for_replacement() -> tuple[dict[str, object], float]:
            state = wait_until(
                "V3 replacement after abrupt Agent and Runtime loss",
                35,
                lambda: (
                    current
                    if int(
                        (current := self.gateway_state())["active_replicas"]
                    )
                    == 3
                    else None
                ),
            )
            assert isinstance(state, dict)
            return state, time.monotonic() - replacement_started

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            replacement = pool.submit(wait_for_replacement)
            during_replacement = client.run(
                "v3-runtime-agent-sigkill-replacement",
                180 if self.args.mode == "smoke" else 1800,
                abrupt_fault_concurrency,
            )
            replacement_state, replacement_duration_seconds = (
                replacement.result(timeout=40.0)
            )
        self.assert_bounded_abrupt_loss(
            during_replacement,
            max_failures=4,
            minimum_routes=2,
            recovery_duration_seconds=replacement_duration_seconds,
            max_duration_seconds=5.0,
        )
        if int(replacement_state["max_active_replicas"]) > 3:
            raise GateFailure(
                f"V3 replacement exceeded device capacity: {replacement_state}"
            )
        after_replacement = client.run(
            "v3-after-runtime-agent-replacement",
            120 if self.args.mode == "smoke" else 1200,
            steady_concurrency,
        )
        self.assert_phase(after_replacement, minimum_routes=3)
        self.report.faults.append(
            {
                "fault": "vllm_agent_runtime_sigkill",
                "crashed_replica": crashed_replica,
                "during_fault_success": during_replacement.succeeded,
                "bounded_transport_failures": during_replacement.failed,
                "recovery_duration_seconds": replacement_duration_seconds,
                "post_replacement_success": after_replacement.succeeded,
                "replacement_active_replicas": replacement_state[
                    "active_replicas"
                ],
            }
        )
        before_failover_metrics = self.scrape_v3_metrics(
            "metrics-before-failover.prom"
        )
        if 'action="SCALE_UP"' not in before_failover_metrics:
            raise GateFailure("pre-failover metrics did not retain SCALE_UP evidence")
        self.report.faults.append(
            {
                "fault": "deployment_create_response_lost_after_side_effect",
                "lost_response_count": scale_up_state["lost_create_response_count"],
                "query_count": scale_up_state["query_count"],
                "active_replicas": scale_up_state["active_replicas"],
            }
        )

        old_leader, new_leader = self.kill_leader()
        wait_until(
            "promoted V3 service readiness",
            25,
            lambda: self.leader_url()
            and requests.get(self.leader_url() + "/readyz", timeout=0.5).status_code
            == 200,
        )
        wait_until(
            "promoted V3 placement leadership",
            20,
            lambda: "xllm_service_v3_placement_leader 1"
            in requests.get(self.leader_url() + "/metrics", timeout=1.0).text,
        )
        after_failover = client.run(
            "v3-after-master-failover",
            180 if self.args.mode == "smoke" else 1800,
            steady_concurrency,
        )
        self.assert_phase(after_failover, minimum_routes=2)
        failover_state = self.gateway_state()
        if (
            int(failover_state["active_replicas"]) != 3
            or int(failover_state["max_active_replicas"]) > 3
        ):
            raise GateFailure(
                "V3 exceeded the configured replica bound during service "
                f"failover: {failover_state}"
            )
        self.report.faults.append(
            {
                "fault": "service_leader_sigkill_with_v3_state",
                "old_leader": old_leader,
                "new_leader": new_leader,
                "post_failover_success": after_failover.succeeded,
                "active_replicas": failover_state["active_replicas"],
                "max_active_replicas": failover_state["max_active_replicas"],
            }
        )

        low_load = client.run_paced(
            "v3-low-load-scale-down",
            max_requests=150,
            interval_seconds=0.2,
            stop_predicate=lambda: int(self.gateway_state()["active_replicas"]) == 1,
        )
        self.assert_phase(low_load, minimum_routes=1)
        final_state = self.gateway_state()
        if int(final_state["active_replicas"]) != 1:
            raise GateFailure(f"V3 did not scale down to min replicas: {final_state}")
        action_counts = final_state["action_counts"]
        if (
            int(action_counts.get("CREATE", 0)) != 4
            or int(action_counts.get("TERMINATE", 0)) != 2
            or int(final_state["max_active_replicas"]) != 3
            or int(final_state["lost_create_response_count"]) != 1
            or int(final_state["query_count"]) < 1
            or int(final_state["drain_proofs"]) < 2
            or int(final_state["heartbeat_forwarded"]) < 1
        ):
            raise GateFailure(f"V3 actuator/lifecycle evidence incomplete: {final_state}")

        resources = list(final_state["active_resources"]) + list(
            final_state["terminated_resources"]
        )
        exercised = 0
        for resource in resources:
            hbm = resource.get("simulated_hbm")
            if not isinstance(hbm, dict):
                raise GateFailure(f"missing per-replica HBM evidence: {resource}")
            if (
                int(hbm["used_blocks"]) != 0
                or int(hbm["active_allocations"]) != 0
                or not bool(hbm["tensor_zero"])
            ):
                raise GateFailure(f"per-replica HBM did not converge: {resource}")
            if int(hbm["high_watermark_blocks"]) > 0:
                exercised += 1
        if exercised < 2:
            raise GateFailure(
                f"load did not exercise multiple simulated HBM replicas: {resources}"
            )

        placement_snapshot = {}
        for logical_prefix in (
            "XLLM:PLACEMENT:DESIRED/",
            "XLLM:PLACEMENT:COMMAND/",
            "XLLM:PLACEMENT:STATUS/",
        ):
            placement_snapshot.update(
                self.etcd_prefix(f"/{self.namespace}/{logical_prefix}")
            )
        (self.root / "placement-etcd-snapshot.json").write_text(
            json.dumps(placement_snapshot, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if not any("XLLM:PLACEMENT:DESIRED/" in key for key in placement_snapshot):
            raise GateFailure("durable V3 desired state was not found in etcd")

        self.report.resource_evidence = {
            "profile_digest": profile_digest,
            "replicas_exercised": exercised,
            "active_resources": final_state["active_resources"],
            "terminated_resources": final_state["terminated_resources"],
            "drain_proofs": final_state["drain_proofs"],
        }
        self.report.metric_evidence = self.collect_v3_metrics(
            [
                cancellation_metrics,
                drain_retry_metrics,
                before_failover_metrics,
            ]
        )

    def cleanup(self) -> None:
        for lease in list(self.engine_leases.values()):
            lease.stop()
        self.engine_leases.clear()
        for process in reversed(self.processes):
            process.stop()

    def run(self) -> Path:
        report_path = self.root / "report.json"
        try:
            if self.args.scenario == "v2":
                self.run_v2()
            elif self.args.scenario == "v3":
                self.run_v3()
            else:
                raise GateFailure("combined scenarios must use isolated clusters")
            self.report.passed = True
            return report_path
        finally:
            self.cleanup()
            report_path.write_text(
                json.dumps(asdict(self.report), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            latest = Path(self.args.artifact_dir) / "latest-report.json"
            latest.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("v2", "v3", "all"), default="all")
    parser.add_argument("--mode", choices=("smoke", "stress"), default="smoke")
    parser.add_argument(
        "--build-dir", default="build/local-arm64-Debug-xllm-override"
    )
    parser.add_argument("--artifact-dir", default="build/e2e-artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenarios = ("v2", "v3") if args.scenario == "all" else (args.scenario,)
    reports = []
    for scenario in scenarios:
        scenario_args = argparse.Namespace(**vars(args))
        scenario_args.scenario = scenario
        gate = ClusterGate(scenario_args)
        try:
            reports.append(str(gate.run()))
        except Exception as error:
            logger.error(
                "OFFLINE_E2E_GATE_FAILED scenario=%s error=%s artifacts=%s",
                scenario,
                error,
                gate.root,
            )
            return 1
    logger.info("OFFLINE_E2E_GATE_PASSED reports=%s", ",".join(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
