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
"""vLLM sidecar entrypoint: health-gated etcd lease registration loop.

Run alongside a vLLM server:

    python -m vllm_sidecar.sidecar \
        --etcd-endpoints 127.0.0.1:2379 \
        --vllm-url http://127.0.0.1:18000 \
        --register-addr 127.0.0.1:18000

See README.md for the full flag list and the lifecycle contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import threading
import time
import uuid
from types import FrameType
from urllib.parse import urlsplit

import requests

from scripts.logger import configure_logging, logger

from .agent import AgentRuntime, split_address, valid_internal_token
from .descriptor import build_provider_descriptor, load_provider_config
from .etcd_registry import EtcdGatewayClient, EtcdError
from .health import VllmHealthProbe
from .meta import InstanceType, build_instance_key, build_instance_meta
from .metrics import VllmMetricsScraper


class Sidecar:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        _validate_runtime_args(args)
        self._instance_type = InstanceType[args.instance_type]
        provider_config_path = getattr(args, "provider_config", "")
        self._provider_config = (
            load_provider_config(provider_config_path) if provider_config_path else None
        )
        if self._provider_config is not None and (
            not valid_internal_token(args.internal_token)
        ):
            raise ValueError(
                "strict Agent requires a bounded printable-ASCII --internal-token"
            )
        self._agent = None
        if self._provider_config is not None:
            if self._instance_type != InstanceType.DEFAULT:
                raise ValueError("strict aggregated Agent must use DEFAULT type")
            agent_listen = getattr(args, "agent_listen", "")
            if not agent_listen:
                raise ValueError("--agent-listen is required with --provider-config")
            _, listen_port = split_address(agent_listen)
            register_host, register_port = split_address(args.register_addr)
            if register_host == "0.0.0.0":
                raise ValueError("strict --register-addr must be routable")
            if listen_port != register_port:
                raise ValueError(
                    "strict Agent listen and registered ports must match"
                )
            build_provider_descriptor(
                self._provider_config,
                args.register_addr,
                "configuration-validation",
                args.register_addr,
            )
            self._agent = AgentRuntime(
                listen_address=agent_listen,
                upstream_url=args.vllm_url,
                max_attempt_records=getattr(args, "max_attempt_records", 8192),
                max_cancel_fences=getattr(args, "max_cancel_fences", 8192),
                terminal_ttl_seconds=getattr(args, "attempt_terminal_ttl", 60.0),
                negative_fence_ttl_seconds=getattr(
                    args, "negative_fence_ttl", 60.0
                ),
                max_lifecycle_records=getattr(
                    args, "max_lifecycle_records", 4096
                ),
                max_inflight_requests=getattr(
                    args, "agent_max_inflight_requests", 256
                ),
                max_request_body_bytes=getattr(
                    args, "agent_max_request_body_bytes", 8 * 1024 * 1024
                ),
                inflight_body_capacity_bytes=getattr(
                    args, "agent_inflight_body_capacity_bytes", 64 * 1024 * 1024
                ),
                connect_timeout_seconds=getattr(args, "agent_connect_timeout", 1.0),
                ingress_timeout_seconds=getattr(args, "agent_ingress_timeout", 5.0),
                internal_token=args.internal_token,
            )
            if register_port == 0:
                _, bound_port = split_address(self._agent.listen_address)
                args.register_addr = f"{register_host}:{bound_port}"
        self._etcd = EtcdGatewayClient(
            args.etcd_endpoints,
            username=args.etcd_username,
            password=args.etcd_password,
            timeout=args.etcd_timeout,
        )
        self._health = VllmHealthProbe(args.vllm_url, timeout=args.health_timeout)
        self._key = build_instance_key(
            args.register_addr, self._instance_type, args.etcd_namespace
        )
        self._stop = threading.Event()
        self._lease_id = None
        self._incarnation_id = None
        # heartbeat (LoadMetrics/LatencyMetrics) -> master HTTP endpoint
        self._hb_url = args.xllm_service_url.rstrip("/") + "/v1/internal/heartbeat"
        topology = (
            self._provider_config.get("topology", {}) if self._provider_config else {}
        )
        scheduler = (
            self._provider_config.get("scheduler", {}) if self._provider_config else {}
        )
        self._metrics = VllmMetricsScraper(
            args.metrics_url,
            timeout=args.health_timeout,
            dp_size=topology.get("dp", 1),
            max_num_seqs=scheduler.get("max_num_seqs", 1),
        )
        self._last_hb = 0.0
        self._state_seq = 0
        # Reuse one connection for the periodic heartbeat POSTs (keep-alive).
        self._hb_session = requests.Session()

    # --- registration primitives ------------------------------------------

    def _new_incarnation(self) -> str:
        token = uuid.uuid4().hex[:12]
        return (
            f"{self._args.instance_name}-{token}" if self._args.instance_name else token
        )

    def _register(self) -> bool:
        """Grant a fresh lease and put the instance key. Returns success."""
        try:
            incarnation = self._new_incarnation()
            lease_id = self._etcd.lease_grant(self._args.lease_ttl)
            descriptor = None
            if self._provider_config is not None:
                descriptor = build_provider_descriptor(
                    self._provider_config,
                    self._args.register_addr,
                    incarnation,
                    self._args.register_addr,
                )
            meta = build_instance_meta(
                self._args.register_addr,
                incarnation,
                self._instance_type,
                self._args.backend_type,
                descriptor,
            )
            self._etcd.put(self._key, json.dumps(meta), lease_id)
            self._lease_id = lease_id
            self._incarnation_id = incarnation
            self._state_seq = 0
            if self._agent is not None:
                self._agent.activate(incarnation, self._args.register_addr)
            logger.info(
                "registered %s (incarnation=%s, lease=%s, ttl=%ds)",
                self._key,
                incarnation,
                lease_id,
                self._args.lease_ttl,
            )
            return True
        except EtcdError as error:
            logger.warning("register failed, will retry: %s", error)
            self._lease_id = None
            if self._agent is not None:
                self._agent.fence("ADMISSION_REASON_ENGINE_DRAINING")
            return False

    def _deregister(self) -> None:
        """Revoke the lease so the master removes the instance immediately."""
        if self._agent is not None:
            self._agent.fence("ADMISSION_REASON_ENGINE_DRAINING")
        if self._lease_id is None:
            return
        try:
            self._etcd.lease_revoke(self._lease_id)
            logger.info("deregistered %s (lease=%s revoked)", self._key, self._lease_id)
        except EtcdError as error:
            # On failure the lease still expires within ttl; not fatal.
            logger.warning(
                "revoke failed (lease expires in <=%ds): %s",
                self._args.lease_ttl,
                error,
            )
        finally:
            self._lease_id = None
            self._incarnation_id = None

    @property
    def _registered(self) -> bool:
        return self._lease_id is not None

    # --- main loop ---------------------------------------------------------

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)

        if self._agent is not None:
            self._agent.start()

        try:
            self._wait_until_healthy()
            if self._stop.is_set():
                return
            model = self._health.served_model()
            logger.info("vLLM healthy (model=%s), registering ...", model or "?")
            self._register()

            fail = 0
            while not self._stop.wait(self._args.keepalive_interval):
                if self._health.is_healthy():
                    fail = 0
                    self._keepalive_or_reregister()
                    self._maybe_heartbeat()
                else:
                    fail += 1
                    if self._agent is not None:
                        self._agent.fence("ADMISSION_REASON_INTERNAL_ERROR")
                    logger.warning(
                        "vLLM health probe failed (%d/%d)",
                        fail,
                        self._args.health_fail_threshold,
                    )
                    if self._registered and fail >= self._args.health_fail_threshold:
                        logger.error("vLLM unhealthy, deregistering")
                        self._deregister()
        finally:
            self._deregister()
            if self._agent is not None:
                self._agent.stop()
            logger.info("sidecar stopped")

    def _wait_until_healthy(self) -> None:
        backoff = 1.0
        while not self._stop.is_set() and not self._health.is_healthy():
            logger.info(
                "waiting for vLLM at %s to become healthy ...", self._args.vllm_url
            )
            self._stop.wait(backoff)
            backoff = min(backoff * 2, self._args.lease_ttl)

    def _keepalive_or_reregister(self) -> None:
        if not self._registered:
            self._register()  # recovered after a previous deregister
            return
        try:
            ttl = self._etcd.lease_keepalive(self._lease_id)
            if ttl <= 0:
                logger.warning("lease %s lost (ttl=0), re-registering", self._lease_id)
                if self._agent is not None:
                    self._agent.fence("ADMISSION_REASON_STALE_INCARNATION")
                self._lease_id = None
                self._register()
            elif self._agent is not None and self._incarnation_id is not None:
                # A transient health failure fences ingress immediately. Only
                # a successful ownership renewal may reopen the same
                # incarnation after health recovers.
                self._agent.activate(
                    self._incarnation_id, self._args.register_addr
                )
        except EtcdError as error:
            logger.warning("keepalive failed, re-registering: %s", error)
            if self._agent is not None:
                self._agent.fence("ADMISSION_REASON_STALE_INCARNATION")
            self._lease_id = None
            self._register()

    def _on_signal(self, signum: int, _frame: FrameType | None) -> None:
        logger.info("received signal %d, shutting down", signum)
        self._stop.set()

    # --- heartbeat (metrics) ----------------------------------------------

    def _maybe_heartbeat(self) -> None:
        """POST LoadMetrics/LatencyMetrics on cadence while registered."""
        if not self._registered:
            return
        now = time.monotonic()
        if now - self._last_hb < self._args.heartbeat_interval:
            return
        self._last_hb = now
        self._send_heartbeat()

    def _send_heartbeat(self) -> None:
        metrics = self._metrics.scrape()
        if metrics is None:
            return  # /metrics transiently unreachable; lease still holds liveness
        body = {
            "name": self._args.register_addr,
            "incarnation_id": self._incarnation_id,
            "load_metrics": metrics["load_metrics"],
            "latency_metrics": metrics["latency_metrics"],
        }
        if self._provider_config is not None:
            self._state_seq += 1
            lifecycle = self._agent.lifecycle.engine_state()
            descriptor = build_provider_descriptor(
                self._provider_config,
                self._args.register_addr,
                self._incarnation_id,
                self._args.register_addr,
            )
            body["engine_state"] = {
                "engine_uid": self._args.register_addr,
                "incarnation_id": self._incarnation_id,
                "state_seq": self._state_seq,
                "observed_at_unix_ms": int(time.time() * 1000),
                "lifecycle": lifecycle["lifecycle"],
                "ownership": "ENGINE_OWNERSHIP_OWNED",
                "shallow_health": "HEALTH_STATUS_HEALTHY",
                "deep_health": "HEALTH_STATUS_UNKNOWN",
                "per_dp": metrics["per_dp"],
                "state_quality": metrics["state_quality"],
                "provider_id": "PROVIDER_ID_VLLM_ASCEND",
                "profile_digest": descriptor["profile_digest"],
                "model_revision": descriptor["model"]["model_revision"],
                "heartbeat_age_ms_at_publish": 0,
                "state_age_ms_at_publish": 0,
                "drain": lifecycle["drain"],
            }
            if lifecycle["operation_id"]:
                body["engine_state"]["lifecycle_operation_id"] = lifecycle[
                    "operation_id"
                ]
                body["engine_state"]["lifecycle_leader_epoch"] = lifecycle[
                    "leader_epoch"
                ]
                body["engine_state"]["lifecycle_desired_generation"] = lifecycle[
                    "desired_generation"
                ]
        headers = {"Content-Type": "application/json"}
        if self._args.internal_token:
            headers["X-Internal-Token"] = self._args.internal_token
        try:
            response = self._hb_session.post(
                self._hb_url,
                json=body,
                headers=headers,
                timeout=self._args.health_timeout,
            )
        except requests.RequestException as error:
            logger.debug("heartbeat POST failed: %s", error)
            return
        if response.status_code == 409:
            # master doesn't know this incarnation -> re-register and resync id
            logger.warning("heartbeat 409 (stale/unknown), re-registering")
            self._deregister()
            self._register()
        elif response.status_code == 401:
            logger.error("heartbeat 401: invalid --internal-token; stopping")
            self._deregister()
            self._stop.set()
        elif response.status_code != 200:
            logger.warning(
                "heartbeat -> HTTP %d: %s",
                response.status_code,
                response.text[:120],
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm_sidecar",
        description="Auto-register a vLLM instance into xllm-service via etcd.",
    )
    parser.add_argument(
        "--etcd-endpoints",
        default=os.environ.get("ETCD_ENDPOINTS", "127.0.0.1:2379"),
        help="comma-separated host:port list (default 127.0.0.1:2379)",
    )
    parser.add_argument(
        "--etcd-namespace",
        default=os.environ.get("ETCD_NAMESPACE", ""),
        help="must match master's --etcd_namespace (default empty)",
    )
    parser.add_argument(
        "--etcd-username", default=os.environ.get("ETCD_USERNAME", "")
    )
    parser.add_argument(
        "--etcd-password", default=os.environ.get("ETCD_PASSWORD", "")
    )
    parser.add_argument("--etcd-timeout", type=float, default=3.0)
    parser.add_argument(
        "--vllm-url",
        default="http://127.0.0.1:18000",
        help="base URL of the local vLLM server",
    )
    parser.add_argument(
        "--register-addr",
        default=None,
        help="host:port the master uses to reach vLLM, NO scheme "
        "(default: derived from --vllm-url)",
    )
    parser.add_argument(
        "--provider-config",
        default="",
        help="verified V2 Provider profile JSON; enables strict Agent mode",
    )
    parser.add_argument(
        "--agent-listen",
        default="",
        help="strict Agent bind host:port; port must match --register-addr",
    )
    parser.add_argument("--max-attempt-records", type=int, default=8192)
    parser.add_argument("--max-cancel-fences", type=int, default=8192)
    parser.add_argument("--max-lifecycle-records", type=int, default=4096)
    parser.add_argument("--attempt-terminal-ttl", type=float, default=60.0)
    parser.add_argument("--negative-fence-ttl", type=float, default=60.0)
    parser.add_argument("--agent-max-inflight-requests", type=int, default=256)
    parser.add_argument(
        "--agent-max-request-body-bytes", type=int, default=8 * 1024 * 1024
    )
    parser.add_argument(
        "--agent-inflight-body-capacity-bytes",
        type=int,
        default=64 * 1024 * 1024,
    )
    parser.add_argument("--agent-connect-timeout", type=float, default=1.0)
    parser.add_argument("--agent-ingress-timeout", type=float, default=5.0)
    parser.add_argument("--backend-type", default="vllm")
    parser.add_argument(
        "--instance-type",
        default="DEFAULT",
        choices=[instance_type.name for instance_type in InstanceType],
        help="single vLLM instance must be DEFAULT to be routable",
    )
    parser.add_argument(
        "--instance-name",
        default="vllm",
        help="human-readable prefix for the incarnation id / logs",
    )
    parser.add_argument(
        "--lease-ttl",
        type=int,
        default=6,
        help="etcd lease TTL in seconds (~2x keepalive interval)",
    )
    parser.add_argument(
        "--keepalive-interval",
        type=float,
        default=2.0,
        help="seconds between health probe + lease refresh",
    )
    parser.add_argument("--health-timeout", type=float, default=3.0)
    parser.add_argument(
        "--health-fail-threshold",
        type=int,
        default=3,
        help="consecutive failed probes before deregistering",
    )
    # heartbeat (LoadMetrics/LatencyMetrics reporting)
    parser.add_argument(
        "--xllm-service-url",
        default="http://127.0.0.1:9998",
        help="master HTTP base URL for /v1/internal/heartbeat",
    )
    parser.add_argument(
        "--internal-token",
        default=os.environ.get("XLLM_INTERNAL_TOKEN", ""),
        help="X-Internal-Token; must match master --internal_api_token",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=3.0,
        help="seconds between metrics heartbeats",
    )
    parser.add_argument(
        "--metrics-url",
        default=None,
        help="vLLM Prometheus endpoint (default: <vllm-url>/metrics)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def _derive_addr(vllm_url: str) -> str:
    parsed = urlsplit(vllm_url)
    if (
        parsed.scheme not in ("http", "https")
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or ":" in parsed.hostname
    ):
        raise ValueError("--vllm-url must be an HTTP(S) IPv4/DNS base URL")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return f"{parsed.hostname}:{port}"


def _positive_finite(value: object) -> bool:
    if type(value) not in (int, float) or value <= 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_runtime_args(args: argparse.Namespace) -> None:
    if args.backend_type != "vllm":
        raise ValueError("vLLM sidecar --backend-type must be vllm")
    if not isinstance(args.register_addr, str) or not args.register_addr:
        raise ValueError("--register-addr must be a nonempty host:port")
    split_address(args.register_addr)
    if (
        type(args.lease_ttl) is not int
        or args.lease_ttl <= 0
        or not _positive_finite(args.etcd_timeout)
        or not _positive_finite(args.keepalive_interval)
        or args.keepalive_interval >= args.lease_ttl
        or not _positive_finite(args.health_timeout)
        or type(args.health_fail_threshold) is not int
        or args.health_fail_threshold <= 0
        or not _positive_finite(args.heartbeat_interval)
    ):
        raise ValueError("sidecar limits and intervals must be positive and bounded")
    if (
        not isinstance(args.instance_name, str)
        or len(args.instance_name.encode("utf-8")) > 240
    ):
        raise ValueError("--instance-name is too long")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    if not args.register_addr:
        args.register_addr = (
            args.agent_listen if args.provider_config else _derive_addr(args.vllm_url)
        )
    if not args.metrics_url:
        args.metrics_url = args.vllm_url.rstrip("/") + "/metrics"
    logger.info(
        "sidecar starting: register %s as backend_type=%s type=%s",
        args.register_addr,
        args.backend_type,
        args.instance_type,
    )
    Sidecar(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
