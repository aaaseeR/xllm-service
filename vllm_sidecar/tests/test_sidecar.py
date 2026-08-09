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
"""Runtime unit tests for the vLLM sidecar without live etcd/vLLM."""

import argparse
import base64
import json

import pytest
import requests

from vllm_sidecar import sidecar as sidecar_mod
from vllm_sidecar.etcd_registry import EtcdError, EtcdGatewayClient
from vllm_sidecar.health import VllmHealthProbe
from vllm_sidecar.meta import InstanceType


def _provider_config():
    return {
        "runtime": {
            "runtime_version": "0.21.0",
            "plugin_version": "0.11.0",
            "hardware_runtime_version": "CANN-8.3",
            "fate_bound_mode": "same_restart_unit",
            "raw_ingress_isolated": True,
        },
        "model": {
            "model_revision": "model-r1",
            "tokenizer_revision": "tokenizer-r1",
            "chat_template_digest": "sha256:template",
            "quantization": "none",
            "renderer_digest": "sha256:renderer",
        },
        "topology": {
            "soc": "ascend-910b",
            "device_count": 1,
            "tp": 1,
            "dp": 1,
            "pp": 1,
            "ep": 1,
            "cp": 1,
        },
        "kv": {
            "kv_layout_digest": "sha256:kv",
            "cache_dtype": "bf16",
            "block_size": 16,
            "cache_groups": ["full-attention"],
            "head_shard_mapping_digest": "sha256:heads",
            "connector": "none",
            "connector_version": "1",
        },
        "scheduler": {
            "scheduler_class": "vllm-v1",
            "max_num_seqs": 64,
            "max_num_batched_tokens": 8192,
            "scheduler_policy_digest": "sha256:scheduler",
        },
    }


class _Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def post(self, url, json, timeout):
        self.calls.append((url, json, timeout))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _Etcd:
    def __init__(self, *_args, **_kwargs):
        self.puts = []
        self.keepalives = []
        self.revoked = []
        self.keepalive_ttl = 5
        self.fail_grant = False
        self.fail_keepalive = False

    def lease_grant(self, ttl_seconds):
        if self.fail_grant:
            raise EtcdError("grant failed")
        return "lease-1"

    def put(self, key, value, lease_id):
        self.puts.append((key, json.loads(value), lease_id))

    def lease_keepalive(self, lease_id):
        if self.fail_keepalive:
            raise EtcdError("keepalive failed")
        self.keepalives.append(lease_id)
        return self.keepalive_ttl

    def lease_revoke(self, lease_id):
        self.revoked.append(lease_id)


class _Health:
    def is_healthy(self):
        return True

    def served_model(self):
        return "demo-model"


def _args(**kwargs):
    values = dict(
        etcd_endpoints="127.0.0.1:2379",
        etcd_username="",
        etcd_password="",
        etcd_timeout=1.0,
        vllm_url="http://127.0.0.1:18000",
        register_addr="127.0.0.1:18000",
        backend_type="vllm",
        instance_type="DEFAULT",
        instance_name="vllm",
        etcd_namespace="",
        lease_ttl=6,
        keepalive_interval=2.0,
        health_timeout=1.0,
        health_fail_threshold=3,
        xllm_service_url="http://127.0.0.1:9998",
        internal_token="test-token",
        heartbeat_interval=3.0,
        metrics_url="http://127.0.0.1:18000/metrics",
        log_level="INFO",
    )
    values.update(kwargs)
    return argparse.Namespace(**values)


def _install_fakes(monkeypatch):
    etcd = _Etcd()
    monkeypatch.setattr(sidecar_mod, "EtcdGatewayClient", lambda *a, **k: etcd)
    monkeypatch.setattr(sidecar_mod, "VllmHealthProbe", lambda *a, **k: _Health())
    return etcd


def test_health_probe_success_failure_and_model(monkeypatch):
    responses = iter(
        [
            _Response(204),
            _Response(503),
            requests.Timeout("timeout"),
            _Response(200, {"data": [{"id": "model-a"}]}),
            _Response(200, {"data": []}),
        ]
    )

    def fake_get(*_args, **_kwargs):
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(requests, "get", fake_get)
    probe = VllmHealthProbe("http://vllm/")
    monkeypatch.setattr(probe._session, "get", fake_get)
    assert probe.is_healthy()
    assert not probe.is_healthy()
    assert not probe.is_healthy()
    assert probe.served_model() == "model-a"
    assert probe.served_model() is None


def test_etcd_gateway_lease_kv_auth_and_errors(monkeypatch):
    value = base64.b64encode(b"payload").decode("ascii")
    session = _Session(
        [
            _Response(payload={"token": "tok"}),
            _Response(payload={"ID": "lease-1"}),
            _Response(payload={"result": {"TTL": "4"}}),
            _Response(),
            _Response(),
            _Response(payload={"kvs": [{"value": value}]}),
            _Response(payload={}),
        ]
    )
    monkeypatch.setattr(requests, "Session", lambda: session)
    client = EtcdGatewayClient("127.0.0.1:2379", username="u", password="p")

    assert client._session.headers["Authorization"] == "tok"
    assert client.lease_grant(6) == "lease-1"
    assert client.lease_keepalive("lease-1") == 4
    client.lease_revoke("lease-1")
    client.put("key", "payload", "lease-1")
    assert client.get("key") == "payload"
    assert client.get("missing") is None

    monkeypatch.setattr(
        requests, "Session", lambda: _Session([_Response(500, text="bad")])
    )
    with pytest.raises(EtcdError):
        EtcdGatewayClient("127.0.0.1:2379").lease_grant(6)
    with pytest.raises(ValueError):
        EtcdGatewayClient("")


def test_etcd_gateway_endpoint_scheme_normalization():
    client = EtcdGatewayClient(
        "127.0.0.1:2379, https://secure:2379, http://plain:2379/"
    )
    assert client._bases == [
        "http://127.0.0.1:2379",
        "https://secure:2379",
        "http://plain:2379",
    ]


def test_etcd_gateway_raises_on_non_json_200(monkeypatch):
    class _BadJson:
        status_code = 200
        text = "<html>proxy error</html>"

        def json(self):
            raise ValueError("not json")

    monkeypatch.setattr(requests, "Session", lambda: _Session([_BadJson()]))
    # A 200 with an undecodable body must surface as EtcdError, not a raw
    # ValueError that would crash the sidecar.
    with pytest.raises(EtcdError):
        EtcdGatewayClient("127.0.0.1:2379").lease_grant(6)


def test_etcd_gateway_keepalive_handles_null_result(monkeypatch):
    # etcd may serialize an expired lease as {"result": null}; that must read
    # back as TTL 0, not raise AttributeError.
    monkeypatch.setattr(
        requests, "Session", lambda: _Session([_Response(payload={"result": None})])
    )
    assert EtcdGatewayClient("127.0.0.1:2379").lease_keepalive("lease-1") == 0


def test_etcd_gateway_get_handles_empty_value(monkeypatch):
    # proto3 JSON omits an empty "value" field; get() must return "" not crash.
    monkeypatch.setattr(
        requests, "Session", lambda: _Session([_Response(payload={"kvs": [{}]})])
    )
    assert EtcdGatewayClient("127.0.0.1:2379").get("key") == ""


def test_etcd_gateway_coerces_non_object_200_to_empty(monkeypatch):
    # A 200 whose JSON body is not an object (e.g. a list/null) must not crash
    # callers that rely on dict .get(); it surfaces as a normal EtcdError.
    monkeypatch.setattr(requests, "Session", lambda: _Session([_Response(payload=[])]))
    with pytest.raises(EtcdError):
        EtcdGatewayClient("127.0.0.1:2379").lease_grant(6)


def test_etcd_gateway_keepalive_handles_null_ttl(monkeypatch):
    monkeypatch.setattr(
        requests,
        "Session",
        lambda: _Session([_Response(payload={"result": {"TTL": None}})]),
    )
    assert EtcdGatewayClient("127.0.0.1:2379").lease_keepalive("lease-1") == 0


def test_send_heartbeat_posts_metrics_with_token(monkeypatch):
    _install_fakes(monkeypatch)

    class _Scraper:
        def __init__(self, *a, **k):
            pass

        def scrape(self):
            return {
                "load_metrics": {
                    "waiting_requests_num": 2,
                    "gpu_cache_usage_perc": 0.4,
                },
                "latency_metrics": {"recent_max_ttft": 12, "recent_max_tbt": 3},
            }

    monkeypatch.setattr(sidecar_mod, "VllmMetricsScraper", _Scraper)
    sc = sidecar_mod.Sidecar(_args(internal_token="tok"))
    sc._lease_id = "lease-1"
    sc._incarnation_id = "vllm-x"

    captured = {}

    def fake_post(url, json, headers, timeout):
        captured.update(url=url, json=json, headers=headers)

        class _R:
            status_code = 200
            text = ""

        return _R()

    monkeypatch.setattr(sc._hb_session, "post", fake_post)
    sc._send_heartbeat()

    assert captured["url"].endswith("/v1/internal/heartbeat")
    assert captured["headers"]["X-Internal-Token"] == "tok"
    assert captured["json"]["load_metrics"]["waiting_requests_num"] == 2


def test_heartbeat_auth_failure_deregisters_and_stops(monkeypatch):
    etcd = _install_fakes(monkeypatch)
    sc = sidecar_mod.Sidecar(_args(internal_token="wrong-token"))
    assert sc._register()

    class _Scraper:
        def scrape(self):
            return {
                "load_metrics": {},
                "latency_metrics": {},
            }

    class _Response:
        status_code = 401
        text = "unauthorized"

    sc._metrics = _Scraper()
    monkeypatch.setattr(
        sc._hb_session, "post", lambda *args, **kwargs: _Response()
    )
    sc._send_heartbeat()

    assert sc._stop.is_set()
    assert not sc._registered
    assert etcd.revoked == ["lease-1"]


def test_sidecar_registration_keepalive_and_deregister(monkeypatch):
    etcd = _install_fakes(monkeypatch)
    sc = sidecar_mod.Sidecar(_args())

    assert sc._register()
    assert sc._registered
    key, meta, lease_id = etcd.puts[0]
    assert key == "XLLM:DEFAULT:127.0.0.1:18000"
    assert meta["backend_type"] == "vllm"
    assert lease_id == "lease-1"

    sc._keepalive_or_reregister()
    assert etcd.keepalives == ["lease-1"]

    sc._deregister()
    assert not sc._registered
    assert etcd.revoked == ["lease-1"]


def test_strict_agent_registers_descriptor_and_publishes_engine_state(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "provider.json"
    config_path.write_text(json.dumps(_provider_config()), encoding="utf-8")
    etcd = _install_fakes(monkeypatch)
    sc = sidecar_mod.Sidecar(
        _args(
            provider_config=str(config_path),
            agent_listen="127.0.0.1:0",
            register_addr="127.0.0.1:0",
            max_cancel_fences=1,
        )
    )
    sc._agent.start()
    try:
        assert sc._register()
        _, meta, _ = etcd.puts[0]
        assert meta["provider_contract_version"] == 1
        assert meta["provider_descriptor"]["identity"]["provider_id"] == (
            "PROVIDER_ID_VLLM_ASCEND"
        )
        assert meta["provider_descriptor"]["endpoint"]["address"] == (
            sc._args.register_addr
        )

        class Scraper:
            def scrape(self):
                return {
                    "load_metrics": {
                        "waiting_requests_num": 1,
                        "gpu_cache_usage_perc": 0.25,
                    },
                    "latency_metrics": {
                        "recent_max_ttft": 2,
                        "recent_max_tbt": 1,
                    },
                    "per_dp": [
                        {
                            "dp_rank": 0,
                            "running": 2,
                            "waiting_capacity": 1,
                            "kv_used_ratio": 0.25,
                            "admission_credit": 61,
                        }
                    ],
                    "state_quality": "STATE_QUALITY_FULL",
                }

        sc._metrics = Scraper()
        captured = {}

        def fake_post(url, json, headers, timeout):
            captured.update(body=json)

            class Response:
                status_code = 200
                text = ""

            return Response()

        monkeypatch.setattr(sc._hb_session, "post", fake_post)
        sc._send_heartbeat()
        state = captured["body"]["engine_state"]
        assert state["state_seq"] == 1
        assert state["state_quality"] == "STATE_QUALITY_FULL"
        assert state["profile_digest"] == meta["provider_profile_digest"]
        assert state["per_dp"][0]["admission_credit"] == 61

        fence = sc._agent.ledger.cancel(
            "capacity-pressure", 0, sc._incarnation_id
        )
        assert fence.accepted
        sc._send_heartbeat()
        assert captured["body"]["engine_state"]["lifecycle"] == (
            "ENGINE_LIFECYCLE_DRAINING"
        )
    finally:
        sc._deregister()
        sc._agent.stop()


def test_strict_agent_separates_bind_and_advertised_hosts(monkeypatch, tmp_path):
    config_path = tmp_path / "provider.json"
    config_path.write_text(json.dumps(_provider_config()), encoding="utf-8")
    _install_fakes(monkeypatch)
    sc = sidecar_mod.Sidecar(
        _args(
            provider_config=str(config_path),
            agent_listen="0.0.0.0:0",
            register_addr="127.0.0.1:0",
        )
    )
    try:
        assert sc._agent.listen_address.startswith("0.0.0.0:")
        assert sc._args.register_addr.startswith("127.0.0.1:")
        assert not sc._args.register_addr.endswith(":0")
    finally:
        sc._agent.stop()


@pytest.mark.parametrize(
    ("agent_listen", "register_addr", "instance_type", "error"),
    [
        ("0.0.0.0:0", "0.0.0.0:0", "DEFAULT", "routable"),
        ("127.0.0.1:18001", "127.0.0.1:18002", "DEFAULT", "ports"),
        ("127.0.0.1:0", "127.0.0.1:0", "PREFILL", "DEFAULT"),
    ],
)
def test_strict_agent_rejects_unroutable_or_mismatched_registration(
    tmp_path, agent_listen, register_addr, instance_type, error
):
    config_path = tmp_path / "provider.json"
    config_path.write_text(json.dumps(_provider_config()), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        sidecar_mod.Sidecar(
            _args(
                provider_config=str(config_path),
                agent_listen=agent_listen,
                register_addr=register_addr,
                instance_type=instance_type,
            )
        )


def test_strict_agent_rejects_missing_internal_token(tmp_path):
    config_path = tmp_path / "provider.json"
    config_path.write_text(json.dumps(_provider_config()), encoding="utf-8")
    with pytest.raises(ValueError, match="internal-token"):
        sidecar_mod.Sidecar(
            _args(
                provider_config=str(config_path),
                agent_listen="127.0.0.1:0",
                register_addr="127.0.0.1:0",
                internal_token="",
            )
        )


def test_strict_agent_reopens_only_after_successful_keepalive(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "provider.json"
    config_path.write_text(json.dumps(_provider_config()), encoding="utf-8")
    _install_fakes(monkeypatch)
    sc = sidecar_mod.Sidecar(
        _args(
            provider_config=str(config_path),
            agent_listen="127.0.0.1:0",
            register_addr="127.0.0.1:0",
        )
    )

    assert sc._register()
    assert sc._agent.ledger.accepting()
    sc._agent.fence("ADMISSION_REASON_INTERNAL_ERROR")
    assert not sc._agent.ledger.accepting()
    sc._keepalive_or_reregister()
    assert sc._agent.ledger.accepting()
    sc._deregister()


def test_heartbeat_stale_incarnation_revokes_before_reregister(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "provider.json"
    config_path.write_text(json.dumps(_provider_config()), encoding="utf-8")
    etcd = _install_fakes(monkeypatch)
    sc = sidecar_mod.Sidecar(
        _args(
            provider_config=str(config_path),
            agent_listen="127.0.0.1:0",
            register_addr="127.0.0.1:0",
        )
    )
    assert sc._register()
    old_incarnation = sc._incarnation_id

    class Scraper:
        def scrape(self):
            return {
                "load_metrics": {},
                "latency_metrics": {},
                "per_dp": [{"dp_rank": 0}],
                "state_quality": "STATE_QUALITY_PARTIAL",
            }

    class Response:
        status_code = 409
        text = "stale"

    sc._metrics = Scraper()
    monkeypatch.setattr(
        sc._hb_session, "post", lambda *_args, **_kwargs: Response()
    )
    sc._send_heartbeat()

    assert etcd.revoked == ["lease-1"]
    assert len(etcd.puts) == 2
    assert sc._incarnation_id != old_incarnation
    assert sc._agent.ledger.accepting()
    sc._deregister()


def test_sidecar_reregisters_on_lost_or_failed_lease(monkeypatch):
    etcd = _install_fakes(monkeypatch)
    sc = sidecar_mod.Sidecar(_args())
    assert sc._register()

    etcd.keepalive_ttl = 0
    sc._keepalive_or_reregister()
    assert len(etcd.puts) == 2

    etcd.fail_keepalive = True
    sc._keepalive_or_reregister()
    assert len(etcd.puts) == 3

    etcd.fail_grant = True
    sc._lease_id = None
    assert not sc._register()
    assert not sc._registered


def test_sidecar_parser_and_main(monkeypatch):
    ran = []

    class FakeSidecar:
        def __init__(self, args):
            self.args = args

        def run(self):
            ran.append((self.args.register_addr, self.args.log_level))

    monkeypatch.setattr(sidecar_mod, "Sidecar", FakeSidecar)
    assert sidecar_mod._derive_addr("http://host:18000/v1/models") == "host:18000"
    assert (
        sidecar_mod.main(["--vllm-url", "http://host:18000", "--log-level", "DEBUG"])
        == 0
    )
    assert ran == [("host:18000", "DEBUG")]
    assert sidecar_mod._derive_addr("https://secure.example/v1") == (
        "secure.example:443"
    )
    with pytest.raises(ValueError):
        sidecar_mod._derive_addr("not-a-url")

    args = sidecar_mod.build_parser().parse_args(["--instance-type", "DEFAULT"])
    assert args.instance_type == InstanceType.DEFAULT.name
