# Copyright 2026 The xLLM Authors. All Rights Reserved.

import copy
import json

import pytest

from vllm_sidecar.descriptor import (
    ProviderConfigError,
    build_provider_descriptor,
    load_provider_config,
    validate_provider_config,
)
from vllm_sidecar.meta import build_instance_meta


def provider_config() -> dict:
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
            "device_count": 2,
            "tp": 1,
            "dp": 2,
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


def test_descriptor_is_strict_deterministic_and_incarnation_scoped() -> None:
    config = provider_config()
    first = build_provider_descriptor(config, "agent:18001", "inc-1", "agent:18001")
    second = build_provider_descriptor(config, "agent:18001", "inc-2", "agent:18001")

    assert first["contract_version"] == 1
    assert first["identity"]["provider_id"] == "PROVIDER_ID_VLLM_ASCEND"
    assert first["identity"]["incarnation_id"] == "inc-1"
    assert first["profile_digest"] == second["profile_digest"]
    assert "PROVIDER_CAPABILITY_SELF_FENCING" in first["capabilities"]
    assert "PROVIDER_CAPABILITY_PER_DP_STATE" in first["capabilities"]

    meta = build_instance_meta(
        "agent:18001", "inc-1", provider_descriptor=first
    )
    assert meta["provider_contract_version"] == 1
    assert meta["provider_profile_digest"] == first["profile_digest"]
    assert meta["provider_descriptor"] == first


def test_config_validation_fails_closed(tmp_path) -> None:
    config = provider_config()
    config["runtime"]["raw_ingress_isolated"] = False
    with pytest.raises(ProviderConfigError, match="raw_ingress_isolated"):
        validate_provider_config(config)

    config = provider_config()
    del config["model"]["renderer_digest"]
    path = tmp_path / "provider.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ProviderConfigError, match="renderer_digest"):
        load_provider_config(str(path))

    config = provider_config()
    bad = copy.deepcopy(config)
    bad["topology"]["dp"] = 0
    with pytest.raises(ProviderConfigError, match="topology.dp"):
        validate_provider_config(bad)

    bad = provider_config()
    bad["runtime"]["runtime_version"] = 21
    with pytest.raises(ProviderConfigError, match="runtime_version"):
        validate_provider_config(bad)

    bad = provider_config()
    bad["kv"]["cache_groups"] = "full-attention"
    with pytest.raises(ProviderConfigError, match="cache_groups"):
        validate_provider_config(bad)

    bad = provider_config()
    bad["api_features"] = ["completions", "completions"]
    with pytest.raises(ProviderConfigError, match="api_features"):
        validate_provider_config(bad)

    bad = provider_config()
    bad["api_features"] = ["anthropic_messages"]
    with pytest.raises(ProviderConfigError, match="unsupported"):
        validate_provider_config(bad)


def test_descriptor_registration_identity_must_match() -> None:
    descriptor = build_provider_descriptor(
        provider_config(), "agent:18001", "inc-1", "agent:18001"
    )
    with pytest.raises(ValueError, match="engine_uid"):
        build_instance_meta("other:18001", "inc-1", provider_descriptor=descriptor)

    wrong_provider = copy.deepcopy(descriptor)
    wrong_provider["identity"]["provider_id"] = "PROVIDER_ID_XLLM_NATIVE"
    with pytest.raises(ValueError, match="provider_id"):
        build_instance_meta(
            "agent:18001", "inc-1", provider_descriptor=wrong_provider
        )

    wrong_endpoint = copy.deepcopy(descriptor)
    wrong_endpoint["endpoint"]["address"] = "raw-vllm:18000"
    with pytest.raises(ValueError, match="endpoint"):
        build_instance_meta(
            "agent:18001", "inc-1", provider_descriptor=wrong_endpoint
        )
