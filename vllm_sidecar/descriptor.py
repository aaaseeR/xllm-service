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
"""Build a strict vLLM-Ascend ProviderDescriptor from verified configuration."""

import copy
import hashlib
import json
from pathlib import Path


class ProviderConfigError(ValueError):
    pass


_REQUIRED_SECTIONS = ("runtime", "model", "topology", "kv", "scheduler")
_REQUIRED_FIELDS = {
    "runtime": (
        "runtime_version",
        "plugin_version",
        "hardware_runtime_version",
        "fate_bound_mode",
        "raw_ingress_isolated",
    ),
    "model": (
        "model_revision",
        "tokenizer_revision",
        "chat_template_digest",
        "quantization",
        "renderer_digest",
    ),
    "topology": ("soc", "device_count", "tp", "dp", "pp", "ep", "cp"),
    "kv": (
        "kv_layout_digest",
        "cache_dtype",
        "block_size",
        "cache_groups",
        "head_shard_mapping_digest",
        "connector",
        "connector_version",
    ),
    "scheduler": (
        "scheduler_class",
        "max_num_seqs",
        "max_num_batched_tokens",
        "scheduler_policy_digest",
    ),
}
_TEXT_FIELDS = {
    "runtime": (
        "runtime_version",
        "plugin_version",
        "hardware_runtime_version",
        "fate_bound_mode",
    ),
    "model": _REQUIRED_FIELDS["model"],
    "topology": ("soc",),
    "kv": (
        "kv_layout_digest",
        "cache_dtype",
        "head_shard_mapping_digest",
        "connector",
        "connector_version",
    ),
    "scheduler": ("scheduler_class", "scheduler_policy_digest"),
}
_SUPPORTED_API_FEATURES = {
    "chat_completions",
    "completions",
    "models",
}
_MAX_UINT32 = (1 << 32) - 1
_MAX_UINT64 = (1 << 64) - 1


def load_provider_config(path: str) -> dict:
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProviderConfigError(f"cannot load provider config: {error}") from error
    validate_provider_config(config)
    return config


def _require_nonempty(section: str, name: str, value: object) -> None:
    if value is None or value == "" or value == [] or value == 0:
        raise ProviderConfigError(f"{section}.{name} must be nonempty")


def _require_positive_integer(
    section: str, name: str, value: object, maximum: int
) -> None:
    if type(value) is not int or value <= 0 or value > maximum:
        raise ProviderConfigError(
            f"{section}.{name} must be a positive bounded integer"
        )


def validate_provider_config(config: object) -> None:
    if not isinstance(config, dict):
        raise ProviderConfigError("provider config must be a JSON object")
    for section in _REQUIRED_SECTIONS:
        values = config.get(section)
        if not isinstance(values, dict):
            raise ProviderConfigError(f"missing provider config section: {section}")
        for name in _REQUIRED_FIELDS[section]:
            if name not in values:
                raise ProviderConfigError(
                    f"missing provider config field: {section}.{name}"
                )
            _require_nonempty(section, name, values[name])
        for name in _TEXT_FIELDS[section]:
            if not isinstance(values[name], str):
                raise ProviderConfigError(f"{section}.{name} must be a string")

    topology = config["topology"]
    for name in ("device_count", "tp", "dp", "pp", "ep", "cp"):
        _require_positive_integer(
            "topology", name, topology[name], _MAX_UINT32
        )
    scheduler = config["scheduler"]
    _require_positive_integer(
        "scheduler",
        "max_num_seqs",
        scheduler["max_num_seqs"],
        _MAX_UINT32,
    )
    _require_positive_integer(
        "scheduler",
        "max_num_batched_tokens",
        scheduler["max_num_batched_tokens"],
        _MAX_UINT64,
    )
    _require_positive_integer(
        "kv", "block_size", config["kv"]["block_size"], _MAX_UINT32
    )
    if not isinstance(config["kv"]["cache_groups"], list) or not all(
        isinstance(item, str) and item for item in config["kv"]["cache_groups"]
    ):
        raise ProviderConfigError("kv.cache_groups must contain nonempty strings")
    if len(set(config["kv"]["cache_groups"])) != len(
        config["kv"]["cache_groups"]
    ):
        raise ProviderConfigError("kv.cache_groups must not contain duplicates")
    api_features = config.get("api_features")
    if api_features is not None and (
        not isinstance(api_features, list)
        or not api_features
        or not all(isinstance(item, str) and item for item in api_features)
        or len(set(api_features)) != len(api_features)
    ):
        raise ProviderConfigError(
            "api_features must contain unique nonempty strings"
        )
    if api_features is not None and not set(api_features).issubset(
        _SUPPORTED_API_FEATURES
    ):
        raise ProviderConfigError("api_features contains an unsupported feature")
    runtime = config["runtime"]
    if runtime["fate_bound_mode"] not in ("parent_death_signal", "same_restart_unit"):
        raise ProviderConfigError("runtime.fate_bound_mode is not supported")
    if runtime["raw_ingress_isolated"] is not True:
        raise ProviderConfigError("runtime.raw_ingress_isolated must be true")


def _profile_digest(config: dict) -> str:
    profile = copy.deepcopy(config)
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_provider_descriptor(
    config: dict,
    engine_uid: str,
    incarnation_id: str,
    endpoint: str,
) -> dict:
    validate_provider_config(config)
    runtime = config["runtime"]
    model = config["model"]
    topology = config["topology"]
    kv = config["kv"]
    scheduler = config["scheduler"]
    profile_digest = _profile_digest(config)
    return {
        "contract_version": 1,
        "identity": {
            "engine_uid": engine_uid,
            "incarnation_id": incarnation_id,
            "provider_id": "PROVIDER_ID_VLLM_ASCEND",
            "runtime_family": "vllm",
            "runtime_version": runtime["runtime_version"],
            "plugin_version": runtime["plugin_version"],
            "hardware_runtime_version": runtime["hardware_runtime_version"],
            "protocol_version": 1,
        },
        "endpoint": {
            "control_transport": "http",
            "data_transport": "http_sse",
            "address": endpoint,
        },
        "serving": {
            "role": "ENGINE_ROLE_AGGREGATED",
            "execution_modes": [
                {
                    "mode": "EXECUTION_MODE_AGGREGATED",
                    "transfer_mode": "TRANSFER_MODE_NONE",
                    "selection_order": "SELECTION_ORDER_SINGLE",
                    "binding_stage": "BINDING_STAGE_AT_SUBMIT",
                    "p_selection_delegated": False,
                }
            ],
            "api_features": list(
                config.get(
                    "api_features",
                    ["chat_completions", "completions", "models"],
                )
            ),
        },
        "model": copy.deepcopy(model),
        "topology": copy.deepcopy(topology),
        "kv": {
            **copy.deepcopy(kv),
            "transfer_modes": ["TRANSFER_MODE_NONE"],
        },
        "scheduler": copy.deepcopy(scheduler),
        "capabilities": [
            "PROVIDER_CAPABILITY_AGGREGATED",
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
