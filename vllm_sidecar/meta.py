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
"""InstanceMetaInfo schema + etcd key construction (Python mirror of C++).

The single source of truth for the wire schema is the C++ struct
`InstanceMetaInfo` in `xllm_service/common/types.h`:

  * `serialize_to_json()` (types.h ~:226) -- what the master writes,
  * `parse_from_json()`   (types.h ~:248) -- what the watcher reads. It requires
    `name`, `rpc_address`, `type` (`.at()`); everything else is optional and
    defaults (e.g. `backend_type` -> "xllm").

We therefore emit exactly the subset the watcher needs, matching the field
names and value encodings used by C++. The etcd key layout and namespace
normalization mirror `instance_mgr.cpp` (`ETCD_KEYS_PREFIX_MAP`) and
`utils.cpp` (`normalize_etcd_namespace` / `build_etcd_key_with_namespace`).
"""

from __future__ import annotations

import time
from enum import IntEnum

# Mirror of `enum class InstanceType` in common/types.h (DEFAULT = 0, ...).
# Only DEFAULT is exercised today; a single vLLM instance with no decode peer is
# routable only as DEFAULT (see the M2 routing constraint).


class InstanceType(IntEnum):
    DEFAULT = 0
    PREFILL = 1
    DECODE = 2
    MIX = 3


# Mirror of `ETCD_KEYS_PREFIX_MAP` in scheduler/managers/instance_mgr.cpp:45.
ETCD_KEYS_PREFIX_MAP = {
    InstanceType.DEFAULT: "XLLM:DEFAULT:",
    InstanceType.PREFILL: "XLLM:PREFILL:",
    InstanceType.DECODE: "XLLM:DECODE:",
    InstanceType.MIX: "XLLM:MIX:",
}


def normalize_etcd_namespace(etcd_namespace: str) -> str:
    """Port of utils::normalize_etcd_namespace (utils.cpp:105).

    "" -> "";  "foo" -> "/foo/";  "/a/b/" -> "/a/b/";  "///" -> "".
    """
    if not etcd_namespace:
        return ""
    trimmed = etcd_namespace.strip("/")
    if not trimmed:
        return ""
    return "/" + trimmed + "/"


def build_instance_key(
    addr: str,
    instance_type: InstanceType = InstanceType.DEFAULT,
    etcd_namespace: str = "",
) -> str:
    """Full etcd key the master watches, e.g. ``XLLM:DEFAULT:127.0.0.1:18000``.

    ``addr`` is the address the master uses to reach the backend -- host:port
    with NO scheme; xllm-service selects HTTP via the channel's protocol option.
    """
    logical_key = ETCD_KEYS_PREFIX_MAP[instance_type] + addr
    return normalize_etcd_namespace(etcd_namespace) + logical_key


def build_instance_meta(
    addr: str,
    incarnation_id: str,
    instance_type: InstanceType = InstanceType.DEFAULT,
    backend_type: str = "vllm",
    provider_descriptor: dict | None = None,
) -> dict:
    """Build the InstanceMetaInfo JSON payload stored under the lease.

    Matches `register_vllm.sh` and the required fields of
    `InstanceMetaInfo::parse_from_json`. ``register_ts_ms`` is real epoch-ms so
    the master's logs/ordering are meaningful (the manual script hard-coded 1).
    """
    if backend_type != "vllm":
        raise ValueError(f"unsupported vLLM sidecar backend_type: {backend_type}")
    meta = {
        "name": addr,
        "rpc_address": addr,
        "type": int(instance_type),
        "backend_type": backend_type,
        "provider_id": 2,
        "incarnation_id": incarnation_id,
        "register_ts_ms": int(time.time() * 1000),
    }
    if provider_descriptor is None:
        # Contract version zero marks this payload as the legacy BEST_EFFORT
        # bridge. Strict V2 registration always supplies a full Descriptor.
        meta["provider_contract_version"] = 0
        meta["provider_profile_digest"] = ""
        return meta

    identity = provider_descriptor.get("identity", {})
    if identity.get("engine_uid") != addr:
        raise ValueError("ProviderDescriptor engine_uid must equal registered address")
    if identity.get("incarnation_id") != incarnation_id:
        raise ValueError(
            "ProviderDescriptor incarnation_id does not match registration"
        )
    if identity.get("provider_id") != "PROVIDER_ID_VLLM_ASCEND":
        raise ValueError("ProviderDescriptor provider_id must be VLLM_ASCEND")
    if provider_descriptor.get("contract_version") != 1:
        raise ValueError("ProviderDescriptor contract_version must be 1")
    if provider_descriptor.get("endpoint", {}).get("address") != addr:
        raise ValueError("ProviderDescriptor endpoint must equal registered address")
    profile_digest = provider_descriptor.get("profile_digest")
    if not isinstance(profile_digest, str) or not profile_digest:
        raise ValueError("ProviderDescriptor profile_digest must be nonempty")
    meta["provider_contract_version"] = 1
    meta["provider_profile_digest"] = profile_digest
    meta["provider_descriptor"] = provider_descriptor
    return meta
