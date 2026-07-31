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

"""Validate the production invariants of the xllm-service Kubernetes base."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml


EXPECTED_RESOURCES = {
    "deployment.yaml",
    "inferencepool.yaml",
    "networkpolicy.yaml",
    "poddisruptionbudget.yaml",
    "service.yaml",
}
RUNTIME_KEYS = {
    "XLLM_ETCD_ADDR",
    "XLLM_ETCD_NAMESPACE",
    "XLLM_RUNTIME_PORT",
    "XLLM_TOKENIZER_PATH",
    "XLLM_SHUTDOWN_GRACE_PERIOD_S",
}
PUBLIC_PATHS = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/models",
}
IMAGE_DIGEST_PATTERN = re.compile(r"^[^@\s]+@sha256:[0-9a-fA-F]{64}$")


class ContractError(ValueError):
    """Raised when a deployment contract invariant is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            documents = list(yaml.safe_load_all(stream))
    except (OSError, yaml.YAMLError) as error:
        raise ContractError(f"cannot load {path}: {error}") from error

    require(len(documents) == 1, f"{path} must contain exactly one YAML document")
    require(isinstance(documents[0], dict), f"{path} must contain a YAML mapping")
    return documents[0]


def load_runtime_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ContractError(f"cannot load {path}: {error}") from error

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        require("=" in line, f"{path}:{line_number} is not KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        require(key and key not in values, f"duplicate runtime key {key!r}")
        require(value != "", f"runtime key {key!r} has an empty value")
        values[key] = value

    require(set(values) == RUNTIME_KEYS,
            "runtime.env keys must exactly match the supported startup inputs")
    return values


def load_contract(base_dir: Path, http_route: Path) -> dict[str, Any]:
    return {
        "kustomization": load_yaml(base_dir / "kustomization.yaml"),
        "deployment": load_yaml(base_dir / "deployment.yaml"),
        "service": load_yaml(base_dir / "service.yaml"),
        "inferencepool": load_yaml(base_dir / "inferencepool.yaml"),
        "pdb": load_yaml(base_dir / "poddisruptionbudget.yaml"),
        "networkpolicy": load_yaml(base_dir / "networkpolicy.yaml"),
        "runtime": load_runtime_env(base_dir / "runtime.env"),
        "httproute": load_yaml(http_route),
    }


def mapping_at(value: Any, path: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{path} must be a mapping")
    return value


def list_at(value: Any, path: str) -> list[Any]:
    require(isinstance(value, list), f"{path} must be a list")
    return value


def named_item(items: Any, name: str, path: str) -> dict[str, Any]:
    matches = [
        item for item in list_at(items, path)
        if isinstance(item, dict) and item.get("name") == name
    ]
    require(len(matches) == 1, f"{path} must contain exactly one {name!r} item")
    return matches[0]


def argument_map(arguments: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for argument in list_at(arguments, "deployment container args"):
        require(isinstance(argument, str) and argument.startswith("--") and "=" in argument,
                f"invalid container argument {argument!r}")
        name, value = argument[2:].split("=", 1)
        require(name not in result, f"duplicate container argument --{name}")
        result[name] = value
    return result


def zero_unavailable(value: Any) -> bool:
    return value in (0, "0", "0%")


def positive_surge(value: Any) -> bool:
    if isinstance(value, int):
        return value > 0
    if isinstance(value, str):
        match = re.fullmatch(r"(\d+)%?", value)
        return match is not None and int(match.group(1)) > 0
    return False


def validate_contract(
    documents: dict[str, Any],
    production: bool = False,
    expected_topology: str = "aggregated",
) -> None:
    require(expected_topology in {"aggregated", "pd"},
            "expected topology must be aggregated or pd")
    kustomization = mapping_at(documents["kustomization"], "kustomization")
    resource_list = list_at(kustomization.get("resources"), "kustomization.resources")
    resources = set(resource_list)
    require(resources == EXPECTED_RESOURCES and len(resource_list) == len(resources),
            "kustomization.resources must include every production base resource exactly once")
    generator = named_item(kustomization.get("configMapGenerator"),
                           "xllm-service-runtime", "kustomization.configMapGenerator")
    require(generator.get("envs") == ["runtime.env"],
            "runtime ConfigMap must be generated from runtime.env")

    deployment = mapping_at(documents["deployment"], "deployment")
    require(deployment.get("apiVersion") == "apps/v1" and deployment.get("kind") == "Deployment",
            "deployment.yaml must define an apps/v1 Deployment")
    deployment_spec = mapping_at(deployment.get("spec"), "deployment.spec")
    require(deployment_spec.get("replicas", 0) >= 2,
            "production base must start with at least two adapter replicas")
    require(deployment_spec.get("minReadySeconds", 0) > 0,
            "Deployment must require a stable readiness window")
    rolling = mapping_at(mapping_at(deployment_spec.get("strategy"),
                                    "deployment.spec.strategy").get("rollingUpdate"),
                         "deployment.spec.strategy.rollingUpdate")
    require(zero_unavailable(rolling.get("maxUnavailable")),
            "rolling updates must keep maxUnavailable at zero")
    require(positive_surge(rolling.get("maxSurge")),
            "rolling updates must allow at least one surge replica")

    selector = mapping_at(mapping_at(deployment_spec.get("selector"),
                                     "deployment.spec.selector").get("matchLabels"),
                          "deployment.spec.selector.matchLabels")
    template = mapping_at(deployment_spec.get("template"), "deployment.spec.template")
    pod_labels = mapping_at(mapping_at(template.get("metadata"),
                                      "deployment.spec.template.metadata").get("labels"),
                           "deployment.spec.template.metadata.labels")
    require(selector and all(pod_labels.get(key) == value for key, value in selector.items()),
            "Deployment selector must match immutable pod labels")

    pod_spec = mapping_at(template.get("spec"), "deployment.spec.template.spec")
    runtime = mapping_at(documents["runtime"], "runtime.env")
    try:
        runtime_port = int(runtime["XLLM_RUNTIME_PORT"])
        drain_seconds = int(runtime["XLLM_SHUTDOWN_GRACE_PERIOD_S"])
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError("runtime port and shutdown grace period must be integers") from error
    require(1 <= runtime_port <= 65535, "XLLM_RUNTIME_PORT must be a valid TCP port")
    require(drain_seconds >= 0, "XLLM_SHUTDOWN_GRACE_PERIOD_S must be non-negative")
    require(pod_spec.get("terminationGracePeriodSeconds", 0) > drain_seconds,
            "terminationGracePeriodSeconds must exceed the application drain window")
    require(pod_spec.get("automountServiceAccountToken") is False,
            "adapter pods must not mount an unused Kubernetes API token")
    require(pod_spec.get("enableServiceLinks") is False,
            "adapter pods must disable implicit Service environment variables")
    require(pod_spec.get("hostNetwork") is not True and
            pod_spec.get("hostPID") is not True and
            pod_spec.get("hostIPC") is not True,
            "adapter pods must not share host network, PID, or IPC namespaces")

    pod_security = mapping_at(pod_spec.get("securityContext"), "pod securityContext")
    require(pod_security.get("runAsNonRoot") is True,
            "adapter pods must require a non-root image user")
    require(mapping_at(pod_security.get("seccompProfile"), "pod seccompProfile").get("type") ==
            "RuntimeDefault", "adapter pods must use RuntimeDefault seccomp")
    spread = list_at(pod_spec.get("topologySpreadConstraints"), "topologySpreadConstraints")
    require(any(item.get("topologyKey") == "kubernetes.io/hostname" for item in spread
                if isinstance(item, dict)),
            "adapter replicas must have a hostname topology spread constraint")

    container = named_item(pod_spec.get("containers"), "xllm-service", "pod containers")
    image = container.get("image", "")
    require(isinstance(image, str) and image, "xllm-service image must be configured")
    if production:
        require(IMAGE_DIGEST_PATTERN.fullmatch(image) is not None,
                "production image must be pinned by sha256 digest")
    require(container.get("command") == ["xllm_master_serving"],
            "xllm-service must run as PID 1 to receive SIGTERM directly")

    container_security = mapping_at(container.get("securityContext"),
                                    "container securityContext")
    require(container_security.get("allowPrivilegeEscalation") is False,
            "xllm-service must disable privilege escalation")
    require(container_security.get("privileged") is False,
            "xllm-service must explicitly disable privileged mode")
    capabilities = mapping_at(container_security.get("capabilities"), "container capabilities")
    require(capabilities.get("drop") == ["ALL"],
            "xllm-service must drop all Linux capabilities")

    args = argument_map(container.get("args"))
    expected_args = {
        "server_host": "0.0.0.0",
        "http_server_port": "8888",
        "rpc_server_port": "8889",
        "etcd_addr": "$(XLLM_ETCD_ADDR)",
        "etcd_namespace": "$(XLLM_ETCD_NAMESPACE)",
        "routing_mode": "external",
        "external_routing_topology": expected_topology,
        "external_backend_endpoint": "$(POD_IP):$(XLLM_RUNTIME_PORT)",
        "tokenizer_path": "$(XLLM_TOKENIZER_PATH)",
        "shutdown_grace_period_s": "$(XLLM_SHUTDOWN_GRACE_PERIOD_S)",
    }
    require(args == expected_args,
            "container args must match the reviewed external-routing startup contract")

    env = {item.get("name"): item for item in list_at(container.get("env"), "container env")
           if isinstance(item, dict)}
    require(set(env) == RUNTIME_KEYS | {"POD_IP"},
            "container env must contain only the reviewed startup inputs")
    require(mapping_at(env["POD_IP"].get("valueFrom"), "POD_IP.valueFrom")
            .get("fieldRef", {}).get("fieldPath") == "status.podIP",
            "POD_IP must come from the Downward API status.podIP field")
    for key in RUNTIME_KEYS:
        value_from = mapping_at(env[key].get("valueFrom"), f"{key}.valueFrom")
        reference = mapping_at(value_from.get("configMapKeyRef"),
                               f"{key}.configMapKeyRef")
        require(reference.get("name") == "xllm-service-runtime" and
                reference.get("key") == key,
                f"{key} must reference its key in xllm-service-runtime")

    ports = {item.get("name"): item.get("containerPort")
             for item in list_at(container.get("ports"), "container ports")
             if isinstance(item, dict)}
    require(ports == {"http": 8888, "rpc": 8889},
            "container ports must expose only HTTP 8888 and pod-local RPC 8889")
    for probe_name, path in (("readinessProbe", "/readyz"),
                             ("livenessProbe", "/livez"),
                             ("startupProbe", "/livez")):
        probe = mapping_at(container.get(probe_name), probe_name)
        http_get = mapping_at(probe.get("httpGet"), f"{probe_name}.httpGet")
        require(http_get.get("path") == path and http_get.get("port") == "http",
                f"{probe_name} must query {path} on the named HTTP port")
    resources_config = mapping_at(container.get("resources"), "container resources")
    require(resources_config.get("requests") and resources_config.get("limits"),
            "container CPU and memory requests and limits are required")

    service = mapping_at(documents["service"], "service")
    service_spec = mapping_at(service.get("spec"), "service.spec")
    require(service_spec.get("type") == "ClusterIP", "adapter Service must be ClusterIP")
    require(service_spec.get("selector") == selector,
            "Service selector must exactly match the Deployment selector")
    service_ports = list_at(service_spec.get("ports"), "service.spec.ports")
    require(service_ports == [{"name": "http", "port": 8888,
                               "targetPort": "http", "protocol": "TCP"}],
            "Service must expose only HTTP 8888")

    pool = mapping_at(documents["inferencepool"], "inferencepool")
    require(pool.get("apiVersion") == "inference.networking.k8s.io/v1" and
            pool.get("kind") == "InferencePool",
            "inferencepool.yaml must use the GAIE v1 InferencePool contract")
    pool_spec = mapping_at(pool.get("spec"), "inferencepool.spec")
    require(pool_spec.get("selector", {}).get("matchLabels") == selector,
            "InferencePool selector must exactly match the Deployment selector")
    require(pool_spec.get("targetPorts") == [{"number": 8888}] and
            pool_spec.get("appProtocol") == "http",
            "InferencePool must expose only HTTP/1.1 port 8888")
    picker = mapping_at(pool_spec.get("endpointPickerRef"),
                        "inferencepool.spec.endpointPickerRef")
    require(picker.get("name") and picker.get("port") == {"number": 9002} and
            picker.get("failureMode") == "FailClose",
            "InferencePool must use an explicit EPP Service on port 9002 with FailClose")

    pdb = mapping_at(documents["pdb"], "poddisruptionbudget")
    pdb_spec = mapping_at(pdb.get("spec"), "poddisruptionbudget.spec")
    require(pdb.get("apiVersion") == "policy/v1" and pdb_spec.get("minAvailable") == 1,
            "PDB must keep at least one adapter replica available")
    require(pdb_spec.get("selector", {}).get("matchLabels") == selector,
            "PDB selector must exactly match the Deployment selector")

    network_policy = mapping_at(documents["networkpolicy"], "networkpolicy")
    policy_spec = mapping_at(network_policy.get("spec"), "networkpolicy.spec")
    require(policy_spec.get("podSelector", {}).get("matchLabels") == selector,
            "NetworkPolicy selector must exactly match the Deployment selector")
    require(policy_spec.get("policyTypes") == ["Ingress"],
            "base NetworkPolicy must isolate ingress without changing egress")
    ingress = list_at(policy_spec.get("ingress"), "networkpolicy.spec.ingress")
    require(len(ingress) == 2,
            "NetworkPolicy must contain only reviewed HTTP and RPC rules")
    http_rules = [rule for rule in ingress
                  if rule.get("ports") == [{"protocol": "TCP", "port": 8888}]]
    require(len(http_rules) == 1,
            "NetworkPolicy must expose exactly one HTTP 8888 rule")
    if production:
        sources = http_rules[0].get("from")
        require(isinstance(sources, list) and sources and
                all(isinstance(source, dict) and source for source in sources),
                "production HTTP 8888 ingress must be restricted to explicit sources")
    rpc_rule = {
        "from": [{"podSelector": {"matchLabels": selector}}],
        "ports": [{"protocol": "TCP", "port": 8889}],
    }
    require(sum(rule == rpc_rule for rule in ingress) == 1,
            "NetworkPolicy must restrict RPC 8889 to pool pods")

    route = mapping_at(documents["httproute"], "httproute")
    require(route.get("apiVersion") == "gateway.networking.k8s.io/v1" and
            route.get("kind") == "HTTPRoute", "httproute.yaml must define a v1 HTTPRoute")
    route_spec = mapping_at(route.get("spec"), "httproute.spec")
    require(route_spec.get("parentRefs"), "HTTPRoute must reference an installed Gateway")
    rules = list_at(route_spec.get("rules"), "httproute.spec.rules")
    require(len(rules) == 1, "HTTPRoute must have exactly one reviewed inference rule")
    matches = list_at(rules[0].get("matches"), "httproute rule matches")
    route_paths = {match.get("path", {}).get("value") for match in matches
                   if isinstance(match, dict)}
    require(route_paths == PUBLIC_PATHS and all(match.get("path", {}).get("type") == "PathPrefix"
                                                for match in matches),
            "HTTPRoute may expose only the reviewed /v1 API prefixes")
    backend_refs = list_at(rules[0].get("backendRefs"), "httproute backendRefs")
    expected_backend = [{
        "group": "inference.networking.k8s.io",
        "kind": "InferencePool",
        "name": pool.get("metadata", {}).get("name"),
        "port": 8888,
    }]
    require(backend_refs == expected_backend,
            "HTTPRoute must target the reviewed InferencePool HTTP port")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True,
                        help="directory containing the Kustomize base")
    parser.add_argument("--http-route", type=Path, required=True,
                        help="optional production HTTPRoute manifest to validate")
    parser.add_argument("--production", action="store_true",
                        help="also require an immutable image digest")
    parser.add_argument("--expected-topology", choices=("aggregated", "pd"),
                        default="aggregated",
                        help="external routing topology required in the manifest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        documents = load_contract(args.base_dir, args.http_route)
        validate_contract(
            documents,
            production=args.production,
            expected_topology=args.expected_topology,
        )
    except ContractError as error:
        print(f"deployment contract validation failed: {error}", file=sys.stderr)
        return 1
    print("xllm-service Kubernetes deployment contract is valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
