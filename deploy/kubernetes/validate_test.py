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

import copy
import sys
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate import ContractError, load_contract, validate_contract


class KubernetesDeploymentContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        manifest_dir = Path(__file__).resolve().parent
        cls.contract = load_contract(manifest_dir / "base",
                                     manifest_dir / "httproute.yaml")

    def contract_copy(self) -> dict[str, Any]:
        return copy.deepcopy(self.contract)

    def production_contract(self) -> dict[str, Any]:
        contract = self.contract_copy()
        container = contract["deployment"]["spec"]["template"]["spec"][
            "containers"
        ][0]
        container["image"] = (
            "registry.example.com/xllm-service@sha256:" + "a" * 64
        )
        contract["networkpolicy"]["spec"]["ingress"][0]["from"] = [
            {
                "namespaceSelector": {
                    "matchLabels": {"kubernetes.io/metadata.name": "gateway"}
                }
            }
        ]
        return contract

    def test_base_contract_is_valid(self) -> None:
        validate_contract(self.contract)

    def test_production_requires_image_digest(self) -> None:
        with self.assertRaisesRegex(ContractError, "sha256 digest"):
            validate_contract(self.contract, production=True)

    def test_production_accepts_image_digest(self) -> None:
        contract = self.production_contract()
        validate_contract(contract, production=True)

    def test_production_rejects_open_http_ingress(self) -> None:
        contract = self.production_contract()
        del contract["networkpolicy"]["spec"]["ingress"][0]["from"]
        with self.assertRaisesRegex(ContractError, "explicit sources"):
            validate_contract(contract, production=True)

    def test_pd_overlay_must_declare_expected_topology(self) -> None:
        contract = self.contract_copy()
        args = contract["deployment"]["spec"]["template"]["spec"][
            "containers"
        ][0]["args"]
        index = args.index("--external_routing_topology=aggregated")
        args[index] = "--external_routing_topology=pd"

        validate_contract(contract, expected_topology="pd")
        with self.assertRaisesRegex(ContractError, "startup contract"):
            validate_contract(contract, expected_topology="aggregated")

    def test_shutdown_window_requires_platform_headroom(self) -> None:
        contract = self.contract_copy()
        contract["deployment"]["spec"]["template"]["spec"][
            "terminationGracePeriodSeconds"
        ] = 30
        with self.assertRaisesRegex(ContractError, "exceed the application drain window"):
            validate_contract(contract)

    def test_external_endpoint_must_use_exact_pod_identity(self) -> None:
        contract = self.contract_copy()
        args = contract["deployment"]["spec"]["template"]["spec"]["containers"][0]["args"]
        args[args.index("--external_backend_endpoint=$(POD_IP):$(XLLM_RUNTIME_PORT)")] = (
            "--external_backend_endpoint=xllm-runtime:8000"
        )
        with self.assertRaisesRegex(ContractError, "external-routing startup contract"):
            validate_contract(contract)

    def test_service_cannot_expose_internal_rpc(self) -> None:
        contract = self.contract_copy()
        contract["service"]["spec"]["ports"].append({
            "name": "rpc", "port": 8889, "targetPort": "rpc", "protocol": "TCP"
        })
        with self.assertRaisesRegex(ContractError, "only HTTP 8888"):
            validate_contract(contract)

    def test_network_policy_cannot_open_rpc_to_all_pods(self) -> None:
        contract = self.contract_copy()
        del contract["networkpolicy"]["spec"]["ingress"][1]["from"]
        with self.assertRaisesRegex(ContractError, "restrict RPC 8889"):
            validate_contract(contract)

    def test_http_route_cannot_expose_operational_endpoints(self) -> None:
        contract = self.contract_copy()
        contract["httproute"]["spec"]["rules"][0]["matches"].append({
            "path": {"type": "PathPrefix", "value": "/metrics"}
        })
        with self.assertRaisesRegex(ContractError, "reviewed /v1 API prefixes"):
            validate_contract(contract)

    def test_http_route_cannot_expose_unsupported_embeddings(self) -> None:
        contract = self.contract_copy()
        contract["httproute"]["spec"]["rules"][0]["matches"].append({
            "path": {"type": "PathPrefix", "value": "/v1/embeddings"}
        })
        with self.assertRaisesRegex(ContractError, "reviewed /v1 API prefixes"):
            validate_contract(contract)

    def test_inference_pool_must_fail_closed(self) -> None:
        contract = self.contract_copy()
        contract["inferencepool"]["spec"]["endpointPickerRef"]["failureMode"] = "FailOpen"
        with self.assertRaisesRegex(ContractError, "FailClose"):
            validate_contract(contract)


if __name__ == "__main__":
    unittest.main()
