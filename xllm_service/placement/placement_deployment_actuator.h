/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <brpc/channel.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "placement/provider_lifecycle_actuator.h"
#include "provider/engine_registry.h"

namespace xllm_service::placement {

struct HttpPlacementDeploymentActuatorConfig {
  std::string address;
  int32_t timeout_ms = 0;
  size_t max_response_bytes = 0;
  std::string internal_api_token;
};

// Generic deployment gateway transport. The gateway owns the provider and
// hardware-specific create/terminate implementation; this client only sends
// a fenced, idempotent operation contract.
class HttpPlacementDeploymentActuator final
    : public PlacementDeploymentActuator {
 public:
  explicit HttpPlacementDeploymentActuator(
      HttpPlacementDeploymentActuatorConfig config);

  PlacementActuatorResponse execute(
      const PlacementOperationIntent& intent) override;
  PlacementActuatorResponse query(
      const PlacementOperationIntent& intent) override;

 private:
  PlacementActuatorResponse call(const PlacementOperationIntent& intent,
                                 bool query);

  HttpPlacementDeploymentActuatorConfig config_;
  std::shared_ptr<brpc::Channel> channel_;
};

// Converts deployment-system completion into controller completion only after
// independent Registry proof. A CREATE requires a fresh schedulable READY
// incarnation. TERMINATE requires Registry absence or an explicit exact
// process-termination proof from the deployment gateway.
class RegistryVerifiedPlacementDeploymentActuator final
    : public PlacementDeploymentActuator {
 public:
  RegistryVerifiedPlacementDeploymentActuator(
      provider::EngineRegistry* registry,
      PlacementDeploymentActuator* backend,
      std::function<uint64_t()> monotonic_clock = {});

  PlacementActuatorResponse execute(
      const PlacementOperationIntent& intent) override;
  PlacementActuatorResponse query(
      const PlacementOperationIntent& intent) override;

 private:
  PlacementActuatorResponse verify(const PlacementOperationIntent& intent,
                                   PlacementActuatorResponse response);

  provider::EngineRegistry* registry_ = nullptr;
  PlacementDeploymentActuator* backend_ = nullptr;
  std::function<uint64_t()> monotonic_clock_;
};

bool parse_placement_deployment_response(const std::string& value,
                                         const PlacementOperationIntent& intent,
                                         PlacementActuatorResponse* response);

}  // namespace xllm_service::placement
