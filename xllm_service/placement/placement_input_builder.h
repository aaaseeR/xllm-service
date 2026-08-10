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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "placement/placement_controller.h"
#include "placement/placement_observation_collector.h"
#include "provider/engine_registry.h"

namespace xllm_service::placement {

enum class PlacementInputBuildStatus : int8_t {
  OK = 0,
  INVALID_INPUT = 1,
  OBSERVATION_UNAVAILABLE = 2,
  CAPACITY_EXCEEDED = 3,
};

struct PlacementInputBuilderConfig {
  size_t max_pools = 0;
  size_t max_members = 0;
};

struct PlacementPoolRuntimeSpec {
  PlacementCapacityProfile profile;
  uint32_t priority = 0;
  double slo_risk_score = 0.0;
  std::string config_digest;
  PlacementObservationExternalInputs external;
};

// Builds the immutable V3 slow-loop input from the same Registry truth used
// by routing and from the allocation-bounded observation plane. It does not
// retain Registry pointers or take any Scheduler request-path lock.
class PlacementInputBuilder final {
 public:
  explicit PlacementInputBuilder(PlacementInputBuilderConfig config);

  PlacementInputBuildStatus build(
      const std::vector<PlacementPoolRuntimeSpec>& pools,
      const std::vector<provider::EngineRegistryMemberSnapshot>& members,
      PlacementObservationCollector* observations,
      uint64_t now_monotonic_ms,
      std::vector<PlacementPoolCycleInput>* inputs) const;

  bool valid() const;

 private:
  PlacementInputBuilderConfig config_;
  bool valid_ = false;
};

bool valid_placement_input_builder_config(
    const PlacementInputBuilderConfig& config);

const char* placement_input_build_status_name(PlacementInputBuildStatus status);

}  // namespace xllm_service::placement
