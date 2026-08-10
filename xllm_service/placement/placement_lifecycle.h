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

#include <cstdint>
#include <string>

#include "placement/placement_types.h"

namespace xllm_service::placement {

enum class PlacementLifecycleState : int8_t {
  ABSENT = 0,
  LOADING = 1,
  WARMING = 2,
  READY = 3,
  DRAINING = 4,
  UNLOADING = 5,
  FAILED = 6,
};

enum class PlacementLifecycleEvent : int8_t {
  CREATE_ACCEPTED = 0,
  LOAD_COMPLETED = 1,
  WARMUP_COMPLETED = 2,
  BEGIN_DRAIN_ACCEPTED = 3,
  CANCEL_DRAIN_ACCEPTED = 4,
  DRAIN_COMPLETED = 5,
  TERMINATE_COMPLETED = 6,
  OPERATION_FAILED = 7,
  CLEANUP_COMPLETED = 8,
};

enum class PlacementTransitionStatus : int8_t {
  APPLIED = 0,
  REPLAYED = 1,
  INVALID_INPUT = 2,
  FENCED = 3,
  OPERATION_CONFLICT = 4,
  INVALID_TRANSITION = 5,
  CLOCK_REGRESSION = 6,
  GENERATION_EXHAUSTED = 7,
};

struct PlacementLifecycleRecord {
  PlacementPoolKey pool;
  std::string engine_uid;
  std::string engine_incarnation;
  PlacementLifecycleState state = PlacementLifecycleState::ABSENT;
  uint64_t state_generation = 0;
  uint64_t desired_generation = 0;
  std::string operation_id;
  uint64_t applied_event_mask = 0;
  uint64_t last_transition_at_ms = 0;
  bool drain_committed = false;
};

struct PlacementLifecycleCommand {
  PlacementLifecycleEvent event = PlacementLifecycleEvent::CREATE_ACCEPTED;
  std::string engine_uid;
  std::string engine_incarnation;
  std::string operation_id;
  uint64_t desired_generation = 0;
  uint64_t observed_at_ms = 0;
};

struct PlacementTransitionResult {
  PlacementTransitionStatus status = PlacementTransitionStatus::INVALID_INPUT;
  PlacementLifecycleRecord record;
};

bool valid_placement_lifecycle_record(const PlacementLifecycleRecord& record);

bool valid_placement_lifecycle_state(PlacementLifecycleState state);

PlacementTransitionResult apply_placement_lifecycle_event(
    const PlacementLifecycleRecord& record,
    const PlacementLifecycleCommand& command);

const char* placement_lifecycle_state_name(PlacementLifecycleState state);

const char* placement_transition_status_name(PlacementTransitionStatus status);

}  // namespace xllm_service::placement
