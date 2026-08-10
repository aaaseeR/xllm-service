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

#include "placement/placement_lifecycle.h"

#include <limits>

namespace xllm_service::placement {
namespace {

bool valid_state(PlacementLifecycleState state) {
  switch (state) {
    case PlacementLifecycleState::ABSENT:
    case PlacementLifecycleState::LOADING:
    case PlacementLifecycleState::WARMING:
    case PlacementLifecycleState::READY:
    case PlacementLifecycleState::DRAINING:
    case PlacementLifecycleState::UNLOADING:
    case PlacementLifecycleState::FAILED:
      return true;
  }
  return false;
}

bool valid_event(PlacementLifecycleEvent event) {
  switch (event) {
    case PlacementLifecycleEvent::CREATE_ACCEPTED:
    case PlacementLifecycleEvent::LOAD_COMPLETED:
    case PlacementLifecycleEvent::WARMUP_COMPLETED:
    case PlacementLifecycleEvent::BEGIN_DRAIN_ACCEPTED:
    case PlacementLifecycleEvent::CANCEL_DRAIN_ACCEPTED:
    case PlacementLifecycleEvent::DRAIN_COMPLETED:
    case PlacementLifecycleEvent::TERMINATE_COMPLETED:
    case PlacementLifecycleEvent::OPERATION_FAILED:
    case PlacementLifecycleEvent::CLEANUP_COMPLETED:
      return true;
  }
  return false;
}

uint64_t event_mask(PlacementLifecycleEvent event) {
  return uint64_t{1} << static_cast<uint8_t>(event);
}

bool valid_applied_event_mask(uint64_t mask) {
  constexpr uint64_t kAllEventMask = (uint64_t{1} << 9) - 1;
  const uint64_t start_mask =
      event_mask(PlacementLifecycleEvent::CREATE_ACCEPTED) |
      event_mask(PlacementLifecycleEvent::BEGIN_DRAIN_ACCEPTED) |
      event_mask(PlacementLifecycleEvent::CANCEL_DRAIN_ACCEPTED) |
      event_mask(PlacementLifecycleEvent::CLEANUP_COMPLETED);
  if ((mask & ~kAllEventMask) != 0) {
    return false;
  }
  const uint64_t applied_start_mask = mask & start_mask;
  return applied_start_mask != 0 &&
         (applied_start_mask & (applied_start_mask - 1)) == 0;
}

bool starts_operation(PlacementLifecycleEvent event) {
  return event == PlacementLifecycleEvent::CREATE_ACCEPTED ||
         event == PlacementLifecycleEvent::BEGIN_DRAIN_ACCEPTED ||
         event == PlacementLifecycleEvent::CANCEL_DRAIN_ACCEPTED ||
         event == PlacementLifecycleEvent::CLEANUP_COMPLETED;
}

bool matching_engine(const PlacementLifecycleRecord& record,
                     const PlacementLifecycleCommand& command) {
  return record.engine_uid == command.engine_uid &&
         record.engine_incarnation == command.engine_incarnation;
}

bool allowed_transition(PlacementLifecycleState state,
                        PlacementLifecycleEvent event,
                        bool drain_committed,
                        PlacementLifecycleState* next_state) {
  if (next_state == nullptr) {
    return false;
  }
  switch (state) {
    case PlacementLifecycleState::ABSENT:
      if (event == PlacementLifecycleEvent::CREATE_ACCEPTED) {
        *next_state = PlacementLifecycleState::LOADING;
        return true;
      }
      break;
    case PlacementLifecycleState::LOADING:
      if (event == PlacementLifecycleEvent::LOAD_COMPLETED) {
        *next_state = PlacementLifecycleState::WARMING;
        return true;
      }
      if (event == PlacementLifecycleEvent::OPERATION_FAILED) {
        *next_state = PlacementLifecycleState::FAILED;
        return true;
      }
      break;
    case PlacementLifecycleState::WARMING:
      if (event == PlacementLifecycleEvent::WARMUP_COMPLETED) {
        *next_state = PlacementLifecycleState::READY;
        return true;
      }
      if (event == PlacementLifecycleEvent::OPERATION_FAILED) {
        *next_state = PlacementLifecycleState::FAILED;
        return true;
      }
      break;
    case PlacementLifecycleState::READY:
      if (event == PlacementLifecycleEvent::BEGIN_DRAIN_ACCEPTED) {
        *next_state = PlacementLifecycleState::DRAINING;
        return true;
      }
      break;
    case PlacementLifecycleState::DRAINING:
      if (event == PlacementLifecycleEvent::CANCEL_DRAIN_ACCEPTED &&
          !drain_committed) {
        *next_state = PlacementLifecycleState::READY;
        return true;
      }
      if (event == PlacementLifecycleEvent::DRAIN_COMPLETED) {
        *next_state = PlacementLifecycleState::UNLOADING;
        return true;
      }
      if (event == PlacementLifecycleEvent::OPERATION_FAILED) {
        *next_state = PlacementLifecycleState::FAILED;
        return true;
      }
      break;
    case PlacementLifecycleState::UNLOADING:
      if (event == PlacementLifecycleEvent::TERMINATE_COMPLETED) {
        *next_state = PlacementLifecycleState::ABSENT;
        return true;
      }
      if (event == PlacementLifecycleEvent::OPERATION_FAILED) {
        *next_state = PlacementLifecycleState::FAILED;
        return true;
      }
      break;
    case PlacementLifecycleState::FAILED:
      if (event == PlacementLifecycleEvent::CLEANUP_COMPLETED) {
        *next_state = PlacementLifecycleState::ABSENT;
        return true;
      }
      break;
  }
  return false;
}

PlacementTransitionResult result(PlacementTransitionStatus status,
                                 const PlacementLifecycleRecord& record) {
  return PlacementTransitionResult{
      .status = status,
      .record = record,
  };
}

}  // namespace

bool valid_placement_lifecycle_state(PlacementLifecycleState state) {
  return valid_state(state);
}

bool valid_placement_lifecycle_record(const PlacementLifecycleRecord& record) {
  if (!valid_placement_pool_key(record.pool) || !valid_state(record.state) ||
      record.state_generation == std::numeric_limits<uint64_t>::max()) {
    return false;
  }
  const bool has_engine =
      !record.engine_uid.empty() || !record.engine_incarnation.empty();
  if (has_engine && (!valid_placement_identity(record.engine_uid) ||
                     !valid_placement_identity(record.engine_incarnation))) {
    return false;
  }
  const bool has_operation = !record.operation_id.empty();
  if (has_operation != (record.desired_generation != 0) ||
      (has_operation && !valid_placement_identity(record.operation_id)) ||
      (has_operation && !valid_applied_event_mask(record.applied_event_mask)) ||
      (!has_operation && record.applied_event_mask != 0)) {
    return false;
  }
  if (record.state != PlacementLifecycleState::ABSENT &&
      (!has_engine || !has_operation)) {
    return false;
  }
  if (record.drain_committed &&
      record.state != PlacementLifecycleState::UNLOADING &&
      record.state != PlacementLifecycleState::FAILED &&
      record.state != PlacementLifecycleState::ABSENT) {
    return false;
  }
  return true;
}

PlacementTransitionResult apply_placement_lifecycle_event(
    const PlacementLifecycleRecord& record,
    const PlacementLifecycleCommand& command) {
  if (!valid_placement_lifecycle_record(record) ||
      !valid_event(command.event) ||
      !valid_placement_identity(command.engine_uid) ||
      !valid_placement_identity(command.engine_incarnation) ||
      !valid_placement_identity(command.operation_id) ||
      command.desired_generation == 0 || command.observed_at_ms == 0) {
    return result(PlacementTransitionStatus::INVALID_INPUT, record);
  }
  if (command.desired_generation < record.desired_generation) {
    return result(PlacementTransitionStatus::FENCED, record);
  }

  const bool same_operation =
      command.operation_id == record.operation_id &&
      command.desired_generation == record.desired_generation;
  if (same_operation && !matching_engine(record, command)) {
    return result(PlacementTransitionStatus::FENCED, record);
  }
  if (same_operation &&
      (record.applied_event_mask & event_mask(command.event)) != 0) {
    return result(PlacementTransitionStatus::REPLAYED, record);
  }
  if (record.last_transition_at_ms != 0 &&
      command.observed_at_ms < record.last_transition_at_ms) {
    return result(PlacementTransitionStatus::CLOCK_REGRESSION, record);
  }

  const bool new_operation = !same_operation;
  if (new_operation) {
    if (command.desired_generation <= record.desired_generation) {
      return result(PlacementTransitionStatus::OPERATION_CONFLICT, record);
    }
    if (!starts_operation(command.event)) {
      return result(PlacementTransitionStatus::OPERATION_CONFLICT, record);
    }
    if (command.event != PlacementLifecycleEvent::CREATE_ACCEPTED &&
        !matching_engine(record, command)) {
      return result(PlacementTransitionStatus::FENCED, record);
    }
    if (command.event == PlacementLifecycleEvent::CREATE_ACCEPTED &&
        !record.engine_uid.empty() && matching_engine(record, command)) {
      return result(PlacementTransitionStatus::FENCED, record);
    }
  }

  PlacementLifecycleState next_state = record.state;
  if (!allowed_transition(
          record.state, command.event, record.drain_committed, &next_state)) {
    return result(PlacementTransitionStatus::INVALID_TRANSITION, record);
  }
  if (record.state_generation == std::numeric_limits<uint64_t>::max() - 1) {
    return result(PlacementTransitionStatus::GENERATION_EXHAUSTED, record);
  }

  PlacementLifecycleRecord next = record;
  if (new_operation) {
    next.operation_id = command.operation_id;
    next.desired_generation = command.desired_generation;
    next.applied_event_mask = 0;
  }
  if (command.event == PlacementLifecycleEvent::CREATE_ACCEPTED) {
    next.engine_uid = command.engine_uid;
    next.engine_incarnation = command.engine_incarnation;
    next.drain_committed = false;
  } else if (command.event == PlacementLifecycleEvent::DRAIN_COMPLETED) {
    next.drain_committed = true;
  } else if (command.event == PlacementLifecycleEvent::CANCEL_DRAIN_ACCEPTED) {
    next.drain_committed = false;
  }
  next.state = next_state;
  ++next.state_generation;
  next.applied_event_mask |= event_mask(command.event);
  next.last_transition_at_ms = command.observed_at_ms;
  return result(PlacementTransitionStatus::APPLIED, next);
}

const char* placement_lifecycle_state_name(PlacementLifecycleState state) {
  switch (state) {
    case PlacementLifecycleState::ABSENT:
      return "ABSENT";
    case PlacementLifecycleState::LOADING:
      return "LOADING";
    case PlacementLifecycleState::WARMING:
      return "WARMING";
    case PlacementLifecycleState::READY:
      return "READY";
    case PlacementLifecycleState::DRAINING:
      return "DRAINING";
    case PlacementLifecycleState::UNLOADING:
      return "UNLOADING";
    case PlacementLifecycleState::FAILED:
      return "FAILED";
  }
  return "UNKNOWN";
}

const char* placement_transition_status_name(PlacementTransitionStatus status) {
  switch (status) {
    case PlacementTransitionStatus::APPLIED:
      return "APPLIED";
    case PlacementTransitionStatus::REPLAYED:
      return "REPLAYED";
    case PlacementTransitionStatus::INVALID_INPUT:
      return "INVALID_INPUT";
    case PlacementTransitionStatus::FENCED:
      return "FENCED";
    case PlacementTransitionStatus::OPERATION_CONFLICT:
      return "OPERATION_CONFLICT";
    case PlacementTransitionStatus::INVALID_TRANSITION:
      return "INVALID_TRANSITION";
    case PlacementTransitionStatus::CLOCK_REGRESSION:
      return "CLOCK_REGRESSION";
    case PlacementTransitionStatus::GENERATION_EXHAUSTED:
      return "GENERATION_EXHAUSTED";
  }
  return "UNKNOWN";
}

}  // namespace xllm_service::placement
