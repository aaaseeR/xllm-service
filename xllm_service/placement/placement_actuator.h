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
#include <map>
#include <string>
#include <vector>

#include "placement/placement_reconciler.h"

namespace xllm_service::placement {

class PlacementOperationStore;

inline constexpr size_t kMaxPlacementActuatorMessageBytes = 1024;

enum class PlacementActuatorCode : int8_t {
  NOT_FOUND = 0,
  ACCEPTED = 1,
  IN_PROGRESS = 2,
  SUCCEEDED = 3,
  RETRYABLE_ERROR = 4,
  TERMINAL_ERROR = 5,
  UNKNOWN = 6,
  FENCED = 7,
  CONFLICT = 8,
  INVALID = 9,
};

enum class PlacementExecutorStatus : int8_t {
  OK = 0,
  INVALID_INPUT = 1,
  CAPACITY_EXCEEDED = 2,
  CLOCK_REGRESSION = 3,
  PERSISTENCE_ERROR = 4,
};

struct PlacementDrainProof {
  bool admission_closed = false;
  uint64_t prefill_queue = 0;
  uint64_t active_transfers = 0;
  uint64_t active_reservations = 0;
  uint64_t decode_sequences = 0;
  uint64_t pending_output = 0;
  uint64_t pending_cleanup = 0;
};

struct PlacementActuatorResponse {
  PlacementActuatorCode code = PlacementActuatorCode::UNKNOWN;
  std::string engine_uid;
  std::string engine_incarnation;
  PlacementLifecycleState lifecycle_state = PlacementLifecycleState::ABSENT;
  bool ready_proven = false;
  PlacementDrainProof drain;
  bool termination_proven = false;
  std::string message;
};

class PlacementActuator {
 public:
  virtual ~PlacementActuator() = default;

  virtual PlacementActuatorResponse execute(
      const PlacementOperationIntent& intent) = 0;

  virtual PlacementActuatorResponse query(
      const PlacementOperationIntent& intent) = 0;
};

struct PlacementOperationRecord {
  PlacementOperationIntent intent;
  PlacementOperationStatus status = PlacementOperationStatus::PLANNED;
  PlacementActuatorCode last_code = PlacementActuatorCode::NOT_FOUND;
  std::string engine_uid;
  std::string engine_incarnation;
  uint32_t execute_attempts = 0;
  uint32_t query_attempts = 0;
  uint64_t created_at_ms = 0;
  uint64_t updated_at_ms = 0;
  bool timed_out = false;
  std::string message;
  int64_t command_revision = 0;
  int64_t status_revision = 0;
};

struct PlacementOperationExecutorConfig {
  size_t max_records = 0;
  size_t max_message_bytes = 0;
  uint64_t operation_timeout_ms = 0;
};

struct PlacementExecutorResult {
  PlacementExecutorStatus status = PlacementExecutorStatus::INVALID_INPUT;
  uint32_t added = 0;
  uint32_t replayed = 0;
  uint32_t driven = 0;
  uint32_t terminal = 0;
  uint32_t unknown = 0;
};

class PlacementOperationExecutor final {
 public:
  PlacementOperationExecutor(PlacementOperationExecutorConfig config,
                             PlacementActuator* actuator,
                             PlacementOperationStore* store = nullptr,
                             PlacementLeaderIdentity leader = {});

  PlacementExecutorStatus recover(uint64_t now_ms, size_t max_total_bytes);

  // Rebuilds the local ledger after leadership acquisition. Persisted
  // monotonic timestamps belong to another process clock domain and are
  // deliberately rebased before any Query is issued.
  PlacementExecutorStatus recover_for_leader(PlacementLeaderIdentity leader,
                                             uint64_t now_ms,
                                             size_t max_total_bytes);

  PlacementExecutorResult add_intents(
      const std::vector<PlacementOperationIntent>& intents,
      uint64_t now_ms);

  PlacementExecutorResult drive(uint64_t now_ms, uint32_t max_actions);

  std::vector<PlacementOperationRecord> snapshot() const;

  std::vector<PlacementOperationView> operation_views() const;

  size_t size() const;

 private:
  PlacementExecutorStatus persist(PlacementOperationRecord* record);

  PlacementOperationExecutorConfig config_;
  PlacementActuator* actuator_ = nullptr;
  PlacementOperationStore* store_ = nullptr;
  PlacementLeaderIdentity leader_;
  std::map<std::string, PlacementOperationRecord> records_;
};

bool valid_placement_operation_intent(const PlacementOperationIntent& intent);

bool placement_operation_intents_equal(const PlacementOperationIntent& left,
                                       const PlacementOperationIntent& right);

bool placement_drain_proof_complete(const PlacementDrainProof& proof);

const char* placement_actuator_code_name(PlacementActuatorCode code);

}  // namespace xllm_service::placement
