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

#include "placement/placement_actuator.h"
#include "placement/placement_desired_store.h"

namespace xllm_service::placement {

inline constexpr char kPlacementCommandPrefix[] = "XLLM:PLACEMENT:COMMAND/";
inline constexpr char kPlacementStatusPrefix[] = "XLLM:PLACEMENT:STATUS/";
inline constexpr size_t kMaxPlacementOperationBytes = 16384;

struct PlacementPersistedOperation {
  PlacementOperationRecord record;
  int64_t command_revision = 0;
  int64_t status_revision = 0;
};

// Durable, leader-fenced command/status ledger. A command is immutable; its
// status advances with CAS. A command without status recovers as UNKNOWN so a
// new leader queries the actuator instead of replaying a possible side effect.
class PlacementOperationStore final {
 public:
  explicit PlacementOperationStore(PlacementFencedKv* backend);

  PlacementStoreStatus create_command(const PlacementOperationIntent& intent,
                                      const PlacementLeaderIdentity& leader,
                                      int64_t* mod_revision);

  PlacementStoreStatus read_command(const std::string& operation_id,
                                    PlacementOperationIntent* intent,
                                    int64_t* mod_revision);

  PlacementStoreStatus compare_and_set_status(
      const PlacementOperationRecord& record,
      int64_t expected_mod_revision,
      const PlacementLeaderIdentity& leader,
      int64_t* mod_revision);

  PlacementStoreStatus delete_terminal(const PlacementOperationRecord& record,
                                       const PlacementLeaderIdentity& leader);

  PlacementStoreStatus load_snapshot(
      size_t max_records,
      size_t max_total_bytes,
      std::vector<PlacementPersistedOperation>* snapshot);

 private:
  PlacementFencedKv* backend_ = nullptr;
};

bool serialize_placement_operation_intent(
    const PlacementOperationIntent& intent,
    std::string* value);

bool parse_placement_operation_intent(const std::string& value,
                                      PlacementOperationIntent* intent);

bool serialize_placement_operation_record(
    const PlacementOperationRecord& record,
    std::string* value);

bool parse_placement_operation_record(const std::string& value,
                                      PlacementOperationRecord* record);

}  // namespace xllm_service::placement
