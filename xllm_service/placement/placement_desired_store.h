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
#include <vector>

#include "placement/placement_types.h"

namespace xllm_service {
class EtcdClient;
}

namespace xllm_service::placement {

inline constexpr size_t kMaxPlacementDesiredBytes = 16384;
inline constexpr char kPlacementDesiredPrefix[] = "XLLM:PLACEMENT:DESIRED/";

enum class PlacementStoreStatus : int8_t {
  OK = 0,
  NOT_FOUND = 1,
  FENCED = 2,
  REVISION_CONFLICT = 3,
  CORRUPT = 4,
  CAPACITY_EXCEEDED = 5,
  UNAVAILABLE = 6,
  INVALID_INPUT = 7,
};

struct PlacementLeaderIdentity {
  std::string address;
  std::string incarnation;
  // Monotonic etcd mod revision of XLLM:SERVICE:MASTER. Incarnation is an
  // opaque identity; epoch is the ordering/fencing token.
  uint64_t epoch = 0;
};

struct PlacementDesiredState {
  uint32_t schema_version = kPlacementSchemaVersion;
  PlacementLeaderIdentity leader;
  uint64_t generation = 0;
  PlacementPoolKey pool;
  uint32_t desired_replicas = 0;
  PlacementReason reason = PlacementReason::NONE;
  uint64_t observation_generation = 0;
  uint64_t created_at_unix_ms = 0;
  std::string config_digest;
};

struct PlacementRawValue {
  std::string key_suffix;
  std::string value;
  int64_t mod_revision = 0;
};

struct PlacementDesiredSnapshot {
  PlacementDesiredState desired;
  int64_t mod_revision = 0;
};

class PlacementFencedKv {
 public:
  virtual ~PlacementFencedKv() = default;

  virtual PlacementStoreStatus read(const std::string& logical_key,
                                    std::string* value,
                                    int64_t* mod_revision) = 0;

  virtual PlacementStoreStatus list(const std::string& logical_prefix,
                                    std::vector<PlacementRawValue>* values) = 0;

  virtual PlacementStoreStatus compare_and_set(
      const std::string& logical_key,
      const std::string& value,
      int64_t expected_mod_revision,
      const PlacementLeaderIdentity& leader) = 0;
};

class EtcdPlacementFencedKv final : public PlacementFencedKv {
 public:
  explicit EtcdPlacementFencedKv(EtcdClient* client);

  PlacementStoreStatus read(const std::string& logical_key,
                            std::string* value,
                            int64_t* mod_revision) override;

  PlacementStoreStatus list(const std::string& logical_prefix,
                            std::vector<PlacementRawValue>* values) override;

  PlacementStoreStatus compare_and_set(
      const std::string& logical_key,
      const std::string& value,
      int64_t expected_mod_revision,
      const PlacementLeaderIdentity& leader) override;

 private:
  EtcdClient* client_ = nullptr;
};

class PlacementDesiredStore final {
 public:
  explicit PlacementDesiredStore(PlacementFencedKv* backend);

  PlacementStoreStatus read(const PlacementPoolKey& pool,
                            PlacementDesiredSnapshot* snapshot);

  PlacementStoreStatus load_snapshot(
      size_t max_records,
      size_t max_total_bytes,
      std::vector<PlacementDesiredSnapshot>* snapshot);

  PlacementStoreStatus compare_and_set(
      const PlacementDesiredState& desired,
      int64_t expected_mod_revision,
      const PlacementLeaderIdentity& expected_leader);

 private:
  PlacementFencedKv* backend_ = nullptr;
};

bool valid_placement_leader_identity(const PlacementLeaderIdentity& leader);

bool valid_placement_desired_state(const PlacementDesiredState& desired);

std::string placement_pool_key_suffix(const PlacementPoolKey& pool);

bool serialize_placement_desired_state(const PlacementDesiredState& desired,
                                       std::string* value);

bool parse_placement_desired_state(const std::string& value,
                                   PlacementDesiredState* desired);

const char* placement_store_status_name(PlacementStoreStatus status);

}  // namespace xllm_service::placement
