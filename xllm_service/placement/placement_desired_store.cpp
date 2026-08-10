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

#include "placement/placement_desired_store.h"

#include <algorithm>
#include <limits>
#include <nlohmann/json.hpp>
#include <utility>

#include "scheduler/etcd_client/etcd_client.h"

namespace xllm_service::placement {
namespace {

std::string escape_key_component(const std::string& value) {
  constexpr char kHex[] = "0123456789ABCDEF";
  std::string escaped;
  escaped.reserve(value.size());
  for (const unsigned char character : value) {
    const bool unreserved = (character >= 'a' && character <= 'z') ||
                            (character >= 'A' && character <= 'Z') ||
                            (character >= '0' && character <= '9') ||
                            character == '-' || character == '_' ||
                            character == '.' || character == '~';
    if (unreserved) {
      escaped.push_back(static_cast<char>(character));
      continue;
    }
    escaped.push_back('%');
    escaped.push_back(kHex[character >> 4]);
    escaped.push_back(kHex[character & 0x0f]);
  }
  return escaped;
}

PlacementStoreStatus map_read_status(EtcdReadStatus status) {
  switch (status) {
    case EtcdReadStatus::OK:
      return PlacementStoreStatus::OK;
    case EtcdReadStatus::NOT_FOUND:
      return PlacementStoreStatus::NOT_FOUND;
    case EtcdReadStatus::UNAVAILABLE:
      return PlacementStoreStatus::UNAVAILABLE;
    case EtcdReadStatus::INVALID_INPUT:
      return PlacementStoreStatus::INVALID_INPUT;
  }
  return PlacementStoreStatus::UNAVAILABLE;
}

PlacementStoreStatus map_write_status(EtcdFencedWriteStatus status) {
  switch (status) {
    case EtcdFencedWriteStatus::OK:
      return PlacementStoreStatus::OK;
    case EtcdFencedWriteStatus::FENCED:
      return PlacementStoreStatus::FENCED;
    case EtcdFencedWriteStatus::REVISION_CONFLICT:
      return PlacementStoreStatus::REVISION_CONFLICT;
    case EtcdFencedWriteStatus::UNAVAILABLE:
      return PlacementStoreStatus::UNAVAILABLE;
    case EtcdFencedWriteStatus::INVALID_INPUT:
      return PlacementStoreStatus::INVALID_INPUT;
  }
  return PlacementStoreStatus::UNAVAILABLE;
}

bool add_overflows(size_t left, size_t right) {
  return left > std::numeric_limits<size_t>::max() - right;
}

bool json_uint64(const nlohmann::json& json, const char* key, uint64_t* value) {
  if (value == nullptr || !json.contains(key) ||
      !json.at(key).is_number_unsigned()) {
    return false;
  }
  *value = json.at(key).get<uint64_t>();
  return true;
}

bool json_uint32(const nlohmann::json& json, const char* key, uint32_t* value) {
  uint64_t wide_value = 0;
  if (value == nullptr || !json_uint64(json, key, &wide_value) ||
      wide_value > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  *value = static_cast<uint32_t>(wide_value);
  return true;
}

bool json_int32(const nlohmann::json& json, const char* key, int32_t* value) {
  if (value == nullptr || !json.contains(key) ||
      !json.at(key).is_number_integer()) {
    return false;
  }
  const int64_t wide_value = json.at(key).get<int64_t>();
  if (wide_value < std::numeric_limits<int32_t>::min() ||
      wide_value > std::numeric_limits<int32_t>::max()) {
    return false;
  }
  *value = static_cast<int32_t>(wide_value);
  return true;
}

bool json_string(const nlohmann::json& json,
                 const char* key,
                 std::string* value) {
  if (value == nullptr || !json.contains(key) || !json.at(key).is_string()) {
    return false;
  }
  *value = json.at(key).get<std::string>();
  return true;
}

}  // namespace

EtcdPlacementFencedKv::EtcdPlacementFencedKv(EtcdClient* client)
    : client_(client) {}

PlacementStoreStatus EtcdPlacementFencedKv::read(const std::string& logical_key,
                                                 std::string* value,
                                                 int64_t* mod_revision) {
  if (client_ == nullptr) {
    return PlacementStoreStatus::UNAVAILABLE;
  }
  return map_read_status(
      client_->get_with_revision(logical_key, value, mod_revision));
}

PlacementStoreStatus EtcdPlacementFencedKv::list(
    const std::string& logical_prefix,
    std::vector<PlacementRawValue>* values) {
  if (client_ == nullptr || values == nullptr) {
    return PlacementStoreStatus::INVALID_INPUT;
  }
  std::vector<EtcdKeyValue> etcd_values;
  const EtcdReadStatus status =
      client_->get_prefix_with_revision(logical_prefix, &etcd_values);
  if (status != EtcdReadStatus::OK) {
    return map_read_status(status);
  }
  values->clear();
  values->reserve(etcd_values.size());
  for (EtcdKeyValue& value : etcd_values) {
    values->emplace_back(PlacementRawValue{
        .key_suffix = std::move(value.key),
        .value = std::move(value.value),
        .mod_revision = value.mod_revision,
    });
  }
  return PlacementStoreStatus::OK;
}

PlacementStoreStatus EtcdPlacementFencedKv::compare_and_set(
    const std::string& logical_key,
    const std::string& value,
    int64_t expected_mod_revision,
    const PlacementLeaderIdentity& leader) {
  if (client_ == nullptr) {
    return PlacementStoreStatus::UNAVAILABLE;
  }
  return map_write_status(client_->compare_and_set_fenced(logical_key,
                                                          value,
                                                          expected_mod_revision,
                                                          leader.address,
                                                          leader.incarnation,
                                                          leader.epoch));
}

PlacementDesiredStore::PlacementDesiredStore(PlacementFencedKv* backend)
    : backend_(backend) {}

PlacementStoreStatus PlacementDesiredStore::read(
    const PlacementPoolKey& pool,
    PlacementDesiredSnapshot* snapshot) {
  if (backend_ == nullptr || !valid_placement_pool_key(pool) ||
      snapshot == nullptr) {
    return PlacementStoreStatus::INVALID_INPUT;
  }
  std::string value;
  int64_t revision = 0;
  const PlacementStoreStatus status = backend_->read(
      std::string(kPlacementDesiredPrefix) + placement_pool_key_suffix(pool),
      &value,
      &revision);
  if (status != PlacementStoreStatus::OK) {
    return status;
  }
  PlacementDesiredState desired;
  if (revision <= 0 || !parse_placement_desired_state(value, &desired) ||
      !placement_pool_keys_equal(pool, desired.pool)) {
    return PlacementStoreStatus::CORRUPT;
  }
  *snapshot = PlacementDesiredSnapshot{
      .desired = std::move(desired),
      .mod_revision = revision,
  };
  return PlacementStoreStatus::OK;
}

PlacementStoreStatus PlacementDesiredStore::load_snapshot(
    size_t max_records,
    size_t max_total_bytes,
    std::vector<PlacementDesiredSnapshot>* snapshot) {
  if (backend_ == nullptr || max_records == 0 || max_total_bytes == 0 ||
      snapshot == nullptr) {
    return PlacementStoreStatus::INVALID_INPUT;
  }
  std::vector<PlacementRawValue> raw_values;
  const PlacementStoreStatus status =
      backend_->list(kPlacementDesiredPrefix, &raw_values);
  if (status != PlacementStoreStatus::OK) {
    return status;
  }
  if (raw_values.size() > max_records) {
    return PlacementStoreStatus::CAPACITY_EXCEEDED;
  }
  std::sort(raw_values.begin(),
            raw_values.end(),
            [](const PlacementRawValue& left, const PlacementRawValue& right) {
              return left.key_suffix < right.key_suffix;
            });

  size_t total_bytes = 0;
  std::vector<PlacementDesiredSnapshot> parsed;
  parsed.reserve(raw_values.size());
  std::string previous_key;
  for (const PlacementRawValue& raw : raw_values) {
    if (raw.key_suffix.empty() || raw.mod_revision <= 0 ||
        raw.value.size() > kMaxPlacementDesiredBytes ||
        add_overflows(total_bytes, raw.value.size())) {
      return PlacementStoreStatus::CORRUPT;
    }
    total_bytes += raw.value.size();
    if (total_bytes > max_total_bytes || raw.key_suffix == previous_key) {
      return raw.key_suffix == previous_key
                 ? PlacementStoreStatus::CORRUPT
                 : PlacementStoreStatus::CAPACITY_EXCEEDED;
    }
    PlacementDesiredState desired;
    if (!parse_placement_desired_state(raw.value, &desired) ||
        raw.key_suffix != placement_pool_key_suffix(desired.pool)) {
      return PlacementStoreStatus::CORRUPT;
    }
    previous_key = raw.key_suffix;
    parsed.emplace_back(PlacementDesiredSnapshot{
        .desired = std::move(desired),
        .mod_revision = raw.mod_revision,
    });
  }
  *snapshot = std::move(parsed);
  return PlacementStoreStatus::OK;
}

PlacementStoreStatus PlacementDesiredStore::compare_and_set(
    const PlacementDesiredState& desired,
    int64_t expected_mod_revision,
    const PlacementLeaderIdentity& expected_leader) {
  if (backend_ == nullptr || expected_mod_revision < 0 ||
      !valid_placement_desired_state(desired) ||
      !valid_placement_leader_identity(expected_leader) ||
      desired.leader.address != expected_leader.address ||
      desired.leader.incarnation != expected_leader.incarnation ||
      desired.leader.epoch != expected_leader.epoch) {
    return PlacementStoreStatus::INVALID_INPUT;
  }
  if (expected_mod_revision > 0) {
    PlacementDesiredSnapshot current;
    const PlacementStoreStatus read_status = read(desired.pool, &current);
    if (read_status != PlacementStoreStatus::OK) {
      return read_status == PlacementStoreStatus::NOT_FOUND
                 ? PlacementStoreStatus::REVISION_CONFLICT
                 : read_status;
    }
    if (current.mod_revision != expected_mod_revision ||
        desired.generation <= current.desired.generation) {
      return PlacementStoreStatus::REVISION_CONFLICT;
    }
  }
  std::string value;
  if (!serialize_placement_desired_state(desired, &value)) {
    return PlacementStoreStatus::INVALID_INPUT;
  }
  return backend_->compare_and_set(std::string(kPlacementDesiredPrefix) +
                                       placement_pool_key_suffix(desired.pool),
                                   value,
                                   expected_mod_revision,
                                   expected_leader);
}

bool valid_placement_leader_identity(const PlacementLeaderIdentity& leader) {
  return valid_placement_identity(leader.address) &&
         valid_placement_identity(leader.incarnation) && leader.epoch > 0;
}

bool valid_placement_desired_state(const PlacementDesiredState& desired) {
  return desired.schema_version == kPlacementSchemaVersion &&
         valid_placement_leader_identity(desired.leader) &&
         desired.generation > 0 && valid_placement_pool_key(desired.pool) &&
         desired.desired_replicas > 0 &&
         valid_placement_reason(desired.reason) &&
         desired.observation_generation > 0 && desired.created_at_unix_ms > 0 &&
         valid_placement_identity(desired.config_digest);
}

std::string placement_pool_key_suffix(const PlacementPoolKey& pool) {
  if (!valid_placement_pool_key(pool)) {
    return "";
  }
  return std::to_string(static_cast<int32_t>(pool.provider_id)) + "/" +
         std::to_string(static_cast<int32_t>(pool.role)) + "/" +
         escape_key_component(pool.model_revision) + "/" +
         escape_key_component(pool.profile_digest);
}

bool serialize_placement_desired_state(const PlacementDesiredState& desired,
                                       std::string* value) {
  if (value == nullptr || !valid_placement_desired_state(desired)) {
    return false;
  }
  const nlohmann::json json = {
      {"schema_version", desired.schema_version},
      {"leader_address", desired.leader.address},
      {"leader_incarnation", desired.leader.incarnation},
      {"leader_epoch", desired.leader.epoch},
      {"generation", desired.generation},
      {"provider_id", static_cast<int32_t>(desired.pool.provider_id)},
      {"model_revision", desired.pool.model_revision},
      {"role", static_cast<int32_t>(desired.pool.role)},
      {"profile_digest", desired.pool.profile_digest},
      {"desired_replicas", desired.desired_replicas},
      {"reason", static_cast<int32_t>(desired.reason)},
      {"observation_generation", desired.observation_generation},
      {"created_at_unix_ms", desired.created_at_unix_ms},
      {"config_digest", desired.config_digest},
  };
  *value = json.dump();
  return value->size() <= kMaxPlacementDesiredBytes;
}

bool parse_placement_desired_state(const std::string& value,
                                   PlacementDesiredState* desired) {
  if (desired == nullptr || value.empty() ||
      value.size() > kMaxPlacementDesiredBytes) {
    return false;
  }
  const nlohmann::json json =
      nlohmann::json::parse(value, nullptr, /*allow_exceptions=*/false);
  if (json.is_discarded() || !json.is_object()) {
    return false;
  }
  try {
    uint32_t schema_version = 0;
    uint64_t generation = 0;
    uint64_t leader_epoch = 0;
    uint32_t desired_replicas = 0;
    uint64_t observation_generation = 0;
    uint64_t created_at_unix_ms = 0;
    int32_t provider_id = 0;
    int32_t role = 0;
    int32_t reason = 0;
    std::string leader_address;
    std::string leader_incarnation;
    std::string model_revision;
    std::string profile_digest;
    std::string config_digest;
    if (!json_uint32(json, "schema_version", &schema_version) ||
        !json_uint64(json, "generation", &generation) ||
        !json_uint64(json, "leader_epoch", &leader_epoch) ||
        !json_uint32(json, "desired_replicas", &desired_replicas) ||
        !json_uint64(json, "observation_generation", &observation_generation) ||
        !json_uint64(json, "created_at_unix_ms", &created_at_unix_ms) ||
        !json_int32(json, "provider_id", &provider_id) ||
        !json_int32(json, "role", &role) ||
        !json_int32(json, "reason", &reason) ||
        !json_string(json, "leader_address", &leader_address) ||
        !json_string(json, "leader_incarnation", &leader_incarnation) ||
        !json_string(json, "model_revision", &model_revision) ||
        !json_string(json, "profile_digest", &profile_digest) ||
        !json_string(json, "config_digest", &config_digest)) {
      return false;
    }
    PlacementDesiredState parsed{
        .schema_version = schema_version,
        .leader =
            PlacementLeaderIdentity{
                .address = std::move(leader_address),
                .incarnation = std::move(leader_incarnation),
                .epoch = leader_epoch,
            },
        .generation = generation,
        .pool =
            PlacementPoolKey{
                .provider_id =
                    static_cast<xllm::proto::ProviderId>(provider_id),
                .model_revision = std::move(model_revision),
                .role = static_cast<xllm::proto::EngineRole>(role),
                .profile_digest = std::move(profile_digest),
            },
        .desired_replicas = desired_replicas,
        .reason = static_cast<PlacementReason>(reason),
        .observation_generation = observation_generation,
        .created_at_unix_ms = created_at_unix_ms,
        .config_digest = std::move(config_digest),
    };
    if (!valid_placement_desired_state(parsed)) {
      return false;
    }
    *desired = std::move(parsed);
    return true;
  } catch (const nlohmann::json::exception&) {
    return false;
  }
}

const char* placement_store_status_name(PlacementStoreStatus status) {
  switch (status) {
    case PlacementStoreStatus::OK:
      return "OK";
    case PlacementStoreStatus::NOT_FOUND:
      return "NOT_FOUND";
    case PlacementStoreStatus::FENCED:
      return "FENCED";
    case PlacementStoreStatus::REVISION_CONFLICT:
      return "REVISION_CONFLICT";
    case PlacementStoreStatus::CORRUPT:
      return "CORRUPT";
    case PlacementStoreStatus::CAPACITY_EXCEEDED:
      return "CAPACITY_EXCEEDED";
    case PlacementStoreStatus::UNAVAILABLE:
      return "UNAVAILABLE";
    case PlacementStoreStatus::INVALID_INPUT:
      return "INVALID_INPUT";
  }
  return "UNKNOWN";
}

}  // namespace xllm_service::placement
