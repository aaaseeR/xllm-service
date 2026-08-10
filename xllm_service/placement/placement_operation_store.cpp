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

#include "placement/placement_operation_store.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <utility>

namespace xllm_service::placement {
namespace {

bool add_overflows(size_t left, size_t right) {
  return left > std::numeric_limits<size_t>::max() - right;
}

std::string bounded_log_value(const std::string& value) {
  constexpr size_t kMaxLoggedBytes = 64;
  constexpr char kHex[] = "0123456789ABCDEF";
  std::string escaped;
  escaped.reserve(std::min(value.size(), kMaxLoggedBytes) * 3);
  const size_t limit = std::min(value.size(), kMaxLoggedBytes);
  for (size_t index = 0; index < limit; ++index) {
    const unsigned char character = static_cast<unsigned char>(value[index]);
    const bool unreserved = (character >= 'a' && character <= 'z') ||
                            (character >= 'A' && character <= 'Z') ||
                            (character >= '0' && character <= '9') ||
                            character == '-' || character == '_' ||
                            character == '.' || character == ':';
    if (unreserved) {
      escaped.push_back(static_cast<char>(character));
    } else {
      escaped.push_back('%');
      escaped.push_back(kHex[character >> 4]);
      escaped.push_back(kHex[character & 0x0f]);
    }
  }
  if (value.size() > limit) {
    escaped.append("...");
  }
  return escaped;
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
  uint64_t wide = 0;
  if (value == nullptr || !json_uint64(json, key, &wide) ||
      wide > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  *value = static_cast<uint32_t>(wide);
  return true;
}

bool json_int32(const nlohmann::json& json, const char* key, int32_t* value) {
  if (value == nullptr || !json.contains(key) ||
      !json.at(key).is_number_integer()) {
    return false;
  }
  const int64_t wide = json.at(key).get<int64_t>();
  if (wide < std::numeric_limits<int32_t>::min() ||
      wide > std::numeric_limits<int32_t>::max()) {
    return false;
  }
  *value = static_cast<int32_t>(wide);
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

nlohmann::json intent_json(const PlacementOperationIntent& intent) {
  return nlohmann::json{
      {"schema_version", kPlacementSchemaVersion},
      {"operation_id", intent.operation_id},
      {"action", static_cast<int32_t>(intent.action)},
      {"provider_id", static_cast<int32_t>(intent.pool.provider_id)},
      {"model_revision", intent.pool.model_revision},
      {"role", static_cast<int32_t>(intent.pool.role)},
      {"profile_digest", intent.pool.profile_digest},
      {"engine_uid", intent.engine_uid},
      {"engine_incarnation", intent.engine_incarnation},
      {"leader_incarnation", intent.leader_incarnation},
      {"leader_epoch", intent.leader_epoch},
      {"desired_generation", intent.desired_generation},
      {"ordinal", intent.ordinal},
  };
}

bool parse_intent_json(const nlohmann::json& json,
                       PlacementOperationIntent* intent) {
  uint32_t schema_version = 0;
  uint64_t ordinal = 0;
  uint64_t leader_epoch = 0;
  uint64_t desired_generation = 0;
  int32_t action = 0;
  int32_t provider_id = 0;
  int32_t role = 0;
  std::string operation_id;
  std::string model_revision;
  std::string profile_digest;
  std::string engine_uid;
  std::string engine_incarnation;
  std::string leader_incarnation;
  if (intent == nullptr || !json.is_object() ||
      !json_uint32(json, "schema_version", &schema_version) ||
      schema_version != kPlacementSchemaVersion ||
      !json_string(json, "operation_id", &operation_id) ||
      !json_int32(json, "action", &action) ||
      !json_int32(json, "provider_id", &provider_id) ||
      !json_string(json, "model_revision", &model_revision) ||
      !json_int32(json, "role", &role) ||
      !json_string(json, "profile_digest", &profile_digest) ||
      !json_string(json, "engine_uid", &engine_uid) ||
      !json_string(json, "engine_incarnation", &engine_incarnation) ||
      !json_string(json, "leader_incarnation", &leader_incarnation) ||
      !json_uint64(json, "leader_epoch", &leader_epoch) ||
      !json_uint64(json, "desired_generation", &desired_generation) ||
      !json_uint64(json, "ordinal", &ordinal)) {
    return false;
  }
  PlacementOperationIntent parsed{
      .operation_id = std::move(operation_id),
      .action = static_cast<PlacementOperationAction>(action),
      .pool =
          PlacementPoolKey{
              .provider_id = static_cast<xllm::proto::ProviderId>(provider_id),
              .model_revision = std::move(model_revision),
              .role = static_cast<xllm::proto::EngineRole>(role),
              .profile_digest = std::move(profile_digest),
          },
      .engine_uid = std::move(engine_uid),
      .engine_incarnation = std::move(engine_incarnation),
      .leader_incarnation = std::move(leader_incarnation),
      .leader_epoch = leader_epoch,
      .desired_generation = desired_generation,
      .ordinal = ordinal,
  };
  if (!valid_placement_operation_intent(parsed)) {
    return false;
  }
  *intent = std::move(parsed);
  return true;
}

bool valid_record(const PlacementOperationRecord& record) {
  const bool valid_status =
      record.status == PlacementOperationStatus::PLANNED ||
      record.status == PlacementOperationStatus::SUBMITTED ||
      record.status == PlacementOperationStatus::IN_PROGRESS ||
      record.status == PlacementOperationStatus::UNKNOWN ||
      record.status == PlacementOperationStatus::SUCCEEDED ||
      record.status == PlacementOperationStatus::FAILED ||
      record.status == PlacementOperationStatus::FENCED ||
      record.status == PlacementOperationStatus::CANCELED;
  const bool valid_cancellation =
      record.status != PlacementOperationStatus::CANCELED ||
      (record.intent.action == PlacementOperationAction::BEGIN_DRAIN &&
       record.last_code == PlacementActuatorCode::SUCCEEDED);
  const bool valid_code =
      record.last_code == PlacementActuatorCode::NOT_FOUND ||
      record.last_code == PlacementActuatorCode::ACCEPTED ||
      record.last_code == PlacementActuatorCode::IN_PROGRESS ||
      record.last_code == PlacementActuatorCode::SUCCEEDED ||
      record.last_code == PlacementActuatorCode::RETRYABLE_ERROR ||
      record.last_code == PlacementActuatorCode::TERMINAL_ERROR ||
      record.last_code == PlacementActuatorCode::UNKNOWN ||
      record.last_code == PlacementActuatorCode::FENCED ||
      record.last_code == PlacementActuatorCode::CONFLICT ||
      record.last_code == PlacementActuatorCode::INVALID;
  const bool valid_observed_identity =
      record.engine_uid.empty() == record.engine_incarnation.empty() &&
      (record.engine_uid.empty() ||
       (valid_placement_identity(record.engine_uid) &&
        valid_placement_identity(record.engine_incarnation)));
  const bool valid_message =
      record.message.size() <= kMaxPlacementActuatorMessageBytes &&
      std::all_of(record.message.begin(),
                  record.message.end(),
                  [](unsigned char character) {
                    return character != 0 && character != '\r' &&
                           character != '\n';
                  });
  return valid_placement_operation_intent(record.intent) && valid_status &&
         valid_cancellation && valid_code && valid_observed_identity &&
         valid_message && record.created_at_ms > 0 &&
         record.updated_at_ms > 0 &&
         record.updated_at_ms >= record.created_at_ms;
}

}  // namespace

PlacementOperationStore::PlacementOperationStore(PlacementFencedKv* backend)
    : backend_(backend) {}

PlacementStoreStatus PlacementOperationStore::create_command(
    const PlacementOperationIntent& intent,
    const PlacementLeaderIdentity& leader,
    int64_t* mod_revision) {
  if (backend_ == nullptr || mod_revision == nullptr ||
      !valid_placement_operation_intent(intent) ||
      !valid_placement_leader_identity(leader) ||
      intent.leader_incarnation != leader.incarnation ||
      intent.leader_epoch != leader.epoch) {
    return PlacementStoreStatus::INVALID_INPUT;
  }
  std::string value;
  if (!serialize_placement_operation_intent(intent, &value)) {
    return PlacementStoreStatus::INVALID_INPUT;
  }
  const std::string key =
      std::string(kPlacementCommandPrefix) + intent.operation_id;
  const PlacementStoreStatus status =
      backend_->compare_and_set(key, value, 0, leader);
  if (status == PlacementStoreStatus::REVISION_CONFLICT) {
    PlacementOperationIntent existing;
    const PlacementStoreStatus read_status =
        read_command(intent.operation_id, &existing, mod_revision);
    if (read_status != PlacementStoreStatus::OK) {
      return read_status;
    }
    return placement_operation_intents_equal(existing, intent)
               ? PlacementStoreStatus::OK
               : PlacementStoreStatus::REVISION_CONFLICT;
  }
  if (status != PlacementStoreStatus::OK) {
    return status;
  }
  std::string stored;
  return backend_->read(key, &stored, mod_revision);
}

PlacementStoreStatus PlacementOperationStore::read_command(
    const std::string& operation_id,
    PlacementOperationIntent* intent,
    int64_t* mod_revision) {
  if (backend_ == nullptr || !valid_placement_identity(operation_id) ||
      intent == nullptr || mod_revision == nullptr) {
    return PlacementStoreStatus::INVALID_INPUT;
  }
  std::string value;
  const PlacementStoreStatus status =
      backend_->read(std::string(kPlacementCommandPrefix) + operation_id,
                     &value,
                     mod_revision);
  if (status != PlacementStoreStatus::OK) {
    return status;
  }
  if (*mod_revision <= 0 || !parse_placement_operation_intent(value, intent) ||
      intent->operation_id != operation_id) {
    return PlacementStoreStatus::CORRUPT;
  }
  return PlacementStoreStatus::OK;
}

PlacementStoreStatus PlacementOperationStore::compare_and_set_status(
    const PlacementOperationRecord& record,
    int64_t expected_mod_revision,
    const PlacementLeaderIdentity& leader,
    int64_t* mod_revision) {
  if (backend_ == nullptr || mod_revision == nullptr ||
      expected_mod_revision < 0 || !valid_record(record) ||
      !valid_placement_leader_identity(leader)) {
    return PlacementStoreStatus::INVALID_INPUT;
  }
  std::string value;
  if (!serialize_placement_operation_record(record, &value)) {
    return PlacementStoreStatus::INVALID_INPUT;
  }
  const std::string key =
      std::string(kPlacementStatusPrefix) + record.intent.operation_id;
  const PlacementStoreStatus status =
      backend_->compare_and_set(key, value, expected_mod_revision, leader);
  if (status != PlacementStoreStatus::OK) {
    return status;
  }
  std::string stored;
  return backend_->read(key, &stored, mod_revision);
}

PlacementStoreStatus PlacementOperationStore::delete_terminal(
    const PlacementOperationRecord& record,
    const PlacementLeaderIdentity& leader) {
  if (backend_ == nullptr || !valid_record(record) ||
      !placement_operation_terminal(record.status) ||
      record.command_revision <= 0 || record.status_revision <= 0 ||
      !valid_placement_leader_identity(leader)) {
    return PlacementStoreStatus::INVALID_INPUT;
  }
  return backend_->compare_and_delete_pair(
      std::string(kPlacementStatusPrefix) + record.intent.operation_id,
      record.status_revision,
      std::string(kPlacementCommandPrefix) + record.intent.operation_id,
      record.command_revision,
      leader);
}

PlacementStoreStatus PlacementOperationStore::load_snapshot(
    size_t max_records,
    size_t max_total_bytes,
    std::vector<PlacementPersistedOperation>* snapshot) {
  if (backend_ == nullptr || max_records == 0 || max_total_bytes == 0 ||
      snapshot == nullptr) {
    return PlacementStoreStatus::INVALID_INPUT;
  }
  std::vector<PlacementRawValue> commands;
  std::vector<PlacementRawValue> statuses;
  PlacementStoreStatus status =
      backend_->list(kPlacementCommandPrefix, &commands);
  if (status != PlacementStoreStatus::OK) {
    return status;
  }
  status = backend_->list(kPlacementStatusPrefix, &statuses);
  if (status != PlacementStoreStatus::OK) {
    return status;
  }
  if (commands.size() > max_records || statuses.size() > max_records) {
    return PlacementStoreStatus::CAPACITY_EXCEEDED;
  }
  size_t total_bytes = 0;
  std::map<std::string, PlacementRawValue> status_by_id;
  for (PlacementRawValue& raw : statuses) {
    if (raw.key_suffix.empty() || raw.mod_revision <= 0 ||
        raw.value.size() > kMaxPlacementOperationBytes ||
        add_overflows(total_bytes, raw.value.size())) {
      return PlacementStoreStatus::CORRUPT;
    }
    total_bytes += raw.value.size();
    const std::string operation_id = raw.key_suffix;
    if (total_bytes > max_total_bytes ||
        !status_by_id.emplace(operation_id, std::move(raw)).second) {
      return total_bytes > max_total_bytes
                 ? PlacementStoreStatus::CAPACITY_EXCEEDED
                 : PlacementStoreStatus::CORRUPT;
    }
  }
  std::sort(commands.begin(),
            commands.end(),
            [](const PlacementRawValue& left, const PlacementRawValue& right) {
              return left.key_suffix < right.key_suffix;
            });
  std::vector<PlacementPersistedOperation> parsed;
  parsed.reserve(commands.size());
  std::string previous_id;
  for (const PlacementRawValue& raw : commands) {
    if (raw.key_suffix.empty() || raw.key_suffix == previous_id ||
        raw.mod_revision <= 0 ||
        raw.value.size() > kMaxPlacementOperationBytes ||
        add_overflows(total_bytes, raw.value.size())) {
      return PlacementStoreStatus::CORRUPT;
    }
    total_bytes += raw.value.size();
    if (total_bytes > max_total_bytes) {
      return PlacementStoreStatus::CAPACITY_EXCEEDED;
    }
    PlacementOperationIntent intent;
    if (!parse_placement_operation_intent(raw.value, &intent) ||
        intent.operation_id != raw.key_suffix) {
      return PlacementStoreStatus::CORRUPT;
    }
    PlacementOperationRecord record{
        .intent = intent,
        .status = PlacementOperationStatus::UNKNOWN,
        .last_code = PlacementActuatorCode::UNKNOWN,
        .execute_attempts = 1,
        .created_at_ms = 1,
        .updated_at_ms = 1,
        .message = "operation status missing; query required",
    };
    int64_t status_revision = 0;
    const auto status_iterator = status_by_id.find(intent.operation_id);
    if (status_iterator != status_by_id.end()) {
      if (!parse_placement_operation_record(status_iterator->second.value,
                                            &record) ||
          !placement_operation_intents_equal(record.intent, intent)) {
        return PlacementStoreStatus::CORRUPT;
      }
      status_revision = status_iterator->second.mod_revision;
      status_by_id.erase(status_iterator);
    }
    parsed.emplace_back(PlacementPersistedOperation{
        .record = std::move(record),
        .command_revision = raw.mod_revision,
        .status_revision = status_revision,
    });
    previous_id = raw.key_suffix;
  }
  if (!status_by_id.empty()) {
    constexpr size_t kMaxLoggedOrphans = 4;
    std::string orphan_ids;
    size_t logged = 0;
    for (const auto& [operation_id, raw] : status_by_id) {
      static_cast<void>(raw);
      if (logged == kMaxLoggedOrphans) {
        break;
      }
      if (!orphan_ids.empty()) {
        orphan_ids.push_back(',');
      }
      orphan_ids.append(bounded_log_value(operation_id));
      ++logged;
    }
    LOG(ERROR) << "Corrupt V3 Placement snapshot has orphan STATUS records, "
               << "count=" << status_by_id.size()
               << ", sampled_operation_ids=" << orphan_ids;
    return PlacementStoreStatus::CORRUPT;
  }
  *snapshot = std::move(parsed);
  return PlacementStoreStatus::OK;
}

bool serialize_placement_operation_intent(
    const PlacementOperationIntent& intent,
    std::string* value) {
  if (value == nullptr || !valid_placement_operation_intent(intent)) {
    return false;
  }
  *value = intent_json(intent).dump();
  return value->size() <= kMaxPlacementOperationBytes;
}

bool parse_placement_operation_intent(const std::string& value,
                                      PlacementOperationIntent* intent) {
  if (intent == nullptr || value.empty() ||
      value.size() > kMaxPlacementOperationBytes) {
    return false;
  }
  const nlohmann::json json =
      nlohmann::json::parse(value, nullptr, /*allow_exceptions=*/false);
  if (json.is_discarded()) {
    return false;
  }
  try {
    return parse_intent_json(json, intent);
  } catch (const nlohmann::json::exception&) {
    return false;
  }
}

bool serialize_placement_operation_record(
    const PlacementOperationRecord& record,
    std::string* value) {
  if (value == nullptr || !valid_record(record)) {
    return false;
  }
  nlohmann::json json = intent_json(record.intent);
  json["status"] = static_cast<int32_t>(record.status);
  json["last_code"] = static_cast<int32_t>(record.last_code);
  json["observed_engine_uid"] = record.engine_uid;
  json["observed_engine_incarnation"] = record.engine_incarnation;
  json["execute_attempts"] = record.execute_attempts;
  json["query_attempts"] = record.query_attempts;
  json["created_at_ms"] = record.created_at_ms;
  json["updated_at_ms"] = record.updated_at_ms;
  json["timed_out"] = record.timed_out;
  json["message"] = record.message;
  *value = json.dump();
  return value->size() <= kMaxPlacementOperationBytes;
}

bool parse_placement_operation_record(const std::string& value,
                                      PlacementOperationRecord* record) {
  if (record == nullptr || value.empty() ||
      value.size() > kMaxPlacementOperationBytes) {
    return false;
  }
  const nlohmann::json json =
      nlohmann::json::parse(value, nullptr, /*allow_exceptions=*/false);
  if (json.is_discarded() || !json.is_object()) {
    return false;
  }
  try {
    PlacementOperationIntent intent;
    int32_t status = 0;
    int32_t last_code = 0;
    uint32_t execute_attempts = 0;
    uint32_t query_attempts = 0;
    uint64_t created_at_ms = 0;
    uint64_t updated_at_ms = 0;
    std::string engine_uid;
    std::string engine_incarnation;
    std::string message;
    if (!parse_intent_json(json, &intent) ||
        !json_int32(json, "status", &status) ||
        !json_int32(json, "last_code", &last_code) ||
        !json_string(json, "observed_engine_uid", &engine_uid) ||
        !json_string(
            json, "observed_engine_incarnation", &engine_incarnation) ||
        !json_uint32(json, "execute_attempts", &execute_attempts) ||
        !json_uint32(json, "query_attempts", &query_attempts) ||
        !json_uint64(json, "created_at_ms", &created_at_ms) ||
        !json_uint64(json, "updated_at_ms", &updated_at_ms) ||
        !json.contains("timed_out") || !json.at("timed_out").is_boolean() ||
        !json_string(json, "message", &message)) {
      return false;
    }
    PlacementOperationRecord parsed{
        .intent = std::move(intent),
        .status = static_cast<PlacementOperationStatus>(status),
        .last_code = static_cast<PlacementActuatorCode>(last_code),
        .engine_uid = std::move(engine_uid),
        .engine_incarnation = std::move(engine_incarnation),
        .execute_attempts = execute_attempts,
        .query_attempts = query_attempts,
        .created_at_ms = created_at_ms,
        .updated_at_ms = updated_at_ms,
        .timed_out = json.at("timed_out").get<bool>(),
        .message = std::move(message),
    };
    if (!valid_record(parsed)) {
      return false;
    }
    *record = std::move(parsed);
    return true;
  } catch (const nlohmann::json::exception&) {
    return false;
  }
}

}  // namespace xllm_service::placement
