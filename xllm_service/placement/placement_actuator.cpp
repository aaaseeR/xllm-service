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

#include "placement/placement_actuator.h"

#include <algorithm>
#include <utility>

#include "placement/placement_operation_store.h"

namespace xllm_service::placement {
namespace {

bool valid_actuator_code(PlacementActuatorCode code) {
  switch (code) {
    case PlacementActuatorCode::NOT_FOUND:
    case PlacementActuatorCode::ACCEPTED:
    case PlacementActuatorCode::IN_PROGRESS:
    case PlacementActuatorCode::SUCCEEDED:
    case PlacementActuatorCode::RETRYABLE_ERROR:
    case PlacementActuatorCode::TERMINAL_ERROR:
    case PlacementActuatorCode::UNKNOWN:
    case PlacementActuatorCode::FENCED:
    case PlacementActuatorCode::CONFLICT:
    case PlacementActuatorCode::INVALID:
      return true;
  }
  return false;
}

bool valid_config(const PlacementOperationExecutorConfig& config) {
  return config.max_records > 0 && config.max_message_bytes > 0 &&
         config.max_message_bytes <= kMaxPlacementActuatorMessageBytes &&
         config.operation_timeout_ms > 0;
}

bool response_identity_matches(const PlacementOperationRecord& record,
                               const PlacementActuatorResponse& response) {
  const bool has_uid = !response.engine_uid.empty();
  const bool has_incarnation = !response.engine_incarnation.empty();
  if (has_uid != has_incarnation ||
      (has_uid && (!valid_placement_identity(response.engine_uid) ||
                   !valid_placement_identity(response.engine_incarnation)))) {
    return false;
  }
  if (record.intent.action == PlacementOperationAction::CREATE) {
    if (has_uid && !record.engine_uid.empty() &&
        (response.engine_uid != record.engine_uid ||
         response.engine_incarnation != record.engine_incarnation)) {
      return false;
    }
    if (response.code != PlacementActuatorCode::SUCCEEDED) {
      return true;
    }
    return valid_placement_identity(response.engine_uid) &&
           valid_placement_identity(response.engine_incarnation);
  }
  if (!has_uid) {
    return response.code == PlacementActuatorCode::NOT_FOUND ||
           response.code == PlacementActuatorCode::RETRYABLE_ERROR ||
           response.code == PlacementActuatorCode::TERMINAL_ERROR ||
           response.code == PlacementActuatorCode::UNKNOWN ||
           response.code == PlacementActuatorCode::FENCED ||
           response.code == PlacementActuatorCode::CONFLICT ||
           response.code == PlacementActuatorCode::INVALID;
  }
  return response.engine_uid == record.intent.engine_uid &&
         response.engine_incarnation == record.intent.engine_incarnation;
}

bool valid_message(const std::string& message) {
  return message.size() <= kMaxPlacementActuatorMessageBytes &&
         std::all_of(message.begin(), message.end(), [](unsigned char value) {
           return value != 0 && value != '\r' && value != '\n';
         });
}

bool success_proven(const PlacementOperationRecord& record,
                    const PlacementActuatorResponse& response) {
  switch (record.intent.action) {
    case PlacementOperationAction::CREATE:
      return response.ready_proven &&
             response.lifecycle_state == PlacementLifecycleState::READY;
    case PlacementOperationAction::BEGIN_DRAIN:
      return response.lifecycle_state == PlacementLifecycleState::DRAINING &&
             placement_drain_proof_complete(response.drain);
    case PlacementOperationAction::CANCEL_DRAIN:
      return response.lifecycle_state == PlacementLifecycleState::READY &&
             !response.drain.admission_closed;
    case PlacementOperationAction::TERMINATE:
      return response.lifecycle_state == PlacementLifecycleState::ABSENT &&
             response.termination_proven;
  }
  return false;
}

void apply_response(const PlacementActuatorResponse& response,
                    size_t max_message_bytes,
                    uint64_t now_ms,
                    PlacementOperationRecord* record) {
  record->last_code = response.code;
  record->updated_at_ms = now_ms;
  record->message = response.message.substr(0, max_message_bytes);
  if (!valid_actuator_code(response.code) || !valid_message(response.message) ||
      !valid_placement_lifecycle_state(response.lifecycle_state) ||
      !response_identity_matches(*record, response)) {
    record->status = PlacementOperationStatus::FAILED;
    record->last_code = PlacementActuatorCode::INVALID;
    return;
  }
  if (record->intent.action == PlacementOperationAction::CREATE &&
      !response.engine_uid.empty()) {
    record->engine_uid = response.engine_uid;
    record->engine_incarnation = response.engine_incarnation;
  }
  switch (response.code) {
    case PlacementActuatorCode::ACCEPTED:
      record->status = PlacementOperationStatus::SUBMITTED;
      return;
    case PlacementActuatorCode::IN_PROGRESS:
      record->status = PlacementOperationStatus::IN_PROGRESS;
      return;
    case PlacementActuatorCode::SUCCEEDED:
      if (!success_proven(*record, response)) {
        record->status = PlacementOperationStatus::FAILED;
        record->last_code = PlacementActuatorCode::INVALID;
        return;
      }
      record->status = PlacementOperationStatus::SUCCEEDED;
      record->engine_uid = response.engine_uid;
      record->engine_incarnation = response.engine_incarnation;
      return;
    case PlacementActuatorCode::FENCED:
      record->status = PlacementOperationStatus::FENCED;
      return;
    case PlacementActuatorCode::TERMINAL_ERROR:
    case PlacementActuatorCode::CONFLICT:
    case PlacementActuatorCode::INVALID:
      record->status = PlacementOperationStatus::FAILED;
      return;
    case PlacementActuatorCode::NOT_FOUND:
    case PlacementActuatorCode::RETRYABLE_ERROR:
    case PlacementActuatorCode::UNKNOWN:
      record->status = PlacementOperationStatus::UNKNOWN;
      return;
  }
}

}  // namespace

PlacementOperationExecutor::PlacementOperationExecutor(
    PlacementOperationExecutorConfig config,
    PlacementActuator* actuator,
    PlacementOperationStore* store,
    PlacementLeaderIdentity leader)
    : config_(config),
      actuator_(actuator),
      store_(store),
      leader_(std::move(leader)) {}

PlacementExecutorStatus PlacementOperationExecutor::recover(
    uint64_t now_ms,
    size_t max_total_bytes) {
  if (!valid_config(config_) || actuator_ == nullptr || store_ == nullptr ||
      !valid_placement_leader_identity(leader_) || now_ms == 0 ||
      max_total_bytes == 0 || !records_.empty()) {
    return PlacementExecutorStatus::INVALID_INPUT;
  }
  std::vector<PlacementPersistedOperation> snapshot;
  const PlacementStoreStatus status =
      store_->load_snapshot(config_.max_records, max_total_bytes, &snapshot);
  if (status == PlacementStoreStatus::CAPACITY_EXCEEDED) {
    return PlacementExecutorStatus::CAPACITY_EXCEEDED;
  }
  if (status != PlacementStoreStatus::OK) {
    return PlacementExecutorStatus::PERSISTENCE_ERROR;
  }
  for (PlacementPersistedOperation& persisted : snapshot) {
    PlacementOperationRecord& record = persisted.record;
    if (!placement_operation_terminal(record.status)) {
      record.created_at_ms = now_ms;
      record.updated_at_ms = now_ms;
      record.timed_out = false;
    }
    record.command_revision = persisted.command_revision;
    record.status_revision = persisted.status_revision;
    if (!records_.emplace(record.intent.operation_id, std::move(record))
             .second) {
      records_.clear();
      return PlacementExecutorStatus::PERSISTENCE_ERROR;
    }
  }
  return PlacementExecutorStatus::OK;
}

PlacementExecutorStatus PlacementOperationExecutor::recover_for_leader(
    PlacementLeaderIdentity leader,
    uint64_t now_ms,
    size_t max_total_bytes) {
  if (!valid_placement_leader_identity(leader)) {
    return PlacementExecutorStatus::INVALID_INPUT;
  }
  records_.clear();
  leader_ = std::move(leader);
  return recover(now_ms, max_total_bytes);
}

PlacementExecutorStatus PlacementOperationExecutor::persist(
    PlacementOperationRecord* record) {
  if (record == nullptr) {
    return PlacementExecutorStatus::INVALID_INPUT;
  }
  if (store_ == nullptr) {
    return PlacementExecutorStatus::OK;
  }
  int64_t revision = 0;
  const PlacementStoreStatus status = store_->compare_and_set_status(
      *record, record->status_revision, leader_, &revision);
  if (status == PlacementStoreStatus::CAPACITY_EXCEEDED) {
    return PlacementExecutorStatus::CAPACITY_EXCEEDED;
  }
  if (status != PlacementStoreStatus::OK) {
    return PlacementExecutorStatus::PERSISTENCE_ERROR;
  }
  record->status_revision = revision;
  return PlacementExecutorStatus::OK;
}

PlacementExecutorResult PlacementOperationExecutor::add_intents(
    const std::vector<PlacementOperationIntent>& intents,
    uint64_t now_ms) {
  PlacementExecutorResult result{
      .status = PlacementExecutorStatus::OK,
  };
  if (!valid_config(config_) || actuator_ == nullptr || now_ms == 0 ||
      intents.empty()) {
    result.status = PlacementExecutorStatus::INVALID_INPUT;
    return result;
  }
  size_t new_records = 0;
  std::map<std::string, const PlacementOperationIntent*> unique_intents;
  for (const PlacementOperationIntent& intent : intents) {
    if (!valid_placement_operation_intent(intent)) {
      result.status = PlacementExecutorStatus::INVALID_INPUT;
      return result;
    }
    const auto [unique_iterator, inserted] =
        unique_intents.emplace(intent.operation_id, &intent);
    if (!inserted &&
        !placement_operation_intents_equal(*unique_iterator->second, intent)) {
      result.status = PlacementExecutorStatus::INVALID_INPUT;
      return result;
    }
    if (!inserted) {
      continue;
    }
    const auto iterator = records_.find(intent.operation_id);
    if (iterator == records_.end()) {
      ++new_records;
      continue;
    }
    if (!placement_operation_intents_equal(iterator->second.intent, intent)) {
      result.status = PlacementExecutorStatus::INVALID_INPUT;
      return result;
    }
  }
  if (records_.size() > config_.max_records ||
      new_records > config_.max_records - records_.size()) {
    result.status = PlacementExecutorStatus::CAPACITY_EXCEEDED;
    return result;
  }
  for (const PlacementOperationIntent& intent : intents) {
    const auto iterator = records_.find(intent.operation_id);
    if (iterator != records_.end()) {
      ++result.replayed;
      continue;
    }
    PlacementOperationRecord record{
        .intent = intent,
        .created_at_ms = now_ms,
        .updated_at_ms = now_ms,
    };
    if (store_ != nullptr) {
      int64_t command_revision = 0;
      const PlacementStoreStatus command_status =
          store_->create_command(intent, leader_, &command_revision);
      if (command_status != PlacementStoreStatus::OK) {
        result.status =
            command_status == PlacementStoreStatus::CAPACITY_EXCEEDED
                ? PlacementExecutorStatus::CAPACITY_EXCEEDED
                : PlacementExecutorStatus::PERSISTENCE_ERROR;
        return result;
      }
      record.command_revision = command_revision;
      const PlacementExecutorStatus persist_status = persist(&record);
      if (persist_status != PlacementExecutorStatus::OK) {
        result.status = persist_status;
        return result;
      }
    }
    records_.emplace(intent.operation_id, std::move(record));
    ++result.added;
  }
  return result;
}

PlacementExecutorResult PlacementOperationExecutor::drive(
    uint64_t now_ms,
    uint32_t max_actions) {
  PlacementExecutorResult result{
      .status = PlacementExecutorStatus::OK,
  };
  if (!valid_config(config_) || actuator_ == nullptr || now_ms == 0 ||
      max_actions == 0) {
    result.status = PlacementExecutorStatus::INVALID_INPUT;
    return result;
  }
  for (const auto& [operation_id, record] : records_) {
    static_cast<void>(operation_id);
    if (!placement_operation_terminal(record.status) &&
        (now_ms < record.created_at_ms || now_ms < record.updated_at_ms)) {
      result.status = PlacementExecutorStatus::CLOCK_REGRESSION;
      return result;
    }
  }
  for (auto& [operation_id, record] : records_) {
    static_cast<void>(operation_id);
    if (result.driven == max_actions) {
      break;
    }
    if (placement_operation_terminal(record.status)) {
      ++result.terminal;
      continue;
    }
    if (now_ms - record.created_at_ms >= config_.operation_timeout_ms) {
      record.timed_out = true;
    }
    PlacementActuatorResponse response;
    if (record.status == PlacementOperationStatus::PLANNED) {
      ++record.execute_attempts;
      record.status = PlacementOperationStatus::UNKNOWN;
      record.last_code = PlacementActuatorCode::UNKNOWN;
      record.updated_at_ms = now_ms;
      const PlacementExecutorStatus persist_status = persist(&record);
      if (persist_status != PlacementExecutorStatus::OK) {
        result.status = persist_status;
        return result;
      }
      response = actuator_->execute(record.intent);
    } else {
      ++record.query_attempts;
      response = actuator_->query(record.intent);
    }
    apply_response(response, config_.max_message_bytes, now_ms, &record);
    const PlacementExecutorStatus persist_status = persist(&record);
    if (persist_status != PlacementExecutorStatus::OK) {
      record.status = PlacementOperationStatus::UNKNOWN;
      record.last_code = PlacementActuatorCode::UNKNOWN;
      result.status = persist_status;
      return result;
    }
    ++result.driven;
    if (placement_operation_terminal(record.status)) {
      ++result.terminal;
    } else if (record.status == PlacementOperationStatus::UNKNOWN) {
      ++result.unknown;
    }
  }
  return result;
}

std::vector<PlacementOperationRecord> PlacementOperationExecutor::snapshot()
    const {
  std::vector<PlacementOperationRecord> snapshot;
  snapshot.reserve(records_.size());
  for (const auto& [operation_id, record] : records_) {
    static_cast<void>(operation_id);
    snapshot.push_back(record);
  }
  return snapshot;
}

std::vector<PlacementOperationView>
PlacementOperationExecutor::operation_views() const {
  std::vector<PlacementOperationView> views;
  views.reserve(records_.size());
  for (const auto& [operation_id, record] : records_) {
    static_cast<void>(operation_id);
    views.emplace_back(PlacementOperationView{
        .operation_id = record.intent.operation_id,
        .action = record.intent.action,
        .status = record.status,
        .pool = record.intent.pool,
        .engine_uid = record.engine_uid.empty() ? record.intent.engine_uid
                                                : record.engine_uid,
        .engine_incarnation = record.engine_incarnation.empty()
                                  ? record.intent.engine_incarnation
                                  : record.engine_incarnation,
        .leader_incarnation = record.intent.leader_incarnation,
        .leader_epoch = record.intent.leader_epoch,
        .desired_generation = record.intent.desired_generation,
    });
  }
  return views;
}

size_t PlacementOperationExecutor::size() const { return records_.size(); }

bool valid_placement_operation_intent(const PlacementOperationIntent& intent) {
  if (!valid_placement_identity(intent.operation_id) ||
      !valid_placement_pool_key(intent.pool) ||
      !valid_placement_identity(intent.leader_incarnation) ||
      intent.leader_epoch == 0 || intent.desired_generation == 0 ||
      intent.operation_id != make_placement_operation_id(
                                 PlacementLeaderIdentity{
                                     .address = "unused",
                                     .incarnation = intent.leader_incarnation,
                                     .epoch = intent.leader_epoch,
                                 },
                                 intent.desired_generation,
                                 intent.pool,
                                 intent.action,
                                 intent.ordinal,
                                 intent.engine_uid,
                                 intent.engine_incarnation)) {
    return false;
  }
  if (intent.action == PlacementOperationAction::CREATE) {
    return intent.engine_uid.empty() && intent.engine_incarnation.empty();
  }
  return valid_placement_identity(intent.engine_uid) &&
         valid_placement_identity(intent.engine_incarnation);
}

bool placement_operation_intents_equal(const PlacementOperationIntent& left,
                                       const PlacementOperationIntent& right) {
  return left.operation_id == right.operation_id &&
         left.action == right.action &&
         placement_pool_keys_equal(left.pool, right.pool) &&
         left.engine_uid == right.engine_uid &&
         left.engine_incarnation == right.engine_incarnation &&
         left.leader_incarnation == right.leader_incarnation &&
         left.leader_epoch == right.leader_epoch &&
         left.desired_generation == right.desired_generation &&
         left.ordinal == right.ordinal;
}

bool placement_drain_proof_complete(const PlacementDrainProof& proof) {
  return proof.admission_closed && proof.prefill_queue == 0 &&
         proof.active_transfers == 0 && proof.active_reservations == 0 &&
         proof.decode_sequences == 0 && proof.pending_output == 0 &&
         proof.pending_cleanup == 0;
}

const char* placement_actuator_code_name(PlacementActuatorCode code) {
  switch (code) {
    case PlacementActuatorCode::NOT_FOUND:
      return "NOT_FOUND";
    case PlacementActuatorCode::ACCEPTED:
      return "ACCEPTED";
    case PlacementActuatorCode::IN_PROGRESS:
      return "IN_PROGRESS";
    case PlacementActuatorCode::SUCCEEDED:
      return "SUCCEEDED";
    case PlacementActuatorCode::RETRYABLE_ERROR:
      return "RETRYABLE_ERROR";
    case PlacementActuatorCode::TERMINAL_ERROR:
      return "TERMINAL_ERROR";
    case PlacementActuatorCode::UNKNOWN:
      return "UNKNOWN";
    case PlacementActuatorCode::FENCED:
      return "FENCED";
    case PlacementActuatorCode::CONFLICT:
      return "CONFLICT";
    case PlacementActuatorCode::INVALID:
      return "INVALID";
  }
  return "UNKNOWN_CODE";
}

}  // namespace xllm_service::placement
