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
         config.operation_timeout_ms > 0 && config.terminal_retention_ms > 0 &&
         config.max_terminal_compactions_per_cycle > 0;
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
  if (status == PlacementStoreStatus::CORRUPT) {
    return PlacementExecutorStatus::CORRUPT_SNAPSHOT;
  }
  if (status != PlacementStoreStatus::OK) {
    return PlacementExecutorStatus::PERSISTENCE_ERROR;
  }
  for (PlacementPersistedOperation& persisted : snapshot) {
    PlacementOperationRecord& record = persisted.record;
    if (placement_operation_terminal(record.status)) {
      // Persisted monotonic timestamps are process-local. Start a fresh,
      // conservative retention window after leadership recovery.
      record.updated_at_ms = now_ms;
      record.visibility_grace_eligible = false;
    } else {
      record.created_at_ms = now_ms;
      record.updated_at_ms = now_ms;
      record.timed_out = false;
      record.visibility_grace_eligible = true;
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
  const auto observe_pending = [&records = records_,
                                now_ms,
                                timeout_ms = config_.operation_timeout_ms,
                                &result] {
    result.pending = 0;
    result.timed_out = 0;
    result.oldest_pending_age_ms = 0;
    for (const auto& [operation_id, record] : records) {
      static_cast<void>(operation_id);
      if (placement_operation_terminal(record.status)) {
        continue;
      }
      ++result.pending;
      const bool expired = now_ms >= record.created_at_ms &&
                           now_ms - record.created_at_ms >= timeout_ms;
      result.timed_out += record.timed_out || expired ? 1 : 0;
      if (now_ms >= record.created_at_ms) {
        result.oldest_pending_age_ms = std::max(result.oldest_pending_age_ms,
                                                now_ms - record.created_at_ms);
      }
    }
  };
  result.events.reserve(
      std::min(records_.size(), static_cast<size_t>(max_actions)));
  const auto append_event = [&result,
                             now_ms](const PlacementOperationRecord& record) {
    result.events.emplace_back(PlacementOperationEvent{
        .operation_id = record.intent.operation_id,
        .action = record.intent.action,
        .status = record.status,
        .code = record.last_code,
        .engine_uid = record.engine_uid.empty() ? record.intent.engine_uid
                                                : record.engine_uid,
        .engine_incarnation = record.engine_incarnation.empty()
                                  ? record.intent.engine_incarnation
                                  : record.engine_incarnation,
        .expected_incarnation = record.intent.engine_incarnation,
        .execute_attempts = record.execute_attempts,
        .query_attempts = record.query_attempts,
        .duration_ms = now_ms - record.created_at_ms,
        .timed_out = record.timed_out,
        .message = record.message,
    });
  };
  for (const auto& [operation_id, record] : records_) {
    static_cast<void>(operation_id);
    if (!placement_operation_terminal(record.status) &&
        (now_ms < record.created_at_ms || now_ms < record.updated_at_ms)) {
      result.status = PlacementExecutorStatus::CLOCK_REGRESSION;
      observe_pending();
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
      result.timeout_transitions += record.timed_out ? 0 : 1;
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
        observe_pending();
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
      append_event(record);
      observe_pending();
      return result;
    }
    ++result.driven;
    if (placement_operation_terminal(record.status)) {
      ++result.terminal;
    } else if (record.status == PlacementOperationStatus::UNKNOWN) {
      ++result.unknown;
    }
    result.conflict +=
        record.last_code == PlacementActuatorCode::CONFLICT ? 1 : 0;
    result.fenced += record.last_code == PlacementActuatorCode::FENCED ? 1 : 0;
    append_event(record);
  }
  uint32_t canceled = 0;
  if (result.status == PlacementExecutorStatus::OK) {
    result.status = cancel_superseded_drains(now_ms, &canceled);
    result.terminal += canceled;
  }
  if (result.status == PlacementExecutorStatus::OK) {
    result.status = compact_terminal(now_ms, &result.compacted);
  }
  observe_pending();
  return result;
}

PlacementExecutorStatus PlacementOperationExecutor::cancel_superseded_drains(
    uint64_t now_ms,
    uint32_t* canceled) {
  if (canceled == nullptr) {
    return PlacementExecutorStatus::INVALID_INPUT;
  }
  *canceled = 0;
  std::map<std::pair<std::string, std::string>, uint64_t>
      canceled_drain_generations;
  for (const auto& [operation_id, record] : records_) {
    static_cast<void>(operation_id);
    if (record.status == PlacementOperationStatus::SUCCEEDED &&
        record.intent.action == PlacementOperationAction::CANCEL_DRAIN) {
      const std::pair<std::string, std::string> identity{
          record.intent.engine_uid, record.intent.engine_incarnation};
      canceled_drain_generations[identity] =
          std::max(canceled_drain_generations[identity],
                   record.intent.desired_generation);
    }
  }
  for (auto& [operation_id, record] : records_) {
    static_cast<void>(operation_id);
    if (record.intent.action != PlacementOperationAction::BEGIN_DRAIN ||
        record.status == PlacementOperationStatus::SUCCEEDED ||
        record.status == PlacementOperationStatus::FENCED ||
        record.status == PlacementOperationStatus::CANCELED) {
      continue;
    }
    const auto cancellation = canceled_drain_generations.find(
        {record.intent.engine_uid, record.intent.engine_incarnation});
    if (cancellation == canceled_drain_generations.end() ||
        cancellation->second < record.intent.desired_generation) {
      continue;
    }
    const bool was_terminal = placement_operation_terminal(record.status);
    const PlacementOperationRecord previous = record;
    record.status = PlacementOperationStatus::CANCELED;
    record.last_code = PlacementActuatorCode::SUCCEEDED;
    record.updated_at_ms = now_ms;
    record.timed_out = false;
    record.message = "superseded by successful cancel drain";
    const PlacementExecutorStatus status = persist(&record);
    if (status != PlacementExecutorStatus::OK) {
      record = previous;
      return status;
    }
    if (!was_terminal) {
      ++*canceled;
    }
  }
  return PlacementExecutorStatus::OK;
}

PlacementExecutorStatus PlacementOperationExecutor::compact_terminal(
    uint64_t now_ms,
    uint32_t* compacted) {
  if (compacted == nullptr) {
    return PlacementExecutorStatus::INVALID_INPUT;
  }
  *compacted = 0;
  std::map<std::pair<std::string, std::string>, uint64_t>
      canceled_drain_generations;
  std::map<std::pair<std::string, std::string>, uint64_t>
      terminated_drain_generations;
  for (const auto& [operation_id, record] : records_) {
    static_cast<void>(operation_id);
    if (record.status == PlacementOperationStatus::SUCCEEDED &&
        record.intent.action == PlacementOperationAction::CANCEL_DRAIN) {
      const std::pair<std::string, std::string> identity{
          record.intent.engine_uid, record.intent.engine_incarnation};
      canceled_drain_generations[identity] =
          std::max(canceled_drain_generations[identity],
                   record.intent.desired_generation);
    } else if (record.status == PlacementOperationStatus::SUCCEEDED &&
               record.intent.action == PlacementOperationAction::TERMINATE) {
      const std::pair<std::string, std::string> identity{
          record.intent.engine_uid, record.intent.engine_incarnation};
      terminated_drain_generations[identity] =
          std::max(terminated_drain_generations[identity],
                   record.intent.desired_generation);
    }
  }

  auto retained = [&](const PlacementOperationRecord& record) {
    const bool compactable =
        record.status == PlacementOperationStatus::SUCCEEDED ||
        (record.status == PlacementOperationStatus::CANCELED &&
         record.intent.action == PlacementOperationAction::BEGIN_DRAIN);
    return !compactable || now_ms < record.updated_at_ms ||
           now_ms - record.updated_at_ms < config_.terminal_retention_ms;
  };
  auto erase_record = [&](const std::string& operation_id) {
    const auto iterator = records_.find(operation_id);
    if (iterator == records_.end()) {
      return PlacementExecutorStatus::OK;
    }
    if (store_ != nullptr) {
      const PlacementStoreStatus status =
          store_->delete_terminal(iterator->second, leader_);
      if (status == PlacementStoreStatus::CAPACITY_EXCEEDED) {
        return PlacementExecutorStatus::CAPACITY_EXCEEDED;
      }
      if (status != PlacementStoreStatus::OK) {
        return PlacementExecutorStatus::PERSISTENCE_ERROR;
      }
    }
    records_.erase(iterator);
    ++*compacted;
    return PlacementExecutorStatus::OK;
  };

  // Drain completion records are the durable proof that makes BEGIN_DRAIN
  // deletion safe. Compact dependency records first and completion proofs
  // last, otherwise a small per-cycle budget can strand BEGIN_DRAIN forever.
  std::vector<std::string> completed_drains;
  std::vector<std::string> independent;
  std::vector<std::string> completion_proofs;
  for (const auto& [operation_id, record] : records_) {
    if (retained(record)) {
      continue;
    }
    if (record.intent.action == PlacementOperationAction::BEGIN_DRAIN) {
      const auto& completion_generations =
          record.status == PlacementOperationStatus::CANCELED
              ? canceled_drain_generations
              : terminated_drain_generations;
      const auto completed = completion_generations.find(
          {record.intent.engine_uid, record.intent.engine_incarnation});
      if (completed != completion_generations.end() &&
          completed->second >= record.intent.desired_generation) {
        completed_drains.push_back(operation_id);
      }
    } else if (record.intent.action == PlacementOperationAction::CANCEL_DRAIN ||
               record.intent.action == PlacementOperationAction::TERMINATE) {
      completion_proofs.push_back(operation_id);
    } else {
      independent.push_back(operation_id);
    }
  }

  const auto compact_candidates = [&](const std::vector<std::string>& ids) {
    for (const std::string& operation_id : ids) {
      if (*compacted == config_.max_terminal_compactions_per_cycle) {
        break;
      }
      const PlacementExecutorStatus status = erase_record(operation_id);
      if (status != PlacementExecutorStatus::OK) {
        return status;
      }
    }
    return PlacementExecutorStatus::OK;
  };
  PlacementExecutorStatus status = compact_candidates(completed_drains);
  if (status != PlacementExecutorStatus::OK) {
    return status;
  }
  status = compact_candidates(independent);
  if (status != PlacementExecutorStatus::OK) {
    return status;
  }
  for (const std::string& operation_id : completion_proofs) {
    if (*compacted == config_.max_terminal_compactions_per_cycle) {
      break;
    }
    const auto completion = records_.find(operation_id);
    if (completion == records_.end()) {
      continue;
    }
    const PlacementOperationRecord& completion_record = completion->second;
    const PlacementOperationStatus dependency_status =
        completion_record.intent.action ==
                PlacementOperationAction::CANCEL_DRAIN
            ? PlacementOperationStatus::CANCELED
            : PlacementOperationStatus::SUCCEEDED;
    bool has_dependent_drain = false;
    for (const auto& [candidate_id, candidate] : records_) {
      static_cast<void>(candidate_id);
      if (candidate.status == dependency_status &&
          candidate.intent.action == PlacementOperationAction::BEGIN_DRAIN &&
          candidate.intent.engine_uid == completion_record.intent.engine_uid &&
          candidate.intent.engine_incarnation ==
              completion_record.intent.engine_incarnation &&
          candidate.intent.desired_generation <=
              completion_record.intent.desired_generation) {
        has_dependent_drain = true;
        break;
      }
    }
    if (!has_dependent_drain) {
      status = erase_record(operation_id);
      if (status != PlacementExecutorStatus::OK) {
        return status;
      }
    }
  }
  return PlacementExecutorStatus::OK;
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
        .updated_at_ms = record.updated_at_ms,
        .visibility_grace_eligible = record.visibility_grace_eligible,
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
