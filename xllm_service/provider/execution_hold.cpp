/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "provider/execution_hold.h"

#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace xllm_service::provider {

struct ExecutionHoldAdoptionState {
  std::mutex mutex;
  ExecutionHoldCleanupTable* table = nullptr;
};

namespace {

using xllm::proto::ExecutionAttemptId;
using xllm::proto::ExecutionHolder;
using xllm::proto::ExecutionHoldProof;
using xllm::proto::ExecutionResourceHold;
using xllm::proto::HolderConvergenceProof;

bool is_uuid_v7(const std::string& value) {
  if (value.size() != 36 || value[8] != '-' || value[13] != '-' ||
      value[18] != '-' || value[23] != '-' || value[14] != '7') {
    return false;
  }
  const char variant =
      static_cast<char>(std::tolower(static_cast<unsigned char>(value[19])));
  if (variant != '8' && variant != '9' && variant != 'a' && variant != 'b') {
    return false;
  }
  for (size_t index = 0; index < value.size(); ++index) {
    if (index == 8 || index == 13 || index == 18 || index == 23) {
      continue;
    }
    if (!std::isxdigit(static_cast<unsigned char>(value[index]))) {
      return false;
    }
  }
  return true;
}

bool valid_identifier(const std::string& value, size_t max_bytes) {
  if (value.empty() || value.size() > max_bytes) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](unsigned char character) {
    return character >= 0x21 && character <= 0x7e;
  });
}

bool valid_attempt(const ExecutionAttemptId& attempt) {
  return is_uuid_v7(attempt.request_uid()) && attempt.has_attempt_seq();
}

std::string holder_key(const ExecutionHolder& holder) {
  std::string result;
  result.reserve(holder.engine_uid().size() + holder.incarnation_id().size() +
                 1);
  result.append(holder.engine_uid());
  result.push_back('\0');
  result.append(holder.incarnation_id());
  return result;
}

bool same_holder(const ExecutionHolder& lhs, const ExecutionHolder& rhs) {
  return lhs.engine_uid() == rhs.engine_uid() &&
         lhs.incarnation_id() == rhs.incarnation_id();
}

bool same_attempt(const ExecutionAttemptId& lhs,
                  const ExecutionAttemptId& rhs) {
  return lhs.request_uid() == rhs.request_uid() && lhs.has_attempt_seq() &&
         rhs.has_attempt_seq() && lhs.attempt_seq() == rhs.attempt_seq();
}

bool contains_holder(const ExecutionResourceHold& hold,
                     const ExecutionHolder& holder) {
  return std::any_of(hold.potential_holders().begin(),
                     hold.potential_holders().end(),
                     [&holder](const ExecutionHolder& candidate) {
                       return same_holder(candidate, holder);
                     });
}

std::optional<size_t> find_holder(const ExecutionResourceHold& hold,
                                  const ExecutionHolder& holder) {
  for (int index = 0; index < hold.potential_holders_size(); ++index) {
    if (same_holder(hold.potential_holders(index), holder)) {
      return static_cast<size_t>(index);
    }
  }
  return std::nullopt;
}

bool valid_convergence_proof(HolderConvergenceProof proof,
                             bool allow_hard_time_bound_proof) {
  switch (proof) {
    case xllm::proto::HOLDER_CONVERGENCE_PROOF_TERMINAL_OUTCOME:
    case xllm::proto::HOLDER_CONVERGENCE_PROOF_CANCEL_FENCE_ACK:
    case xllm::proto::HOLDER_CONVERGENCE_PROOF_SELF_FENCED:
    case xllm::proto::HOLDER_CONVERGENCE_PROOF_PROCESS_TERMINATED:
      return true;
    case xllm::proto::HOLDER_CONVERGENCE_PROOF_HARD_TIME_BOUND:
      return allow_hard_time_bound_proof;
    case xllm::proto::HOLDER_CONVERGENCE_PROOF_UNSPECIFIED:
    case xllm::proto::HOLDER_CONVERGENCE_PROOF_QUERY_ABSENT:
    default:
      return false;
  }
}

bool all_holders_converged(const ExecutionResourceHold& hold,
                           const std::vector<bool>& converged_holders) {
  return converged_holders.size() ==
             static_cast<size_t>(hold.potential_holders_size()) &&
         std::all_of(converged_holders.begin(),
                     converged_holders.end(),
                     [](bool converged) { return converged; });
}

size_t cleanup_record_size(const ExecutionResourceHold& hold) {
  const size_t convergence_mask_bytes =
      (static_cast<size_t>(hold.potential_holders_size()) + 7) / 8;
  return hold.ByteSizeLong() + convergence_mask_bytes;
}

ExecutionHoldStatus validate_hold(
    const ExecutionResourceHold& hold,
    const ExecutionHoldCleanupTable::Config& config) {
  if (!valid_attempt(hold.attempt()) ||
      hold.kind() == xllm::proto::EXECUTION_HOLD_KIND_UNSPECIFIED ||
      !xllm::proto::ExecutionHoldKind_IsValid(hold.kind()) ||
      hold.proof() != xllm::proto::EXECUTION_HOLD_PROOF_OUTCOME_UNKNOWN ||
      hold.has_confirmed_holder() || hold.has_likely_holder() ||
      hold.potential_holders().empty() ||
      static_cast<size_t>(hold.potential_holders_size()) >
          config.max_potential_holders ||
      (!hold.coordinator_incarnation_id().empty() &&
       !valid_identifier(hold.coordinator_incarnation_id(),
                         config.max_identifier_bytes))) {
    return ExecutionHoldStatus::kInvalidArgument;
  }

  std::unordered_set<std::string> seen;
  for (const ExecutionHolder& holder : hold.potential_holders()) {
    if (!valid_identifier(holder.engine_uid(), config.max_identifier_bytes) ||
        !valid_identifier(holder.incarnation_id(),
                          config.max_identifier_bytes) ||
        !seen.insert(holder_key(holder)).second) {
      return ExecutionHoldStatus::kInvalidArgument;
    }
  }
  if (cleanup_record_size(hold) > config.max_cleanup_record_bytes) {
    return ExecutionHoldStatus::kRecordTooLarge;
  }
  return ExecutionHoldStatus::kOk;
}

}  // namespace

struct ExecutionHoldCleanupTable::Reservation::CapacityState {
  explicit CapacityState(Config value) : config(std::move(value)) {}

  Config config;
  std::mutex mutex;
  size_t reserved_records = 0;
  size_t reserved_bytes = 0;
};

ExecutionHoldCleanupTable::Reservation::Reservation(
    std::shared_ptr<CapacityState> state,
    std::weak_ptr<ExecutionHoldAdoptionState> adoption_state)
    : state_(std::move(state)), adoption_state_(std::move(adoption_state)) {}

ExecutionHoldCleanupTable::Reservation::~Reservation() { reset(); }

ExecutionHoldCleanupTable::Reservation::Reservation(
    Reservation&& other) noexcept
    : state_(std::move(other.state_)),
      adoption_state_(std::move(other.adoption_state_)) {}

ExecutionHoldCleanupTable::Reservation&
ExecutionHoldCleanupTable::Reservation::operator=(
    Reservation&& other) noexcept {
  if (this != &other) {
    reset();
    state_ = std::move(other.state_);
    adoption_state_ = std::move(other.adoption_state_);
  }
  return *this;
}

void ExecutionHoldCleanupTable::Reservation::reset() {
  if (state_ == nullptr) {
    return;
  }
  std::shared_ptr<CapacityState> state = std::move(state_);
  std::lock_guard<std::mutex> lock(state->mutex);
  --state->reserved_records;
  state->reserved_bytes -= state->config.max_cleanup_record_bytes;
}

ExecutionHoldCleanupTable::ExecutionHoldCleanupTable(Config config) {
  if (config.record_capacity == 0 || config.byte_capacity == 0 ||
      config.max_cleanup_record_bytes == 0 ||
      config.max_cleanup_record_bytes > config.byte_capacity ||
      config.max_potential_holders == 0 ||
      config.max_potential_holders >
          static_cast<size_t>(std::numeric_limits<int>::max()) ||
      config.max_identifier_bytes == 0) {
    throw std::invalid_argument("invalid execution hold cleanup capacity");
  }
  capacity_ = std::make_shared<Reservation::CapacityState>(std::move(config));
  adoption_state_ = std::make_shared<ExecutionHoldAdoptionState>();
  adoption_state_->table = this;
}

ExecutionHoldCleanupTable::~ExecutionHoldCleanupTable() {
  std::lock_guard<std::mutex> lock(adoption_state_->mutex);
  adoption_state_->table = nullptr;
}

std::optional<ExecutionHoldCleanupTable::Reservation>
ExecutionHoldCleanupTable::try_reserve() {
  std::lock_guard<std::mutex> lock(capacity_->mutex);
  const Config& config = capacity_->config;
  if (capacity_->reserved_records >= config.record_capacity ||
      capacity_->reserved_bytes >
          config.byte_capacity - config.max_cleanup_record_bytes) {
    return std::nullopt;
  }
  ++capacity_->reserved_records;
  capacity_->reserved_bytes += config.max_cleanup_record_bytes;
  return Reservation(capacity_, adoption_state_);
}

ExecutionHoldStatus ExecutionHoldCleanupTable::install_request_hold(
    RequestExecutionHold* request_hold,
    xllm::proto::ExecutionHoldKind kind,
    const ExecutionAttemptId& attempt,
    const std::string& coordinator_incarnation_id,
    const std::vector<ExecutionHolder>& potential_holders) {
  if (request_hold == nullptr ||
      potential_holders.size() > capacity_->config.max_potential_holders) {
    return ExecutionHoldStatus::kInvalidArgument;
  }
  std::optional<Reservation> reservation = try_reserve();
  if (!reservation.has_value()) {
    return ExecutionHoldStatus::kCleanupCapacityRetryable;
  }

  ExecutionResourceHold hold;
  hold.set_kind(kind);
  *hold.mutable_attempt() = attempt;
  hold.set_coordinator_incarnation_id(coordinator_incarnation_id);
  hold.mutable_potential_holders()->Reserve(
      static_cast<int>(potential_holders.size()));
  for (const ExecutionHolder& holder : potential_holders) {
    *hold.add_potential_holders() = holder;
  }
  hold.set_proof(xllm::proto::EXECUTION_HOLD_PROOF_OUTCOME_UNKNOWN);
  return request_hold->install(std::move(*reservation), std::move(hold));
}

size_t ExecutionHoldCleanupTable::AttemptKeyHash::operator()(
    const AttemptKey& key) const {
  size_t result = std::hash<std::string>{}(key.request_uid);
  result ^= std::hash<uint64_t>{}(key.attempt_seq) + 0x9e3779b9 +
            (result << 6) + (result >> 2);
  return result;
}

ExecutionHoldCleanupTable::AttemptKey ExecutionHoldCleanupTable::to_key(
    const ExecutionAttemptId& attempt) {
  return AttemptKey{attempt.request_uid(), attempt.attempt_seq()};
}

ExecutionHoldStatus ExecutionHoldCleanupTable::adopt(
    RequestExecutionHold* request_hold) {
  if (request_hold == nullptr) {
    return ExecutionHoldStatus::kInvalidArgument;
  }
  std::lock_guard<std::mutex> request_lock(request_hold->mutex_);
  if (request_hold->detached_) {
    return ExecutionHoldStatus::kAlreadyDetached;
  }
  if (!request_hold->hold_.has_value() || !request_hold->reservation_) {
    return ExecutionHoldStatus::kNoHold;
  }
  if (request_hold->reservation_.state_.get() != capacity_.get()) {
    return ExecutionHoldStatus::kInvalidArgument;
  }

  const AttemptKey key = to_key(request_hold->hold_->attempt());
  std::lock_guard<std::mutex> table_lock(mutex_);
  if (records_.find(key) != records_.end()) {
    return ExecutionHoldStatus::kAlreadyDetached;
  }

  CleanupRecord record;
  record.hold = std::move(*request_hold->hold_);
  record.converged_holders = std::move(request_hold->converged_holders_);
  record.reservation = std::move(request_hold->reservation_);
  retry_order_.push_back(key);
  record.retry_order_it = std::prev(retry_order_.end());
  records_.emplace(key, std::move(record));
  request_hold->hold_.reset();
  request_hold->detached_ = true;
  return ExecutionHoldStatus::kOk;
}

ExecutionHoldStatus ExecutionHoldCleanupTable::apply_convergence_proof(
    const ExecutionAttemptId& attempt,
    const ExecutionHolder& holder,
    HolderConvergenceProof proof) {
  if (!valid_attempt(attempt) ||
      !valid_convergence_proof(proof,
                               capacity_->config.allow_hard_time_bound_proof)) {
    return ExecutionHoldStatus::kUnsafeProof;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = records_.find(to_key(attempt));
  if (it == records_.end()) {
    return ExecutionHoldStatus::kNoHold;
  }
  if (!same_attempt(attempt, it->second.hold.attempt())) {
    return ExecutionHoldStatus::kAttemptMismatch;
  }
  const std::optional<size_t> holder_index =
      find_holder(it->second.hold, holder);
  if (!holder_index.has_value()) {
    return ExecutionHoldStatus::kHolderMismatch;
  }
  it->second.converged_holders[*holder_index] = true;
  if (!all_holders_converged(it->second.hold, it->second.converged_holders)) {
    return ExecutionHoldStatus::kOk;
  }
  it->second.hold.set_proof(xllm::proto::EXECUTION_HOLD_PROOF_TERMINAL);
  retry_order_.erase(it->second.retry_order_it);
  records_.erase(it);
  return ExecutionHoldStatus::kResolved;
}

size_t ExecutionHoldCleanupTable::mark_holder_process_terminated(
    const ExecutionHolder& holder) {
  const Config& config = capacity_->config;
  if (!valid_identifier(holder.engine_uid(), config.max_identifier_bytes) ||
      !valid_identifier(holder.incarnation_id(), config.max_identifier_bytes)) {
    return 0;
  }

  size_t resolved = 0;
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = records_.begin(); it != records_.end();) {
    const std::optional<size_t> holder_index =
        find_holder(it->second.hold, holder);
    if (!holder_index.has_value()) {
      ++it;
      continue;
    }
    it->second.converged_holders[*holder_index] = true;
    if (!all_holders_converged(it->second.hold, it->second.converged_holders)) {
      ++it;
      continue;
    }
    it->second.hold.set_proof(xllm::proto::EXECUTION_HOLD_PROOF_TERMINAL);
    retry_order_.erase(it->second.retry_order_it);
    it = records_.erase(it);
    ++resolved;
  }
  return resolved;
}

std::vector<ExecutionResourceHold> ExecutionHoldCleanupTable::next_retry_batch(
    size_t max_records) {
  std::vector<ExecutionResourceHold> result;
  if (max_records == 0) {
    return result;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  const size_t records_to_visit = retry_order_.size();
  result.reserve(std::min(max_records, records_to_visit));
  size_t records_visited = 0;
  while (result.size() < max_records && records_visited < records_to_visit &&
         !retry_order_.empty()) {
    ++records_visited;
    const auto order_it = retry_order_.begin();
    const auto record_it = records_.find(*order_it);
    if (record_it == records_.end()) {
      retry_order_.erase(order_it);
      continue;
    }
    ExecutionResourceHold pending = record_it->second.hold;
    pending.clear_potential_holders();
    for (int holder_index = 0;
         holder_index < record_it->second.hold.potential_holders_size();
         ++holder_index) {
      if (!record_it->second.converged_holders[holder_index]) {
        *pending.add_potential_holders() =
            record_it->second.hold.potential_holders(holder_index);
      }
    }
    result.emplace_back(std::move(pending));
    retry_order_.splice(retry_order_.end(), retry_order_, order_it);
  }
  return result;
}

bool ExecutionHoldCleanupTable::contains(
    const ExecutionAttemptId& attempt) const {
  if (!valid_attempt(attempt)) {
    return false;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  return records_.find(to_key(attempt)) != records_.end();
}

std::optional<ExecutionResourceHold> ExecutionHoldCleanupTable::query(
    const ExecutionAttemptId& attempt) const {
  if (!valid_attempt(attempt)) {
    return std::nullopt;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = records_.find(to_key(attempt));
  if (it == records_.end()) {
    return std::nullopt;
  }
  return it->second.hold;
}

ExecutionHoldCleanupTable::Stats ExecutionHoldCleanupTable::stats() const {
  Stats result;
  {
    std::lock_guard<std::mutex> lock(capacity_->mutex);
    result.reserved_records = capacity_->reserved_records;
    result.reserved_bytes = capacity_->reserved_bytes;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    result.cleanup_records = records_.size();
  }
  return result;
}

ExecutionHoldStatus RequestExecutionHold::install(
    ExecutionHoldCleanupTable::Reservation reservation,
    ExecutionResourceHold hold) {
  if (!reservation) {
    return ExecutionHoldStatus::kCleanupCapacityRetryable;
  }
  const ExecutionHoldStatus validation =
      validate_hold(hold, reservation.state_->config);
  if (validation != ExecutionHoldStatus::kOk) {
    return validation;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (hold_.has_value()) {
    return ExecutionHoldStatus::kAlreadyInstalled;
  }
  if (detached_) {
    return ExecutionHoldStatus::kAlreadyDetached;
  }
  adoption_state_ = reservation.adoption_state_;
  reservation_ = std::move(reservation);
  hold_ = std::move(hold);
  converged_holders_.assign(
      static_cast<size_t>(hold_->potential_holders_size()), false);
  return ExecutionHoldStatus::kOk;
}

RequestExecutionHold::~RequestExecutionHold() {
  if (!has_hold()) {
    return;
  }
  const std::shared_ptr<ExecutionHoldAdoptionState> adoption_state =
      adoption_state_.lock();
  if (adoption_state == nullptr) {
    LOG(ERROR) << "Destroying an unresolved execution hold after its cleanup "
                  "table was destroyed";
    return;
  }
  std::lock_guard<std::mutex> lock(adoption_state->mutex);
  if (adoption_state->table == nullptr) {
    LOG(ERROR) << "Destroying an unresolved execution hold after its cleanup "
                  "table was destroyed";
    return;
  }
  const ExecutionHoldStatus status = adoption_state->table->adopt(this);
  if (status != ExecutionHoldStatus::kOk &&
      status != ExecutionHoldStatus::kNoHold &&
      status != ExecutionHoldStatus::kAlreadyDetached) {
    LOG(ERROR) << "Failed to preserve an unresolved execution hold during "
                  "destruction, status="
               << static_cast<int>(status);
  }
}

ExecutionHoldStatus RequestExecutionHold::set_likely_holder(
    const ExecutionHolder& holder) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!hold_.has_value()) {
    return ExecutionHoldStatus::kNoHold;
  }
  if (!contains_holder(*hold_, holder)) {
    return ExecutionHoldStatus::kHolderMismatch;
  }
  *hold_->mutable_likely_holder() = holder;
  if (cleanup_record_size(*hold_) >
      reservation_.state_->config.max_cleanup_record_bytes) {
    hold_->clear_likely_holder();
    return ExecutionHoldStatus::kRecordTooLarge;
  }
  return ExecutionHoldStatus::kOk;
}

ExecutionHoldStatus RequestExecutionHold::confirm_holder(
    const ExecutionHolder& holder,
    ExecutionHoldProof proof) {
  if (proof != xllm::proto::EXECUTION_HOLD_PROOF_PROVEN_PRECOMMIT &&
      proof != xllm::proto::EXECUTION_HOLD_PROOF_GENERATION_COMMITTED) {
    return ExecutionHoldStatus::kUnsafeProof;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (!hold_.has_value()) {
    return ExecutionHoldStatus::kNoHold;
  }
  if (!contains_holder(*hold_, holder)) {
    return ExecutionHoldStatus::kHolderMismatch;
  }
  if (hold_->proof() > proof) {
    return ExecutionHoldStatus::kUnsafeProof;
  }

  const std::optional<size_t> holder_index = find_holder(*hold_, holder);
  CHECK(holder_index.has_value());
  const bool holder_already_converged = converged_holders_[*holder_index];

  hold_->clear_potential_holders();
  *hold_->add_potential_holders() = holder;
  *hold_->mutable_confirmed_holder() = holder;
  hold_->clear_likely_holder();
  hold_->set_proof(proof);
  converged_holders_.assign(1, holder_already_converged);
  if (holder_already_converged) {
    hold_->set_proof(xllm::proto::EXECUTION_HOLD_PROOF_TERMINAL);
    hold_.reset();
    converged_holders_.clear();
    reservation_ = ExecutionHoldCleanupTable::Reservation();
    return ExecutionHoldStatus::kResolved;
  }
  return ExecutionHoldStatus::kOk;
}

ExecutionHoldStatus RequestExecutionHold::advance_proof(
    ExecutionHoldProof proof) {
  if (proof != xllm::proto::EXECUTION_HOLD_PROOF_PROVEN_PRECOMMIT &&
      proof != xllm::proto::EXECUTION_HOLD_PROOF_GENERATION_COMMITTED) {
    return ExecutionHoldStatus::kUnsafeProof;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (!hold_.has_value()) {
    return ExecutionHoldStatus::kNoHold;
  }
  if (!hold_->has_confirmed_holder() || hold_->proof() > proof) {
    return ExecutionHoldStatus::kUnsafeProof;
  }
  hold_->set_proof(proof);
  return ExecutionHoldStatus::kOk;
}

ExecutionHoldStatus RequestExecutionHold::apply_convergence_proof(
    const ExecutionAttemptId& attempt,
    const ExecutionHolder& holder,
    HolderConvergenceProof proof) {
  if (!valid_attempt(attempt)) {
    return ExecutionHoldStatus::kUnsafeProof;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (!hold_.has_value() || !reservation_) {
    return ExecutionHoldStatus::kNoHold;
  }
  if (!valid_convergence_proof(
          proof, reservation_.state_->config.allow_hard_time_bound_proof)) {
    return ExecutionHoldStatus::kUnsafeProof;
  }
  if (!same_attempt(attempt, hold_->attempt())) {
    return ExecutionHoldStatus::kAttemptMismatch;
  }
  const std::optional<size_t> holder_index = find_holder(*hold_, holder);
  if (!holder_index.has_value()) {
    return ExecutionHoldStatus::kHolderMismatch;
  }
  converged_holders_[*holder_index] = true;
  if (!all_holders_converged(*hold_, converged_holders_)) {
    return ExecutionHoldStatus::kOk;
  }
  hold_->set_proof(xllm::proto::EXECUTION_HOLD_PROOF_TERMINAL);
  hold_.reset();
  converged_holders_.clear();
  reservation_ = ExecutionHoldCleanupTable::Reservation();
  return ExecutionHoldStatus::kResolved;
}

ExecutionHoldStatus RequestExecutionHold::abandon_before_dispatch() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (detached_) {
    return ExecutionHoldStatus::kAlreadyDetached;
  }
  if (!hold_.has_value() || !reservation_) {
    return ExecutionHoldStatus::kNoHold;
  }
  if (hold_->proof() != xllm::proto::EXECUTION_HOLD_PROOF_OUTCOME_UNKNOWN) {
    return ExecutionHoldStatus::kUnsafeProof;
  }
  hold_.reset();
  converged_holders_.clear();
  reservation_ = ExecutionHoldCleanupTable::Reservation();
  return ExecutionHoldStatus::kResolved;
}

bool RequestExecutionHold::has_hold() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return hold_.has_value();
}

std::optional<ExecutionResourceHold> RequestExecutionHold::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return hold_;
}

}  // namespace xllm_service::provider
