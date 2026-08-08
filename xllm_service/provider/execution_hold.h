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

#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "provider.pb.h"

namespace xllm_service::provider {

// SERVICE_CLEANUP_CAPACITY_RETRYABLE is the stable external mapping for
// kCleanupCapacityRetryable. All other values describe programmer/protocol
// errors and fail closed.
enum class ExecutionHoldStatus {
  kOk = 0,
  kResolved,
  kCleanupCapacityRetryable,
  kInvalidArgument,
  kAlreadyInstalled,
  kNoHold,
  kAttemptMismatch,
  kHolderMismatch,
  kUnsafeProof,
  kAlreadyDetached,
  kRecordTooLarge,
};

class RequestExecutionHold;

// Bounded Service-local table for resource convergence that may outlive the
// client request. Capacity is reserved before dispatch and the same move-only
// token follows the hold from RequestContext into this table.
class ExecutionHoldCleanupTable final {
 public:
  struct Config {
    size_t record_capacity = 0;
    size_t byte_capacity = 0;
    size_t max_cleanup_record_bytes = 0;
    size_t max_potential_holders = 0;
    size_t max_identifier_bytes = 0;
    // Enabled only for a workload profile whose complete local monotonic hard
    // convergence bound is configured and enforceable.
    bool allow_hard_time_bound_proof = false;
  };

  class Reservation final {
   public:
    Reservation() = default;
    ~Reservation();

    Reservation(Reservation&& other) noexcept;
    Reservation& operator=(Reservation&& other) noexcept;

    Reservation(const Reservation&) = delete;
    Reservation& operator=(const Reservation&) = delete;

    explicit operator bool() const { return state_ != nullptr; }

   private:
    struct CapacityState;

    explicit Reservation(std::shared_ptr<CapacityState> state);
    void reset();

    std::shared_ptr<CapacityState> state_;

    friend class ExecutionHoldCleanupTable;
    friend class RequestExecutionHold;
  };

  struct Stats {
    size_t reserved_records = 0;
    size_t reserved_bytes = 0;
    size_t cleanup_records = 0;
  };

  explicit ExecutionHoldCleanupTable(Config config);

  ExecutionHoldCleanupTable(const ExecutionHoldCleanupTable&) = delete;
  ExecutionHoldCleanupTable& operator=(const ExecutionHoldCleanupTable&) =
      delete;

  // Returns an empty optional when dispatch must be rejected with
  // SERVICE_CLEANUP_CAPACITY_RETRYABLE.
  std::optional<Reservation> try_reserve();

  // Atomically reserves cleanup capacity and installs the request's initial
  // outcome-unknown hold. Callers must complete this before dispatch.
  ExecutionHoldStatus install_request_hold(
      RequestExecutionHold* request_hold,
      xllm::proto::ExecutionHoldKind kind,
      const xllm::proto::ExecutionAttemptId& attempt,
      const std::string& coordinator_incarnation_id,
      const std::vector<xllm::proto::ExecutionHolder>& potential_holders);

  // Detaches the request's minimal hold record. Prompt, output, parser, and
  // retry state cannot enter this API.
  ExecutionHoldStatus adopt(RequestExecutionHold* request_hold);

  // Only a terminal/fence/fencing/hard-bound proof can converge a holder.
  // QUERY_ABSENT is deliberately represented by the wire enum and rejected.
  ExecutionHoldStatus apply_convergence_proof(
      const xllm::proto::ExecutionAttemptId& attempt,
      const xllm::proto::ExecutionHolder& holder,
      xllm::proto::HolderConvergenceProof proof);

  // Applies exact process-lifetime evidence to all detached cleanup records.
  // Returns the number of records that became fully resolved.
  size_t mark_holder_process_terminated(
      const xllm::proto::ExecutionHolder& holder);

  bool contains(const xllm::proto::ExecutionAttemptId& attempt) const;
  std::optional<xllm::proto::ExecutionResourceHold> query(
      const xllm::proto::ExecutionAttemptId& attempt) const;
  Stats stats() const;

 private:
  struct AttemptKey {
    std::string request_uid;
    uint64_t attempt_seq = 0;

    bool operator==(const AttemptKey& other) const {
      return request_uid == other.request_uid &&
             attempt_seq == other.attempt_seq;
    }
  };

  struct AttemptKeyHash {
    size_t operator()(const AttemptKey& key) const;
  };

  struct CleanupRecord {
    xllm::proto::ExecutionResourceHold hold;
    std::vector<bool> converged_holders;
    Reservation reservation;
  };

  static AttemptKey to_key(const xllm::proto::ExecutionAttemptId& attempt);

  std::shared_ptr<Reservation::CapacityState> capacity_;
  mutable std::mutex mutex_;
  std::unordered_map<AttemptKey, CleanupRecord, AttemptKeyHash> records_;
};

// RequestContext-owned, concurrency-safe single-hold gate. Installation
// consumes a pre-dispatch Reservation. Resolution releases it immediately;
// early request termination transfers it to ExecutionHoldCleanupTable.
class RequestExecutionHold final {
 public:
  RequestExecutionHold() = default;

  RequestExecutionHold(const RequestExecutionHold&) = delete;
  RequestExecutionHold& operator=(const RequestExecutionHold&) = delete;

  ExecutionHoldStatus install(
      ExecutionHoldCleanupTable::Reservation reservation,
      xllm::proto::ExecutionResourceHold hold);

  ExecutionHoldStatus set_likely_holder(
      const xllm::proto::ExecutionHolder& holder);

  // A holder may narrow the candidate set only with explicit precommit or
  // generation-commit evidence from that exact candidate incarnation.
  ExecutionHoldStatus confirm_holder(const xllm::proto::ExecutionHolder& holder,
                                     xllm::proto::ExecutionHoldProof proof);

  ExecutionHoldStatus advance_proof(xllm::proto::ExecutionHoldProof proof);

  ExecutionHoldStatus apply_convergence_proof(
      const xllm::proto::ExecutionAttemptId& attempt,
      const xllm::proto::ExecutionHolder& holder,
      xllm::proto::HolderConvergenceProof proof);

  bool has_hold() const;
  std::optional<xllm::proto::ExecutionResourceHold> snapshot() const;

 private:
  mutable std::mutex mutex_;
  std::optional<xllm::proto::ExecutionResourceHold> hold_;
  std::vector<bool> converged_holders_;
  ExecutionHoldCleanupTable::Reservation reservation_;
  bool detached_ = false;

  friend class ExecutionHoldCleanupTable;
};

}  // namespace xllm_service::provider
