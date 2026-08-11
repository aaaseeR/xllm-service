/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm_service {

// Capacity evidence is deliberately tri-state. Missing or stale evidence is
// never silently converted to either an available or a saturated pool.
enum class SaturationState {
  AVAILABLE = 0,
  SATURATED = 1,
  UNKNOWN = 2,
};

// Converts the latest readiness/capacity observation into the tri-state used
// by admission and dispatch. Keeping this classification independent from the
// registry lets Scheduler cache it and keeps O(P*D) route enumeration off the
// request lock.
SaturationState saturation_state_from_readiness(bool ready,
                                                bool draining,
                                                bool has_capacity);

enum class FlowOrder {
  FCFS = 0,
  EDF = 1,
};

enum class FlowControlStatus {
  OK = 0,
  QUEUE_CAPACITY_EXHAUSTED = 1,
  QUEUE_DEADLINE_UNSATISFIABLE = 2,
  DUPLICATE_REQUEST = 3,
  UNKNOWN_REQUEST = 4,
  INVALID_ARGUMENT = 5,
};

struct FlowControlConfig {
  size_t max_queued_requests = 1;
  size_t max_dispatched_request_contexts = 1;
  uint64_t max_queued_prompt_tokens = 1;
  uint64_t max_queued_bytes = 1;
  uint64_t max_queue_wait_ms = 1;
  size_t max_queued_requests_per_tenant = 1;
  uint64_t max_queued_tokens_per_tenant = 1;

  // The same hard limits apply independently to every ModelPool.
  size_t max_model_queued_requests = 1;
  size_t max_model_dispatched_request_contexts = 1;
  uint64_t max_model_queued_prompt_tokens = 1;
  uint64_t max_model_queued_bytes = 1;

  size_t service_crash_request_budget = 2;
  uint64_t service_memory_budget_bytes = 2;
  uint64_t dispatched_context_bytes = 1;

  // A zero rate admits only an immediately dispatchable request. There is no
  // configurable positive floor for blind operation.
  double dispatch_rate_lb_per_second = 1.0;
  uint64_t probe_round_ub_ms = 0;
  size_t blind_dispatch_probe_concurrency = 1;
  size_t starvation_dispatch_bound = 32;
  FlowOrder flow_order = FlowOrder::FCFS;
};

struct FlowControlWork {
  std::string request_uid;
  std::string model_pool;
  std::string tenant_id;
  std::string flow_id;

  // Lower numeric values are higher priority, matching xllm::proto::Priority.
  int32_t priority_band = 0;
  uint64_t prompt_tokens = 0;
  uint64_t request_bytes = 0;
  std::chrono::steady_clock::time_point deadline;
  bool strict = false;
};

struct FlowControlSnapshot {
  size_t queued_requests = 0;
  size_t dispatched_requests = 0;
  uint64_t queued_prompt_tokens = 0;
  uint64_t queued_bytes = 0;
  uint64_t dispatched_context_bytes = 0;
  size_t blind_probes_inflight = 0;
  size_t active_model_accounts = 0;
  size_t active_tenant_accounts = 0;
  size_t active_queued_flows = 0;
};

struct FlowControlModelSnapshot {
  size_t queued_requests = 0;
  size_t dispatched_requests = 0;
  uint64_t queued_prompt_tokens = 0;
  uint64_t queued_bytes = 0;
};

struct FlowControlAdmission {
  FlowControlStatus status = FlowControlStatus::INVALID_ARGUMENT;
  uint64_t earliest_dispatch_ms_ub = 0;
};

struct FlowControlDispatch {
  FlowControlStatus status = FlowControlStatus::UNKNOWN_REQUEST;
  std::optional<FlowControlWork> work;
  bool blind_probe = false;
};

// A bounded, request-content-free Service queue. All accounting transitions
// happen under one mutex so admission cannot partially reserve token, byte,
// tenant, ModelPool, or crash-exposure budgets.
class FlowControlQueue final {
 public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  explicit FlowControlQueue(FlowControlConfig config);

  bool valid() const;

  FlowControlAdmission admit(const FlowControlWork& work,
                             TimePoint now,
                             SaturationState state);

  // AVAILABLE performs normal work-conserving dispatch. SATURATED keeps all
  // work queued. UNKNOWN permits only bounded BEST_EFFORT blind probes.
  FlowControlDispatch take_next(TimePoint now, SaturationState state);

  FlowControlStatus cancel(const std::string& request_uid);
  FlowControlStatus complete(const std::string& request_uid);
  FlowControlStatus return_to_queue(const std::string& request_uid);

  std::vector<FlowControlWork> take_expired(TimePoint now, size_t max_items);
  std::vector<FlowControlWork> retry_undispatched(size_t max_items);

  FlowControlSnapshot snapshot() const;
  FlowControlModelSnapshot model_snapshot(const std::string& model_pool) const;

 private:
  struct Entry {
    FlowControlWork work;
    TimePoint enqueue_time;
    TimePoint queue_expiry;
    uint64_t enqueue_sequence = 0;
    bool dispatched = false;
    bool blind_probe = false;
  };

  struct Usage {
    size_t queued_requests = 0;
    size_t dispatched_requests = 0;
    uint64_t queued_prompt_tokens = 0;
    uint64_t queued_bytes = 0;
  };

  struct RoundRobinState {
    uint64_t turn = 0;
    size_t queued_requests = 0;
  };

  bool has_capacity_locked(const FlowControlWork& work) const;
  uint64_t earliest_dispatch_ms_ub_locked(const FlowControlWork& work,
                                          SaturationState state) const;
  std::optional<std::string> select_next_locked(TimePoint now,
                                                bool allow_strict) const;
  void activate_queued_work_locked(const FlowControlWork& work);
  void remove_queued_work_locked(const FlowControlWork& work);
  void rotate_queued_work_after_dispatch_locked(const FlowControlWork& work);
  void erase_queued_locked(std::unordered_map<std::string, Entry>::iterator it);
  void decrement_dispatched_locked(const Entry& entry, bool erase_empty_usage);
  void erase_empty_usage_locked(const FlowControlWork& work);

  FlowControlConfig config_;
  bool valid_ = false;

  mutable std::mutex mutex_;
  std::unordered_map<std::string, Entry> entries_;
  Usage service_usage_;
  std::unordered_map<std::string, Usage> model_usage_;
  std::unordered_map<std::string, Usage> tenant_usage_;
  std::unordered_map<std::string, RoundRobinState> tenant_round_robin_;
  std::unordered_map<std::string, RoundRobinState> flow_round_robin_;
  uint64_t next_tenant_turn_ = 0;
  uint64_t next_flow_turn_ = 0;
  uint64_t next_enqueue_sequence_ = 0;
  int32_t last_priority_band_ = 0;
  size_t consecutive_priority_dispatches_ = 0;
  size_t blind_probes_inflight_ = 0;
};

}  // namespace xllm_service
