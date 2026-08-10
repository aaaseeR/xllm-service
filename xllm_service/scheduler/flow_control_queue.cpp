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

#include "scheduler/flow_control_queue.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace xllm_service {

namespace {

constexpr size_t kMaxFlowIdentityLength = 256;

bool valid_flow_identity(const std::string& value) {
  return !value.empty() && value.size() <= kMaxFlowIdentityLength &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return character != 0 && character != '\n' && character != '\r';
         });
}

bool add_overflows(uint64_t left, uint64_t right) {
  return left > std::numeric_limits<uint64_t>::max() - right;
}

bool multiply_overflows(uint64_t left, uint64_t right) {
  return left != 0 && right > std::numeric_limits<uint64_t>::max() / left;
}

std::string flow_key(const FlowControlWork& work) {
  return work.model_pool + "\n" + work.tenant_id + "\n" + work.flow_id + "\n" +
         std::to_string(work.priority_band);
}

std::string tenant_band_key(const FlowControlWork& work) {
  return work.tenant_id + "\n" + std::to_string(work.priority_band);
}

}  // namespace

FlowControlQueue::FlowControlQueue(FlowControlConfig config)
    : config_(std::move(config)) {
  const bool crash_budget_overflows =
      config_.max_queued_requests > std::numeric_limits<size_t>::max() -
                                        config_.max_dispatched_request_contexts;
  valid_ = config_.max_queued_requests > 0 &&
           config_.max_dispatched_request_contexts > 0 &&
           config_.max_queued_prompt_tokens > 0 &&
           config_.max_queued_bytes > 0 && config_.max_queue_wait_ms > 0 &&
           config_.max_queued_requests_per_tenant > 0 &&
           config_.max_queued_tokens_per_tenant > 0 &&
           config_.max_model_queued_requests > 0 &&
           config_.max_model_dispatched_request_contexts > 0 &&
           config_.max_model_queued_prompt_tokens > 0 &&
           config_.max_model_queued_bytes > 0 && !crash_budget_overflows &&
           config_.service_crash_request_budget >=
               config_.max_queued_requests +
                   config_.max_dispatched_request_contexts &&
           config_.dispatched_context_bytes > 0 &&
           config_.service_memory_budget_bytes >= config_.max_queued_bytes &&
           config_.blind_dispatch_probe_concurrency > 0 &&
           config_.starvation_dispatch_bound > 0 &&
           std::isfinite(config_.dispatch_rate_lb_per_second) &&
           config_.dispatch_rate_lb_per_second >= 0.0;
  if (valid_) {
    const uint64_t contexts =
        static_cast<uint64_t>(config_.max_dispatched_request_contexts);
    valid_ = !multiply_overflows(contexts, config_.dispatched_context_bytes) &&
             !add_overflows(config_.max_queued_bytes,
                            contexts * config_.dispatched_context_bytes) &&
             config_.max_queued_bytes +
                     contexts * config_.dispatched_context_bytes <=
                 config_.service_memory_budget_bytes;
  }
}

bool FlowControlQueue::valid() const { return valid_; }

bool FlowControlQueue::has_capacity_locked(const FlowControlWork& work) const {
  const Usage& service = service_usage_;
  const auto model_it = model_usage_.find(work.model_pool);
  const Usage empty;
  const Usage& model =
      model_it == model_usage_.end() ? empty : model_it->second;
  const auto tenant_it = tenant_usage_.find(work.tenant_id);
  const Usage& tenant =
      tenant_it == tenant_usage_.end() ? empty : tenant_it->second;

  if (service.queued_requests >= config_.max_queued_requests ||
      model.queued_requests >= config_.max_model_queued_requests ||
      tenant.queued_requests >= config_.max_queued_requests_per_tenant) {
    return false;
  }
  if (add_overflows(service.queued_prompt_tokens, work.prompt_tokens) ||
      service.queued_prompt_tokens + work.prompt_tokens >
          config_.max_queued_prompt_tokens ||
      add_overflows(model.queued_prompt_tokens, work.prompt_tokens) ||
      model.queued_prompt_tokens + work.prompt_tokens >
          config_.max_model_queued_prompt_tokens ||
      add_overflows(tenant.queued_prompt_tokens, work.prompt_tokens) ||
      tenant.queued_prompt_tokens + work.prompt_tokens >
          config_.max_queued_tokens_per_tenant) {
    return false;
  }
  if (add_overflows(service.queued_bytes, work.request_bytes) ||
      service.queued_bytes + work.request_bytes > config_.max_queued_bytes ||
      add_overflows(model.queued_bytes, work.request_bytes) ||
      model.queued_bytes + work.request_bytes >
          config_.max_model_queued_bytes) {
    return false;
  }
  return service.queued_requests + service.dispatched_requests <
         config_.service_crash_request_budget;
}

uint64_t FlowControlQueue::earliest_dispatch_ms_ub_locked(
    const FlowControlWork& work,
    SaturationState state) const {
  const Usage& service = service_usage_;
  const auto model_it = model_usage_.find(work.model_pool);
  const size_t model_dispatched =
      model_it == model_usage_.end() ? 0 : model_it->second.dispatched_requests;
  const bool immediate =
      state == SaturationState::AVAILABLE &&
      service.dispatched_requests < config_.max_dispatched_request_contexts &&
      model_dispatched < config_.max_model_dispatched_request_contexts &&
      service.queued_requests == 0;
  if (immediate) {
    return 0;
  }
  if (config_.dispatch_rate_lb_per_second == 0.0 ||
      state != SaturationState::AVAILABLE) {
    return std::numeric_limits<uint64_t>::max();
  }

  size_t work_ahead = service.dispatched_requests;
  for (const auto& pair : entries_) {
    const Entry& entry = pair.second;
    if (!entry.dispatched && entry.work.priority_band <= work.priority_band) {
      ++work_ahead;
    }
  }
  const double dispatch_ms =
      std::ceil(static_cast<double>(work_ahead) * 1000.0 /
                config_.dispatch_rate_lb_per_second);
  if (dispatch_ms >= static_cast<double>(std::numeric_limits<uint64_t>::max() -
                                         config_.probe_round_ub_ms)) {
    return std::numeric_limits<uint64_t>::max();
  }
  return static_cast<uint64_t>(dispatch_ms) + config_.probe_round_ub_ms;
}

FlowControlAdmission FlowControlQueue::admit(const FlowControlWork& work,
                                             TimePoint now,
                                             SaturationState state) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (!valid_ || !valid_flow_identity(work.request_uid) ||
      !valid_flow_identity(work.model_pool) ||
      !valid_flow_identity(work.tenant_id) ||
      !valid_flow_identity(work.flow_id) || work.request_bytes == 0 ||
      work.deadline <= now) {
    return {FlowControlStatus::INVALID_ARGUMENT, 0};
  }
  if (entries_.find(work.request_uid) != entries_.end()) {
    return {FlowControlStatus::DUPLICATE_REQUEST, 0};
  }
  if (!has_capacity_locked(work)) {
    return {FlowControlStatus::QUEUE_CAPACITY_EXHAUSTED, 0};
  }

  const uint64_t earliest_ms = earliest_dispatch_ms_ub_locked(work, state);
  const uint64_t deadline_ms = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(work.deadline - now)
          .count());
  if (work.strict && earliest_ms == std::numeric_limits<uint64_t>::max()) {
    return {FlowControlStatus::QUEUE_DEADLINE_UNSATISFIABLE, earliest_ms};
  }
  if (earliest_ms != std::numeric_limits<uint64_t>::max() &&
      earliest_ms >= deadline_ms) {
    return {FlowControlStatus::QUEUE_DEADLINE_UNSATISFIABLE, earliest_ms};
  }

  const TimePoint wait_expiry =
      now + std::chrono::milliseconds(config_.max_queue_wait_ms);
  Entry entry{
      .work = work,
      .enqueue_time = now,
      .queue_expiry = std::min(wait_expiry, work.deadline),
      .enqueue_sequence = next_enqueue_sequence_++,
  };
  entries_.emplace(work.request_uid, std::move(entry));
  activate_queued_work_locked(work);

  Usage& service = service_usage_;
  Usage& model = model_usage_[work.model_pool];
  Usage& tenant = tenant_usage_[work.tenant_id];
  for (Usage* usage : {&service, &model, &tenant}) {
    ++usage->queued_requests;
    usage->queued_prompt_tokens += work.prompt_tokens;
  }
  service.queued_bytes += work.request_bytes;
  model.queued_bytes += work.request_bytes;
  return {FlowControlStatus::OK, earliest_ms};
}

void FlowControlQueue::activate_queued_work_locked(
    const FlowControlWork& work) {
  RoundRobinState& tenant = tenant_round_robin_[tenant_band_key(work)];
  if (tenant.queued_requests == 0) {
    CHECK_NE(next_tenant_turn_, std::numeric_limits<uint64_t>::max());
    tenant.turn = next_tenant_turn_++;
  }
  ++tenant.queued_requests;

  RoundRobinState& flow = flow_round_robin_[flow_key(work)];
  if (flow.queued_requests == 0) {
    CHECK_NE(next_flow_turn_, std::numeric_limits<uint64_t>::max());
    flow.turn = next_flow_turn_++;
  }
  ++flow.queued_requests;
}

void FlowControlQueue::remove_queued_work_locked(const FlowControlWork& work) {
  const std::string tenant_key = tenant_band_key(work);
  auto tenant = tenant_round_robin_.find(tenant_key);
  CHECK(tenant != tenant_round_robin_.end());
  CHECK_GT(tenant->second.queued_requests, 0u);
  if (--tenant->second.queued_requests == 0) {
    tenant_round_robin_.erase(tenant);
  }

  const std::string key = flow_key(work);
  auto flow = flow_round_robin_.find(key);
  CHECK(flow != flow_round_robin_.end());
  CHECK_GT(flow->second.queued_requests, 0u);
  if (--flow->second.queued_requests == 0) {
    flow_round_robin_.erase(flow);
  }
}

void FlowControlQueue::rotate_queued_work_after_dispatch_locked(
    const FlowControlWork& work) {
  const std::string tenant_key = tenant_band_key(work);
  auto tenant = tenant_round_robin_.find(tenant_key);
  CHECK(tenant != tenant_round_robin_.end());
  CHECK_GT(tenant->second.queued_requests, 0u);
  if (--tenant->second.queued_requests == 0) {
    tenant_round_robin_.erase(tenant);
  } else {
    CHECK_NE(next_tenant_turn_, std::numeric_limits<uint64_t>::max());
    tenant->second.turn = next_tenant_turn_++;
  }

  const std::string key = flow_key(work);
  auto flow = flow_round_robin_.find(key);
  CHECK(flow != flow_round_robin_.end());
  CHECK_GT(flow->second.queued_requests, 0u);
  if (--flow->second.queued_requests == 0) {
    flow_round_robin_.erase(flow);
  } else {
    CHECK_NE(next_flow_turn_, std::numeric_limits<uint64_t>::max());
    flow->second.turn = next_flow_turn_++;
  }
}

std::optional<std::string> FlowControlQueue::select_next_locked(
    TimePoint now,
    bool allow_strict) const {
  int32_t highest_band = std::numeric_limits<int32_t>::max();
  int32_t oldest_starved_band = std::numeric_limits<int32_t>::max();
  const Entry* oldest_lower = nullptr;

  for (const auto& pair : entries_) {
    const Entry& entry = pair.second;
    const auto model_it = model_usage_.find(entry.work.model_pool);
    const bool model_full = model_it != model_usage_.end() &&
                            model_it->second.dispatched_requests >=
                                config_.max_model_dispatched_request_contexts;
    if (entry.dispatched || entry.queue_expiry <= now || model_full ||
        (!allow_strict && entry.work.strict)) {
      continue;
    }
    highest_band = std::min(highest_band, entry.work.priority_band);
  }
  if (highest_band == std::numeric_limits<int32_t>::max()) {
    return std::nullopt;
  }

  if (consecutive_priority_dispatches_ >= config_.starvation_dispatch_bound) {
    for (const auto& pair : entries_) {
      const Entry& entry = pair.second;
      const auto model_it = model_usage_.find(entry.work.model_pool);
      const bool model_full = model_it != model_usage_.end() &&
                              model_it->second.dispatched_requests >=
                                  config_.max_model_dispatched_request_contexts;
      if (entry.dispatched || entry.queue_expiry <= now || model_full ||
          (!allow_strict && entry.work.strict) ||
          entry.work.priority_band <= highest_band) {
        continue;
      }
      if (oldest_lower == nullptr ||
          entry.enqueue_sequence < oldest_lower->enqueue_sequence) {
        oldest_lower = &entry;
        oldest_starved_band = entry.work.priority_band;
      }
    }
  }
  const int32_t target_band =
      oldest_lower == nullptr ? highest_band : oldest_starved_band;

  const Entry* selected_tenant_entry = nullptr;
  for (const auto& pair : entries_) {
    const Entry& entry = pair.second;
    const auto model_it = model_usage_.find(entry.work.model_pool);
    const bool model_full = model_it != model_usage_.end() &&
                            model_it->second.dispatched_requests >=
                                config_.max_model_dispatched_request_contexts;
    if (entry.dispatched || entry.queue_expiry <= now || model_full ||
        (!allow_strict && entry.work.strict) ||
        entry.work.priority_band != target_band) {
      continue;
    }
    if (selected_tenant_entry == nullptr) {
      selected_tenant_entry = &entry;
      continue;
    }
    const RoundRobinState& entry_tenant =
        tenant_round_robin_.at(tenant_band_key(entry.work));
    const RoundRobinState& selected_tenant =
        tenant_round_robin_.at(tenant_band_key(selected_tenant_entry->work));
    if (entry_tenant.turn < selected_tenant.turn ||
        (entry_tenant.turn == selected_tenant.turn &&
         entry.enqueue_sequence < selected_tenant_entry->enqueue_sequence)) {
      selected_tenant_entry = &entry;
    }
  }

  if (selected_tenant_entry == nullptr) {
    return std::nullopt;
  }
  const std::string& selected_tenant_id = selected_tenant_entry->work.tenant_id;

  const Entry* selected_flow_entry = nullptr;
  for (const auto& pair : entries_) {
    const Entry& entry = pair.second;
    const auto model_it = model_usage_.find(entry.work.model_pool);
    const bool model_full = model_it != model_usage_.end() &&
                            model_it->second.dispatched_requests >=
                                config_.max_model_dispatched_request_contexts;
    if (entry.dispatched || entry.queue_expiry <= now || model_full ||
        (!allow_strict && entry.work.strict) ||
        entry.work.priority_band != target_band ||
        entry.work.tenant_id != selected_tenant_id) {
      continue;
    }
    if (selected_flow_entry == nullptr) {
      selected_flow_entry = &entry;
      continue;
    }
    const RoundRobinState& entry_flow =
        flow_round_robin_.at(flow_key(entry.work));
    const RoundRobinState& selected_flow =
        flow_round_robin_.at(flow_key(selected_flow_entry->work));
    if (entry_flow.turn < selected_flow.turn ||
        (entry_flow.turn == selected_flow.turn &&
         entry.enqueue_sequence < selected_flow_entry->enqueue_sequence)) {
      selected_flow_entry = &entry;
    }
  }

  if (selected_flow_entry == nullptr) {
    return std::nullopt;
  }
  const std::string selected_flow_key = flow_key(selected_flow_entry->work);
  const Entry* selected = nullptr;
  for (const auto& pair : entries_) {
    const Entry& entry = pair.second;
    const auto model_it = model_usage_.find(entry.work.model_pool);
    const bool model_full = model_it != model_usage_.end() &&
                            model_it->second.dispatched_requests >=
                                config_.max_model_dispatched_request_contexts;
    if (entry.dispatched || entry.queue_expiry <= now || model_full ||
        (!allow_strict && entry.work.strict) ||
        entry.work.priority_band != target_band ||
        flow_key(entry.work) != selected_flow_key) {
      continue;
    }
    if (selected == nullptr) {
      selected = &entry;
      continue;
    }
    if (config_.flow_order == FlowOrder::EDF &&
        entry.work.deadline != selected->work.deadline) {
      if (entry.work.deadline < selected->work.deadline) {
        selected = &entry;
      }
      continue;
    }
    if (entry.enqueue_sequence < selected->enqueue_sequence) {
      selected = &entry;
    }
  }
  return selected == nullptr
             ? std::nullopt
             : std::optional<std::string>(selected->work.request_uid);
}

FlowControlDispatch FlowControlQueue::take_next(TimePoint now,
                                                SaturationState state) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (!valid_) {
    return {FlowControlStatus::INVALID_ARGUMENT, std::nullopt, false};
  }
  if (state == SaturationState::SATURATED) {
    return {FlowControlStatus::QUEUE_CAPACITY_EXHAUSTED, std::nullopt, false};
  }
  const bool blind_probe = state == SaturationState::UNKNOWN;
  if (blind_probe &&
      blind_probes_inflight_ >= config_.blind_dispatch_probe_concurrency) {
    return {FlowControlStatus::QUEUE_CAPACITY_EXHAUSTED, std::nullopt, false};
  }
  if (service_usage_.dispatched_requests >=
      config_.max_dispatched_request_contexts) {
    return {FlowControlStatus::QUEUE_CAPACITY_EXHAUSTED, std::nullopt, false};
  }

  std::optional<std::string> uid = select_next_locked(
      now, /*allow_strict=*/state == SaturationState::AVAILABLE);
  if (!uid.has_value()) {
    return {FlowControlStatus::UNKNOWN_REQUEST, std::nullopt, false};
  }
  Entry& entry = entries_.at(*uid);
  Usage& model = model_usage_.at(entry.work.model_pool);
  if (model.dispatched_requests >=
      config_.max_model_dispatched_request_contexts) {
    return {FlowControlStatus::QUEUE_CAPACITY_EXHAUSTED, std::nullopt, false};
  }

  Usage& service = service_usage_;
  Usage& tenant = tenant_usage_.at(entry.work.tenant_id);
  for (Usage* usage : {&service, &model, &tenant}) {
    --usage->queued_requests;
    usage->queued_prompt_tokens -= entry.work.prompt_tokens;
    ++usage->dispatched_requests;
  }
  service.queued_bytes -= entry.work.request_bytes;
  model.queued_bytes -= entry.work.request_bytes;
  rotate_queued_work_after_dispatch_locked(entry.work);
  entry.dispatched = true;
  entry.blind_probe = blind_probe;
  if (blind_probe) {
    ++blind_probes_inflight_;
  }
  if (entry.work.priority_band == last_priority_band_) {
    ++consecutive_priority_dispatches_;
  } else {
    last_priority_band_ = entry.work.priority_band;
    consecutive_priority_dispatches_ = 1;
  }
  return {FlowControlStatus::OK, entry.work, blind_probe};
}

void FlowControlQueue::erase_queued_locked(
    std::unordered_map<std::string, Entry>::iterator it) {
  const FlowControlWork work = it->second.work;
  Usage& service = service_usage_;
  Usage& model = model_usage_.at(work.model_pool);
  Usage& tenant = tenant_usage_.at(work.tenant_id);
  for (Usage* usage : {&service, &model, &tenant}) {
    --usage->queued_requests;
    usage->queued_prompt_tokens -= work.prompt_tokens;
  }
  service.queued_bytes -= work.request_bytes;
  model.queued_bytes -= work.request_bytes;
  remove_queued_work_locked(work);
  entries_.erase(it);
  erase_empty_usage_locked(work);
}

void FlowControlQueue::decrement_dispatched_locked(const Entry& entry,
                                                   bool erase_empty_usage) {
  Usage& service = service_usage_;
  Usage& model = model_usage_.at(entry.work.model_pool);
  Usage& tenant = tenant_usage_.at(entry.work.tenant_id);
  for (Usage* usage : {&service, &model, &tenant}) {
    CHECK_GT(usage->dispatched_requests, 0u);
    --usage->dispatched_requests;
  }
  if (entry.blind_probe) {
    --blind_probes_inflight_;
  }
  if (erase_empty_usage) {
    erase_empty_usage_locked(entry.work);
  }
}

void FlowControlQueue::erase_empty_usage_locked(const FlowControlWork& work) {
  const auto empty = [](const Usage& usage) {
    return usage.queued_requests == 0 && usage.dispatched_requests == 0 &&
           usage.queued_prompt_tokens == 0 && usage.queued_bytes == 0;
  };
  const auto model = model_usage_.find(work.model_pool);
  if (model != model_usage_.end() && empty(model->second)) {
    model_usage_.erase(model);
  }
  const auto tenant = tenant_usage_.find(work.tenant_id);
  if (tenant != tenant_usage_.end() && empty(tenant->second)) {
    tenant_usage_.erase(tenant);
  }
}

FlowControlStatus FlowControlQueue::cancel(const std::string& request_uid) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = entries_.find(request_uid);
  if (it == entries_.end()) {
    return FlowControlStatus::UNKNOWN_REQUEST;
  }
  if (it->second.dispatched) {
    decrement_dispatched_locked(it->second, /*erase_empty_usage=*/true);
    entries_.erase(it);
  } else {
    erase_queued_locked(it);
  }
  return FlowControlStatus::OK;
}

FlowControlStatus FlowControlQueue::complete(const std::string& request_uid) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = entries_.find(request_uid);
  if (it == entries_.end()) {
    return FlowControlStatus::UNKNOWN_REQUEST;
  }
  if (!it->second.dispatched) {
    return FlowControlStatus::INVALID_ARGUMENT;
  }
  decrement_dispatched_locked(it->second, /*erase_empty_usage=*/true);
  entries_.erase(it);
  return FlowControlStatus::OK;
}

FlowControlStatus FlowControlQueue::return_to_queue(
    const std::string& request_uid) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = entries_.find(request_uid);
  if (it == entries_.end()) {
    return FlowControlStatus::UNKNOWN_REQUEST;
  }
  Entry& entry = it->second;
  if (!entry.dispatched) {
    return FlowControlStatus::INVALID_ARGUMENT;
  }
  decrement_dispatched_locked(entry, /*erase_empty_usage=*/false);

  Usage& service = service_usage_;
  Usage& model = model_usage_.at(entry.work.model_pool);
  Usage& tenant = tenant_usage_.at(entry.work.tenant_id);
  for (Usage* usage : {&service, &model, &tenant}) {
    ++usage->queued_requests;
    usage->queued_prompt_tokens += entry.work.prompt_tokens;
  }
  service.queued_bytes += entry.work.request_bytes;
  model.queued_bytes += entry.work.request_bytes;
  activate_queued_work_locked(entry.work);
  entry.dispatched = false;
  entry.blind_probe = false;
  return FlowControlStatus::OK;
}

std::vector<FlowControlWork> FlowControlQueue::take_expired(TimePoint now,
                                                            size_t max_items) {
  std::lock_guard<std::mutex> guard(mutex_);
  std::vector<FlowControlWork> expired;
  expired.reserve(std::min(max_items, entries_.size()));
  for (auto it = entries_.begin();
       it != entries_.end() && expired.size() < max_items;) {
    if (it->second.dispatched || it->second.queue_expiry > now) {
      ++it;
      continue;
    }
    expired.emplace_back(it->second.work);
    auto erase_it = it++;
    erase_queued_locked(erase_it);
  }
  return expired;
}

std::vector<FlowControlWork> FlowControlQueue::retry_undispatched(
    size_t max_items) {
  std::lock_guard<std::mutex> guard(mutex_);
  std::vector<FlowControlWork> retried;
  retried.reserve(std::min(max_items, entries_.size()));
  for (auto it = entries_.begin();
       it != entries_.end() && retried.size() < max_items;) {
    if (it->second.dispatched) {
      ++it;
      continue;
    }
    retried.emplace_back(it->second.work);
    auto erase_it = it++;
    erase_queued_locked(erase_it);
  }
  return retried;
}

FlowControlSnapshot FlowControlQueue::snapshot() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return FlowControlSnapshot{
      .queued_requests = service_usage_.queued_requests,
      .dispatched_requests = service_usage_.dispatched_requests,
      .queued_prompt_tokens = service_usage_.queued_prompt_tokens,
      .queued_bytes = service_usage_.queued_bytes,
      .dispatched_context_bytes =
          service_usage_.dispatched_requests * config_.dispatched_context_bytes,
      .blind_probes_inflight = blind_probes_inflight_,
      .active_model_accounts = model_usage_.size(),
      .active_tenant_accounts = tenant_usage_.size(),
      .active_queued_flows = flow_round_robin_.size(),
  };
}

FlowControlModelSnapshot FlowControlQueue::model_snapshot(
    const std::string& model_pool) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto iterator = model_usage_.find(model_pool);
  if (iterator == model_usage_.end()) {
    return {};
  }
  return FlowControlModelSnapshot{
      .queued_requests = iterator->second.queued_requests,
      .dispatched_requests = iterator->second.dispatched_requests,
      .queued_prompt_tokens = iterator->second.queued_prompt_tokens,
      .queued_bytes = iterator->second.queued_bytes,
  };
}

}  // namespace xllm_service
