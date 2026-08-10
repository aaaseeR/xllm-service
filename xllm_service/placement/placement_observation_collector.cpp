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

#include "placement/placement_observation_collector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace xllm_service::placement {
namespace {

bool add_saturated(uint64_t value, uint64_t* target) {
  if (target == nullptr) {
    return false;
  }
  if (value > std::numeric_limits<uint64_t>::max() - *target) {
    *target = std::numeric_limits<uint64_t>::max();
    return false;
  }
  *target += value;
  return true;
}

bool finite_nonnegative(double value) {
  return std::isfinite(value) && value >= 0.0;
}

double quantile95(std::vector<double> samples) {
  if (samples.empty()) {
    return 0.0;
  }
  const size_t rank = static_cast<size_t>(
      std::ceil(0.95 * static_cast<double>(samples.size())));
  const size_t index = std::max<size_t>(rank, 1) - 1;
  std::nth_element(samples.begin(), samples.begin() + index, samples.end());
  return samples[index];
}

}  // namespace

PlacementObservationCollector::PlacementObservationCollector(
    PlacementObservationCollectorConfig config)
    : config_(std::move(config)),
      valid_(valid_placement_observation_collector_config(config_)) {}

PlacementObservationStatus PlacementObservationCollector::register_model(
    const std::string& model_revision,
    uint64_t now_monotonic_ms) {
  std::lock_guard<std::mutex> lock(mutex_);
  ModelWindow* window = nullptr;
  Bucket* bucket = nullptr;
  return get_bucket_locked(model_revision, now_monotonic_ms, &window, &bucket);
}

PlacementObservationStatus PlacementObservationCollector::get_bucket_locked(
    const std::string& model_revision,
    uint64_t now_monotonic_ms,
    ModelWindow** window,
    Bucket** bucket) {
  if (!valid_ || !valid_placement_identity(model_revision) ||
      now_monotonic_ms == 0 || window == nullptr || bucket == nullptr) {
    return PlacementObservationStatus::INVALID_INPUT;
  }
  const uint64_t epoch = now_monotonic_ms / config_.bucket_width_ms;
  auto iterator = models_.find(model_revision);
  if (iterator == models_.end()) {
    if (models_.size() >= config_.max_models) {
      return PlacementObservationStatus::CAPACITY_EXCEEDED;
    }
    ModelWindow created;
    created.buckets.resize(config_.bucket_count);
    iterator = models_.emplace(model_revision, std::move(created)).first;
  }
  ModelWindow& model = iterator->second;
  if (model.latest_epoch != 0 && epoch < model.latest_epoch &&
      model.latest_epoch - epoch >= config_.bucket_count) {
    return PlacementObservationStatus::CLOCK_REGRESSION;
  }
  model.latest_epoch = std::max(model.latest_epoch, epoch);
  Bucket& selected = model.buckets[epoch % config_.bucket_count];
  if (!selected.initialized || selected.epoch != epoch) {
    if (selected.initialized && selected.epoch > epoch) {
      return PlacementObservationStatus::CLOCK_REGRESSION;
    }
    selected = Bucket{};
    selected.initialized = true;
    selected.epoch = epoch;
    selected.ttft_ms.reserve(config_.max_latency_samples_per_bucket);
    selected.tpot_ms.reserve(config_.max_latency_samples_per_bucket);
  }
  *window = &model;
  *bucket = &selected;
  return PlacementObservationStatus::OK;
}

PlacementObservationStatus PlacementObservationCollector::record_ingress(
    const std::string& model_revision,
    uint64_t prompt_tokens,
    uint64_t effective_max_output_tokens,
    uint64_t now_monotonic_ms) {
  std::lock_guard<std::mutex> lock(mutex_);
  ModelWindow* window = nullptr;
  Bucket* bucket = nullptr;
  const PlacementObservationStatus status =
      get_bucket_locked(model_revision, now_monotonic_ms, &window, &bucket);
  static_cast<void>(window);
  if (status != PlacementObservationStatus::OK) {
    return status;
  }
  const bool accepted_ok = add_saturated(1, &bucket->accepted);
  const bool prompt_ok = add_saturated(prompt_tokens, &bucket->prompt_tokens);
  const bool output_ok = add_saturated(effective_max_output_tokens,
                                       &bucket->requested_output_tokens);
  bucket->overflowed =
      bucket->overflowed || !accepted_ok || !prompt_ok || !output_ok;
  return PlacementObservationStatus::OK;
}

PlacementObservationStatus
PlacementObservationCollector::record_admission_reject(
    const std::string& model_revision,
    uint64_t now_monotonic_ms) {
  std::lock_guard<std::mutex> lock(mutex_);
  ModelWindow* window = nullptr;
  Bucket* bucket = nullptr;
  const PlacementObservationStatus status =
      get_bucket_locked(model_revision, now_monotonic_ms, &window, &bucket);
  static_cast<void>(window);
  if (status != PlacementObservationStatus::OK) {
    return status;
  }
  if (!add_saturated(1, &bucket->rejected)) {
    bucket->overflowed = true;
  }
  return PlacementObservationStatus::OK;
}

PlacementObservationStatus PlacementObservationCollector::record_terminal(
    const std::string& model_revision,
    uint64_t output_tokens,
    std::optional<double> ttft_ms,
    std::optional<double> tpot_ms,
    uint64_t now_monotonic_ms) {
  if ((ttft_ms.has_value() && !finite_nonnegative(*ttft_ms)) ||
      (tpot_ms.has_value() && !finite_nonnegative(*tpot_ms))) {
    return PlacementObservationStatus::INVALID_INPUT;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  ModelWindow* window = nullptr;
  Bucket* bucket = nullptr;
  const PlacementObservationStatus status =
      get_bucket_locked(model_revision, now_monotonic_ms, &window, &bucket);
  static_cast<void>(window);
  if (status != PlacementObservationStatus::OK) {
    return status;
  }
  const bool terminal_ok = add_saturated(1, &bucket->terminal);
  const bool output_ok = add_saturated(output_tokens, &bucket->output_tokens);
  bucket->overflowed = bucket->overflowed || !terminal_ok || !output_ok;
  auto append_latency = [this, bucket](std::optional<double> value,
                                       std::vector<double>* samples) {
    if (!value.has_value()) {
      return;
    }
    if (samples->size() >= config_.max_latency_samples_per_bucket) {
      bucket->latency_truncated = true;
      return;
    }
    samples->push_back(*value);
  };
  append_latency(ttft_ms, &bucket->ttft_ms);
  append_latency(tpot_ms, &bucket->tpot_ms);
  return PlacementObservationStatus::OK;
}

PlacementObservationStatus PlacementObservationCollector::snapshot(
    const std::string& model_revision,
    uint64_t now_monotonic_ms,
    const PlacementObservationExternalInputs& external,
    PlacementObservation* observation) {
  if (!valid_ || !valid_placement_identity(model_revision) ||
      now_monotonic_ms == 0 || observation == nullptr ||
      !finite_nonnegative(external.queue_depth) ||
      !finite_nonnegative(external.kv_used_ratio) ||
      external.kv_used_ratio > 1.0 ||
      !finite_nonnegative(external.full_cache_loss_cost) ||
      !finite_nonnegative(external.confirmed_store_coverage) ||
      external.confirmed_store_coverage > 1.0) {
    return PlacementObservationStatus::INVALID_INPUT;
  }
  const uint64_t now_epoch = now_monotonic_ms / config_.bucket_width_ms;
  uint64_t generation = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iterator = models_.find(model_revision);
    if (iterator == models_.end()) {
      return PlacementObservationStatus::NOT_FOUND;
    }
    ModelWindow& model = iterator->second;
    if (now_epoch < model.latest_epoch) {
      return PlacementObservationStatus::CLOCK_REGRESSION;
    }
    if (model.next_generation == std::numeric_limits<uint64_t>::max()) {
      return PlacementObservationStatus::CAPACITY_EXCEEDED;
    }
    model.latest_epoch = now_epoch;
    generation = model.next_generation++;
  }
  const uint64_t earliest_epoch =
      now_epoch >= config_.bucket_count - 1
          ? now_epoch - static_cast<uint64_t>(config_.bucket_count - 1)
          : 0;
  uint64_t accepted = 0;
  uint64_t rejected = 0;
  uint64_t terminal = 0;
  uint64_t prompt_tokens = 0;
  uint64_t requested_output_tokens = 0;
  uint64_t output_tokens = 0;
  uint64_t first_epoch = now_epoch;
  bool has_bucket = false;
  bool data_loss = false;
  std::vector<double> ttft_samples;
  std::vector<double> tpot_samples;
  const size_t latency_capacity =
      config_.bucket_count * config_.max_latency_samples_per_bucket;
  ttft_samples.reserve(latency_capacity);
  tpot_samples.reserve(latency_capacity);
  auto add_total = [&data_loss](uint64_t value, uint64_t* total) {
    if (!add_saturated(value, total)) {
      data_loss = true;
    }
  };
  for (size_t bucket_index = 0; bucket_index < config_.bucket_count;
       ++bucket_index) {
    Bucket bucket;
    {
      // Copy at most one bounded bucket while request producers are paused.
      // Aggregation and quantile selection deliberately remain outside the
      // request-path mutex.
      std::lock_guard<std::mutex> lock(mutex_);
      const auto iterator = models_.find(model_revision);
      if (iterator == models_.end()) {
        return PlacementObservationStatus::NOT_FOUND;
      }
      bucket = iterator->second.buckets[bucket_index];
    }
    if (!bucket.initialized || bucket.epoch < earliest_epoch ||
        bucket.epoch > now_epoch) {
      continue;
    }
    has_bucket = true;
    first_epoch = std::min(first_epoch, bucket.epoch);
    add_total(bucket.accepted, &accepted);
    add_total(bucket.rejected, &rejected);
    add_total(bucket.terminal, &terminal);
    add_total(bucket.prompt_tokens, &prompt_tokens);
    add_total(bucket.requested_output_tokens, &requested_output_tokens);
    add_total(bucket.output_tokens, &output_tokens);
    data_loss = data_loss || bucket.overflowed || bucket.latency_truncated;
    ttft_samples.insert(
        ttft_samples.end(), bucket.ttft_ms.begin(), bucket.ttft_ms.end());
    tpot_samples.insert(
        tpot_samples.end(), bucket.tpot_ms.begin(), bucket.tpot_ms.end());
  }
  const uint64_t full_window_ms =
      static_cast<uint64_t>(config_.bucket_count) * config_.bucket_width_ms;
  uint64_t observed_window_ms = config_.bucket_width_ms;
  if (has_bucket && now_epoch >= first_epoch) {
    const uint64_t bucket_span = now_epoch - first_epoch + 1;
    if (bucket_span <=
        std::numeric_limits<uint64_t>::max() / config_.bucket_width_ms) {
      observed_window_ms =
          std::min(full_window_ms, bucket_span * config_.bucket_width_ms);
    } else {
      observed_window_ms = full_window_ms;
      data_loss = true;
    }
  }
  const double seconds = static_cast<double>(observed_window_ms) / 1000.0;
  const double raw_rate = static_cast<double>(accepted) / seconds;
  const uint64_t admission_total =
      rejected > std::numeric_limits<uint64_t>::max() - accepted
          ? std::numeric_limits<uint64_t>::max()
          : accepted + rejected;
  if (admission_total == std::numeric_limits<uint64_t>::max() &&
      rejected > std::numeric_limits<uint64_t>::max() - accepted) {
    data_loss = true;
  }
  const double average_prompt =
      accepted == 0
          ? 0.0
          : static_cast<double>(prompt_tokens) / static_cast<double>(accepted);
  const double average_output =
      terminal > 0
          ? static_cast<double>(output_tokens) / static_cast<double>(terminal)
          : (accepted == 0 ? 0.0
                           : static_cast<double>(requested_output_tokens) /
                                 static_cast<double>(accepted));
  const double forecast_rate = raw_rate * config_.forecast_headroom;
  if (!std::isfinite(forecast_rate) || !std::isfinite(average_prompt) ||
      !std::isfinite(average_output)) {
    return PlacementObservationStatus::CAPACITY_EXCEEDED;
  }
  *observation = PlacementObservation{
      .generation = generation,
      .observed_at_ms = now_monotonic_ms,
      .window_ms = observed_window_ms,
      .forecast_horizon_ms = config_.forecast_horizon_ms,
      .forecast_request_rate = forecast_rate,
      .prompt_tokens_per_request = average_prompt,
      .output_tokens_per_request = average_output,
      .queue_depth = external.queue_depth,
      .admission_reject_rate = admission_total == 0
                                   ? 0.0
                                   : static_cast<double>(rejected) /
                                         static_cast<double>(admission_total),
      .ttft_p95_ms = quantile95(std::move(ttft_samples)),
      .tpot_p95_ms = quantile95(std::move(tpot_samples)),
      .kv_used_ratio = external.kv_used_ratio,
      .major_bucket_samples = terminal,
      .out_of_distribution = external.out_of_distribution || data_loss,
      .cold_start = accepted == 0 || terminal == 0,
      .full_cache_loss_cost = external.full_cache_loss_cost,
      .confirmed_store_coverage = external.confirmed_store_coverage,
  };
  return valid_placement_observation(*observation)
             ? PlacementObservationStatus::OK
             : PlacementObservationStatus::CAPACITY_EXCEEDED;
}

bool PlacementObservationCollector::valid() const { return valid_; }

size_t PlacementObservationCollector::model_count() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return models_.size();
}

bool valid_placement_observation_collector_config(
    const PlacementObservationCollectorConfig& config) {
  if (config.max_models == 0 || config.bucket_count == 0 ||
      config.bucket_width_ms == 0 ||
      config.max_latency_samples_per_bucket == 0 ||
      config.forecast_horizon_ms == 0 ||
      !std::isfinite(config.forecast_headroom) ||
      config.forecast_headroom < 1.0) {
    return false;
  }
  if (config.max_models > kMaxPlacementObservationModels ||
      config.bucket_count > kMaxPlacementObservationBuckets ||
      config.max_latency_samples_per_bucket >
          kMaxPlacementLatencySamplesPerBucket) {
    return false;
  }
  if (config.bucket_count >
          std::numeric_limits<uint64_t>::max() / config.bucket_width_ms ||
      config.bucket_count > std::numeric_limits<size_t>::max() /
                                config.max_latency_samples_per_bucket ||
      config.max_models >
          std::numeric_limits<size_t>::max() / config.bucket_count ||
      config.max_models * config.bucket_count >
          std::numeric_limits<size_t>::max() /
              config.max_latency_samples_per_bucket) {
    return false;
  }
  return config.max_models * config.bucket_count *
             config.max_latency_samples_per_bucket <=
         kMaxPlacementLatencySlots;
}

const char* placement_observation_status_name(
    PlacementObservationStatus status) {
  switch (status) {
    case PlacementObservationStatus::OK:
      return "OK";
    case PlacementObservationStatus::INVALID_INPUT:
      return "INVALID_INPUT";
    case PlacementObservationStatus::CAPACITY_EXCEEDED:
      return "CAPACITY_EXCEEDED";
    case PlacementObservationStatus::CLOCK_REGRESSION:
      return "CLOCK_REGRESSION";
    case PlacementObservationStatus::NOT_FOUND:
      return "NOT_FOUND";
  }
  return "UNKNOWN";
}

}  // namespace xllm_service::placement
