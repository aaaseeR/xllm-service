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

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "placement/placement_types.h"

namespace xllm_service::placement {

enum class PlacementObservationStatus : int8_t {
  OK = 0,
  INVALID_INPUT = 1,
  CAPACITY_EXCEEDED = 2,
  CLOCK_REGRESSION = 3,
  NOT_FOUND = 4,
};

struct PlacementObservationCollectorConfig {
  size_t max_models = 0;
  size_t bucket_count = 0;
  uint64_t bucket_width_ms = 0;
  size_t max_latency_samples_per_bucket = 0;
  uint64_t forecast_horizon_ms = 0;
  double forecast_headroom = 0.0;
};

struct PlacementObservationExternalInputs {
  double queue_depth = 0.0;
  double kv_used_ratio = 0.0;
  double full_cache_loss_cost = 0.0;
  double confirmed_store_coverage = 0.0;
  bool out_of_distribution = false;
};

// Thread-safe, allocation-bounded workload observation plane. Producers feed
// it regardless of VLOG settings; fixed monotonic buckets prevent diagnostic
// logging or exporter backpressure from affecting V3 decisions.
class PlacementObservationCollector final {
 public:
  explicit PlacementObservationCollector(
      PlacementObservationCollectorConfig config);

  PlacementObservationStatus record_ingress(
      const std::string& model_revision,
      uint64_t prompt_tokens,
      uint64_t effective_max_output_tokens,
      uint64_t now_monotonic_ms);

  PlacementObservationStatus record_admission_reject(
      const std::string& model_revision,
      uint64_t now_monotonic_ms);

  PlacementObservationStatus record_terminal(const std::string& model_revision,
                                             uint64_t output_tokens,
                                             std::optional<double> ttft_ms,
                                             std::optional<double> tpot_ms,
                                             uint64_t now_monotonic_ms);

  PlacementObservationStatus snapshot(
      const std::string& model_revision,
      uint64_t now_monotonic_ms,
      const PlacementObservationExternalInputs& external,
      PlacementObservation* observation);

  bool valid() const;
  size_t model_count() const;

 private:
  struct Bucket {
    bool initialized = false;
    uint64_t epoch = 0;
    uint64_t accepted = 0;
    uint64_t rejected = 0;
    uint64_t terminal = 0;
    uint64_t prompt_tokens = 0;
    uint64_t requested_output_tokens = 0;
    uint64_t output_tokens = 0;
    bool overflowed = false;
    bool latency_truncated = false;
    std::vector<double> ttft_ms;
    std::vector<double> tpot_ms;
  };

  struct ModelWindow {
    std::vector<Bucket> buckets;
    uint64_t latest_epoch = 0;
    uint64_t next_generation = 1;
  };

  PlacementObservationStatus get_bucket_locked(
      const std::string& model_revision,
      uint64_t now_monotonic_ms,
      ModelWindow** window,
      Bucket** bucket);

  PlacementObservationCollectorConfig config_;
  bool valid_ = false;
  mutable std::mutex mutex_;
  std::map<std::string, ModelWindow> models_;
};

bool valid_placement_observation_collector_config(
    const PlacementObservationCollectorConfig& config);

const char* placement_observation_status_name(
    PlacementObservationStatus status);

}  // namespace xllm_service::placement
