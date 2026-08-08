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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "observability.pb.h"

namespace xllm_service::observability {

inline constexpr uint32_t kRequestEventSchemaVersion = 1;

class MonotonicClock {
 public:
  virtual ~MonotonicClock() = default;
  virtual int64_t now_ns() const = 0;
};

class SteadyMonotonicClock final : public MonotonicClock {
 public:
  int64_t now_ns() const override;
};

const MonotonicClock& default_monotonic_clock();

enum class RecordStatus {
  kRecorded,
  kInvalid,
  kDroppedCapacity,
  kDroppedContention,
  kDuplicateTerminal,
};

struct RecorderStats {
  uint64_t recorded = 0;
  uint64_t dropped_capacity = 0;
  uint64_t dropped_contention = 0;
  uint64_t invalid_events = 0;
  uint64_t invalid_metric_samples = 0;
  uint64_t admission_terminals_generated = 0;
  uint64_t duplicate_terminal_calls = 0;
};

class AdmissionAttempt;

class RequestEventRecorder {
 public:
  explicit RequestEventRecorder(
      size_t capacity,
      const MonotonicClock& clock = default_monotonic_clock());

  RecordStatus record(xllm::proto::RequestEvent event);
  std::vector<xllm::proto::RequestEvent> drain(size_t max_events);
  size_t size() const;
  RecorderStats stats() const;

  AdmissionAttempt begin_admission(xllm::proto::RequestEvent base_event);

 private:
  friend class AdmissionAttempt;

  void note_admission_terminal();
  void note_duplicate_terminal();

  size_t capacity_;
  const MonotonicClock& clock_;
  mutable std::mutex mutex_;
  std::vector<xllm::proto::RequestEvent> ring_;
  size_t head_ = 0;
  size_t size_ = 0;
  std::atomic<uint64_t> recorded_{0};
  std::atomic<uint64_t> dropped_capacity_{0};
  std::atomic<uint64_t> dropped_contention_{0};
  std::atomic<uint64_t> invalid_events_{0};
  std::atomic<uint64_t> invalid_metric_samples_{0};
  std::atomic<uint64_t> admission_terminals_generated_{0};
  std::atomic<uint64_t> duplicate_terminal_calls_{0};
};

class AdmissionAttempt final {
 public:
  AdmissionAttempt(AdmissionAttempt&& other) noexcept;
  AdmissionAttempt& operator=(AdmissionAttempt&& other) noexcept;
  AdmissionAttempt(const AdmissionAttempt&) = delete;
  AdmissionAttempt& operator=(const AdmissionAttempt&) = delete;
  ~AdmissionAttempt();

  RecordStatus terminal(xllm::proto::EventResult result,
                        xllm::proto::ErrorStage error_stage,
                        xllm::proto::EventReason reason);

 private:
  friend class RequestEventRecorder;

  AdmissionAttempt(RequestEventRecorder* recorder,
                   xllm::proto::RequestEvent base_event);
  RecordStatus emit_terminal(xllm::proto::EventResult result,
                             xllm::proto::ErrorStage error_stage,
                             xllm::proto::EventReason reason);
  void finish_abandoned_attempt();

  RequestEventRecorder* recorder_ = nullptr;
  xllm::proto::RequestEvent base_event_;
  int64_t started_ns_ = 0;
  bool terminal_emitted_ = false;
};

struct LatencyReport {
  xllm::proto::RequestMetric ttft;
  xllm::proto::RequestMetric tpot;
  xllm::proto::RequestMetric e2e;
};

class RequestLatencyTracker final {
 public:
  explicit RequestLatencyTracker(
      std::string measurement_boundary,
      const MonotonicClock& clock = default_monotonic_clock());

  std::optional<xllm::proto::RequestMetric> on_token();
  LatencyReport finish(uint64_t output_tokens) const;

 private:
  const MonotonicClock& clock_;
  std::string measurement_boundary_;
  int64_t started_ns_;
  std::optional<int64_t> first_token_ns_;
  std::optional<int64_t> last_token_ns_;
};

}  // namespace xllm_service::observability
