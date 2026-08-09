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

#include "observability/request_event_recorder.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "common/xllm/uuid.h"
#include "nlohmann/json.hpp"

namespace xllm_service::observability {
namespace {

constexpr size_t kMaxIdentityLength = 256;
constexpr size_t kMaxMeasurementBoundaryLength = 64;

bool valid_identity(const std::string& value) {
  return !value.empty() && value.size() <= kMaxIdentityLength;
}

bool valid_correlation_id(const std::string& value) {
  return valid_identity(value) &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return character >= 0x21 && character <= 0x7e;
         });
}

bool is_failure_result(xllm::proto::EventResult result) {
  return result == xllm::proto::EVENT_RESULT_REJECTED ||
         result == xllm::proto::EVENT_RESULT_FAILED ||
         result == xllm::proto::EVENT_RESULT_CANCELLED ||
         result == xllm::proto::EVENT_RESULT_DEADLINE_EXCEEDED;
}

bool is_admission_terminal(xllm::proto::EventResult result) {
  return result == xllm::proto::EVENT_RESULT_ACCEPTED ||
         is_failure_result(result);
}

bool valid_admission_terminal(xllm::proto::EventResult result,
                              xllm::proto::ErrorStage error_stage,
                              xllm::proto::EventReason reason) {
  if (result == xllm::proto::EVENT_RESULT_ACCEPTED) {
    return error_stage == xllm::proto::ERROR_STAGE_NONE &&
           reason == xllm::proto::EVENT_REASON_NONE;
  }
  return is_failure_result(result) &&
         error_stage != xllm::proto::ERROR_STAGE_NONE &&
         reason != xllm::proto::EVENT_REASON_NONE;
}

bool valid_metric(const xllm::proto::RequestMetric& metric) {
  if (!xllm::proto::RequestMetricKind_IsValid(metric.kind()) ||
      metric.kind() == xllm::proto::REQUEST_METRIC_KIND_UNSPECIFIED ||
      !xllm::proto::MetricValidity_IsValid(metric.validity()) ||
      metric.validity() == xllm::proto::METRIC_VALIDITY_UNSPECIFIED ||
      !xllm::proto::InvalidMetricReason_IsValid(metric.invalid_reason()) ||
      metric.measurement_boundary().empty() ||
      metric.measurement_boundary().size() > kMaxMeasurementBoundaryLength) {
    return false;
  }
  if (metric.validity() == xllm::proto::METRIC_VALIDITY_VALID) {
    return metric.has_duration_ns() &&
           metric.invalid_reason() == xllm::proto::INVALID_METRIC_REASON_NONE &&
           (metric.kind() != xllm::proto::REQUEST_METRIC_KIND_SERVER_TPOT ||
            metric.duration_ns() != 0);
  }
  if (metric.validity() == xllm::proto::METRIC_VALIDITY_INVALID) {
    return !metric.has_duration_ns() &&
           metric.invalid_reason() != xllm::proto::INVALID_METRIC_REASON_NONE;
  }
  return !metric.has_duration_ns() &&
         metric.invalid_reason() != xllm::proto::INVALID_METRIC_REASON_NONE;
}

bool valid_stage_duration(const xllm::proto::RequestEvent& event) {
  if (!xllm::proto::MetricValidity_IsValid(event.stage_duration_validity()) ||
      event.stage_duration_validity() ==
          xllm::proto::METRIC_VALIDITY_UNSPECIFIED ||
      !xllm::proto::InvalidMetricReason_IsValid(
          event.stage_duration_invalid_reason())) {
    return false;
  }
  if (event.stage_duration_validity() == xllm::proto::METRIC_VALIDITY_VALID) {
    return event.has_stage_duration_ns() &&
           event.stage_duration_invalid_reason() ==
               xllm::proto::INVALID_METRIC_REASON_NONE;
  }
  if (event.stage_duration_validity() == xllm::proto::METRIC_VALIDITY_INVALID) {
    return !event.has_stage_duration_ns() &&
           event.stage_duration_invalid_reason() !=
               xllm::proto::INVALID_METRIC_REASON_NONE;
  }
  return !event.has_stage_duration_ns() &&
         event.stage_duration_invalid_reason() ==
             xllm::proto::INVALID_METRIC_REASON_NONE;
}

bool valid_runtime_profile(const xllm::proto::RuntimeProfileIdentity& profile) {
  return xllm::proto::ProviderId_IsValid(profile.provider_id()) &&
         profile.provider_id() != xllm::proto::PROVIDER_ID_UNSPECIFIED &&
         valid_identity(profile.runtime_version()) &&
         valid_identity(profile.plugin_version()) &&
         valid_identity(profile.hardware_runtime_version()) &&
         valid_identity(profile.profile_digest()) &&
         valid_identity(profile.model_revision()) &&
         xllm::proto::ExecutionMode_IsValid(profile.execution_mode()) &&
         profile.execution_mode() != xllm::proto::EXECUTION_MODE_UNSPECIFIED;
}

bool valid_event(const xllm::proto::RequestEvent& event) {
  if (event.schema_version() != kRequestEventSchemaVersion ||
      !event.has_correlation() ||
      !valid_correlation_id(event.correlation().global_request_id()) ||
      !valid_correlation_id(event.correlation().trace_id()) ||
      !llm::is_uuid_v7(event.correlation().request_uid()) ||
      !event.correlation().has_attempt_seq() ||
      !xllm::proto::CorrelationIdSource_IsValid(
          event.correlation().global_request_id_source()) ||
      event.correlation().global_request_id_source() ==
          xllm::proto::CORRELATION_ID_SOURCE_UNSPECIFIED ||
      !xllm::proto::CorrelationIdSource_IsValid(
          event.correlation().trace_id_source()) ||
      event.correlation().trace_id_source() ==
          xllm::proto::CORRELATION_ID_SOURCE_UNSPECIFIED ||
      !event.has_event_seq() ||
      !xllm::proto::RequestEventType_IsValid(event.event_type()) ||
      event.event_type() == xllm::proto::REQUEST_EVENT_TYPE_UNSPECIFIED ||
      !xllm::proto::EventOwnerRole_IsValid(event.owner_role()) ||
      event.owner_role() == xllm::proto::EVENT_OWNER_ROLE_UNSPECIFIED ||
      !valid_identity(event.owner_incarnation_id()) ||
      !xllm::proto::EventResult_IsValid(event.result()) ||
      event.result() == xllm::proto::EVENT_RESULT_UNSPECIFIED ||
      !xllm::proto::ErrorStage_IsValid(event.error_stage()) ||
      !xllm::proto::EventReason_IsValid(event.reason()) ||
      !valid_identity(event.build_id()) || !valid_stage_duration(event) ||
      (!event.target_engine_uid().empty() &&
       !valid_identity(event.target_engine_uid())) ||
      (!event.target_incarnation_id().empty() &&
       !valid_identity(event.target_incarnation_id()))) {
    return false;
  }
  if (event.has_runtime_profile() &&
      !valid_runtime_profile(event.runtime_profile())) {
    return false;
  }
  if (is_failure_result(event.result())) {
    if (event.error_stage() == xllm::proto::ERROR_STAGE_NONE ||
        event.reason() == xllm::proto::EVENT_REASON_NONE) {
      return false;
    }
  } else if (event.error_stage() != xllm::proto::ERROR_STAGE_NONE ||
             event.reason() != xllm::proto::EVENT_REASON_NONE) {
    return false;
  }
  if (event.event_type() == xllm::proto::REQUEST_EVENT_TYPE_D_ADMISSION &&
      (event.target_engine_uid().empty() ||
       event.target_incarnation_id().empty() ||
       (event.result() != xllm::proto::EVENT_RESULT_STARTED &&
        !is_admission_terminal(event.result())))) {
    return false;
  }
  if (event.event_type() == xllm::proto::REQUEST_EVENT_TYPE_METRIC_SAMPLE) {
    return event.has_metric() && valid_metric(event.metric());
  }
  return !event.has_metric();
}

xllm::proto::RequestMetric new_metric(xllm::proto::RequestMetricKind kind,
                                      const std::string& measurement_boundary,
                                      uint64_t output_tokens) {
  xllm::proto::RequestMetric metric;
  metric.set_kind(kind);
  metric.set_measurement_boundary(measurement_boundary);
  metric.set_output_tokens(output_tokens);
  return metric;
}

void set_valid_duration(xllm::proto::RequestMetric* metric,
                        int64_t duration_ns) {
  metric->set_validity(xllm::proto::METRIC_VALIDITY_VALID);
  metric->set_invalid_reason(xllm::proto::INVALID_METRIC_REASON_NONE);
  metric->set_duration_ns(static_cast<uint64_t>(duration_ns));
}

void set_invalid(xllm::proto::RequestMetric* metric,
                 xllm::proto::InvalidMetricReason reason) {
  metric->set_validity(xllm::proto::METRIC_VALIDITY_INVALID);
  metric->set_invalid_reason(reason);
  metric->clear_duration_ns();
}

}  // namespace

int64_t SteadyMonotonicClock::now_ns() const {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

const MonotonicClock& default_monotonic_clock() {
  static const SteadyMonotonicClock clock;
  return clock;
}

const char* record_status_name(RecordStatus status) {
  switch (status) {
    case RecordStatus::kRecorded:
      return "recorded";
    case RecordStatus::kInvalid:
      return "invalid";
    case RecordStatus::kDroppedCapacity:
      return "dropped_capacity";
    case RecordStatus::kDroppedContention:
      return "dropped_contention";
    case RecordStatus::kDuplicateTerminal:
      return "duplicate_terminal";
  }
  return "unknown";
}

std::string format_request_event_log(const xllm::proto::RequestEvent& event) {
  nlohmann::json output{
      {"schema_version", event.schema_version()},
      {"event_seq",
       event.has_event_seq() ? nlohmann::json(event.event_seq())
                             : nlohmann::json(nullptr)},
      {"event_type", xllm::proto::RequestEventType_Name(event.event_type())},
      {"owner_role", xllm::proto::EventOwnerRole_Name(event.owner_role())},
      {"owner_incarnation_id", event.owner_incarnation_id()},
      {"result", xllm::proto::EventResult_Name(event.result())},
      {"error_stage", xllm::proto::ErrorStage_Name(event.error_stage())},
      {"reason", xllm::proto::EventReason_Name(event.reason())},
      {"build_id", event.build_id()},
  };
  if (event.has_correlation()) {
    const xllm::proto::RequestCorrelation& correlation = event.correlation();
    output["global_request_id"] = correlation.global_request_id();
    output["trace_id"] = correlation.trace_id();
    output["request_uid"] = correlation.request_uid();
    output["attempt_seq"] = correlation.has_attempt_seq()
                                ? nlohmann::json(correlation.attempt_seq())
                                : nlohmann::json(nullptr);
  }
  output["stage_duration_ns"] = event.has_stage_duration_ns()
                                    ? nlohmann::json(event.stage_duration_ns())
                                    : nlohmann::json(nullptr);
  output["stage_duration_validity"] =
      xllm::proto::MetricValidity_Name(event.stage_duration_validity());
  output["stage_duration_invalid_reason"] =
      xllm::proto::InvalidMetricReason_Name(
          event.stage_duration_invalid_reason());
  if (!event.target_engine_uid().empty()) {
    output["target_engine_uid"] = event.target_engine_uid();
    output["target_incarnation_id"] = event.target_incarnation_id();
  }
  if (event.has_runtime_profile()) {
    const xllm::proto::RuntimeProfileIdentity& profile =
        event.runtime_profile();
    output["provider_id"] = xllm::proto::ProviderId_Name(profile.provider_id());
    output["runtime_version"] = profile.runtime_version();
    output["plugin_version"] = profile.plugin_version();
    output["hardware_runtime_version"] = profile.hardware_runtime_version();
    output["profile_digest"] = profile.profile_digest();
    output["model_revision"] = profile.model_revision();
    output["execution_mode"] =
        xllm::proto::ExecutionMode_Name(profile.execution_mode());
  }
  const xllm::proto::WorkloadShape& workload = event.workload();
  if (workload.has_prompt_tokens()) {
    output["prompt_tokens"] = workload.prompt_tokens();
  }
  if (workload.has_output_tokens()) {
    output["output_tokens"] = workload.output_tokens();
  }
  if (workload.has_effective_max_new_tokens()) {
    output["effective_max_new_tokens"] = workload.effective_max_new_tokens();
  }
  if (workload.has_kv_blocks()) {
    output["kv_blocks"] = workload.kv_blocks();
  }
  if (workload.has_batch_size()) {
    output["batch_size"] = workload.batch_size();
  }
  if (workload.has_predicted_prefill_hit_tokens()) {
    output["predicted_prefill_hit_tokens"] =
        workload.predicted_prefill_hit_tokens();
  }
  if (workload.has_predicted_decode_hit_tokens()) {
    output["predicted_decode_hit_tokens"] =
        workload.predicted_decode_hit_tokens();
  }
  if (workload.has_predicted_transfer_bytes()) {
    output["predicted_transfer_bytes"] = workload.predicted_transfer_bytes();
  }
  if (workload.has_actual_prefill_hit_tokens()) {
    output["actual_prefill_hit_tokens"] = workload.actual_prefill_hit_tokens();
  }
  if (workload.has_actual_decode_hit_tokens()) {
    output["actual_decode_hit_tokens"] = workload.actual_decode_hit_tokens();
  }
  if (workload.has_skipped_transfer_bytes()) {
    output["skipped_transfer_bytes"] = workload.skipped_transfer_bytes();
  }
  if (workload.has_shadow_prefill_host_hit_tokens_ub()) {
    output["shadow_prefill_host_hit_tokens_ub"] =
        workload.shadow_prefill_host_hit_tokens_ub();
  }
  if (workload.has_shadow_prefill_ssd_hit_tokens_ub()) {
    output["shadow_prefill_ssd_hit_tokens_ub"] =
        workload.shadow_prefill_ssd_hit_tokens_ub();
  }
  if (workload.has_shadow_prefill_store_hit_tokens_ub()) {
    output["shadow_prefill_store_hit_tokens_ub"] =
        workload.shadow_prefill_store_hit_tokens_ub();
  }
  if (workload.has_shadow_decode_host_hit_tokens_ub()) {
    output["shadow_decode_host_hit_tokens_ub"] =
        workload.shadow_decode_host_hit_tokens_ub();
  }
  if (workload.has_shadow_decode_ssd_hit_tokens_ub()) {
    output["shadow_decode_ssd_hit_tokens_ub"] =
        workload.shadow_decode_ssd_hit_tokens_ub();
  }
  if (workload.has_shadow_decode_store_hit_tokens_ub()) {
    output["shadow_decode_store_hit_tokens_ub"] =
        workload.shadow_decode_store_hit_tokens_ub();
  }
  if (event.has_metric()) {
    const xllm::proto::RequestMetric& metric = event.metric();
    output["metric_kind"] = xllm::proto::RequestMetricKind_Name(metric.kind());
    output["metric_validity"] =
        xllm::proto::MetricValidity_Name(metric.validity());
    output["metric_invalid_reason"] =
        xllm::proto::InvalidMetricReason_Name(metric.invalid_reason());
    output["metric_duration_ns"] = metric.has_duration_ns()
                                       ? nlohmann::json(metric.duration_ns())
                                       : nlohmann::json(nullptr);
    output["measurement_boundary"] = metric.measurement_boundary();
    if (metric.has_output_tokens()) {
      output["metric_output_tokens"] = metric.output_tokens();
    }
  }
  return output.dump();
}

RequestEventRecorder::RequestEventRecorder(size_t capacity,
                                           const MonotonicClock& clock)
    : capacity_(capacity), clock_(clock), ring_(capacity) {}

RecordStatus RequestEventRecorder::record(xllm::proto::RequestEvent event) {
  if (!valid_event(event)) {
    invalid_events_.fetch_add(1, std::memory_order_relaxed);
    return RecordStatus::kInvalid;
  }
  if ((event.has_metric() &&
       event.metric().validity() == xllm::proto::METRIC_VALIDITY_INVALID) ||
      event.stage_duration_validity() == xllm::proto::METRIC_VALIDITY_INVALID) {
    invalid_metric_samples_.fetch_add(1, std::memory_order_relaxed);
  }

  std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock()) {
    dropped_contention_.fetch_add(1, std::memory_order_relaxed);
    return RecordStatus::kDroppedContention;
  }
  if (size_ >= capacity_) {
    dropped_capacity_.fetch_add(1, std::memory_order_relaxed);
    return RecordStatus::kDroppedCapacity;
  }
  ring_[(head_ + size_) % capacity_] = std::move(event);
  ++size_;
  recorded_.fetch_add(1, std::memory_order_relaxed);
  return RecordStatus::kRecorded;
}

std::vector<xllm::proto::RequestEvent> RequestEventRecorder::drain(
    size_t max_events) {
  std::vector<xllm::proto::RequestEvent> events;
  std::lock_guard<std::mutex> lock(mutex_);
  const size_t count = std::min(max_events, size_);
  events.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    events.push_back(std::move(ring_[head_]));
    ring_[head_].Clear();
    head_ = (head_ + 1) % capacity_;
    --size_;
  }
  return events;
}

size_t RequestEventRecorder::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return size_;
}

RecorderStats RequestEventRecorder::stats() const {
  RecorderStats result;
  result.recorded = recorded_.load(std::memory_order_relaxed);
  result.dropped_capacity = dropped_capacity_.load(std::memory_order_relaxed);
  result.dropped_contention =
      dropped_contention_.load(std::memory_order_relaxed);
  result.invalid_events = invalid_events_.load(std::memory_order_relaxed);
  result.invalid_metric_samples =
      invalid_metric_samples_.load(std::memory_order_relaxed);
  result.admission_terminals_generated =
      admission_terminals_generated_.load(std::memory_order_relaxed);
  result.duplicate_terminal_calls =
      duplicate_terminal_calls_.load(std::memory_order_relaxed);
  return result;
}

AdmissionAttempt RequestEventRecorder::begin_admission(
    xllm::proto::RequestEvent base_event) {
  return AdmissionAttempt(this, std::move(base_event));
}

void RequestEventRecorder::note_admission_terminal() {
  admission_terminals_generated_.fetch_add(1, std::memory_order_relaxed);
}

void RequestEventRecorder::note_duplicate_terminal() {
  duplicate_terminal_calls_.fetch_add(1, std::memory_order_relaxed);
}

AdmissionAttempt::AdmissionAttempt(RequestEventRecorder* recorder,
                                   xllm::proto::RequestEvent base_event)
    : recorder_(recorder),
      base_event_(std::move(base_event)),
      started_ns_(recorder_->clock_.now_ns()) {
  base_event_.set_schema_version(kRequestEventSchemaVersion);
  base_event_.set_event_type(xllm::proto::REQUEST_EVENT_TYPE_D_ADMISSION);
  if (!base_event_.has_event_seq()) {
    base_event_.set_event_seq(0);
  }
  base_event_.set_result(xllm::proto::EVENT_RESULT_STARTED);
  base_event_.set_error_stage(xllm::proto::ERROR_STAGE_NONE);
  base_event_.set_reason(xllm::proto::EVENT_REASON_NONE);
  base_event_.clear_stage_duration_ns();
  base_event_.set_stage_duration_validity(
      xllm::proto::METRIC_VALIDITY_NOT_APPLICABLE);
  base_event_.set_stage_duration_invalid_reason(
      xllm::proto::INVALID_METRIC_REASON_NONE);
  recorder_->record(base_event_);
}

AdmissionAttempt::AdmissionAttempt(AdmissionAttempt&& other) noexcept
    : recorder_(other.recorder_),
      base_event_(std::move(other.base_event_)),
      started_ns_(other.started_ns_),
      terminal_emitted_(other.terminal_emitted_) {
  other.recorder_ = nullptr;
  other.terminal_emitted_ = true;
}

AdmissionAttempt& AdmissionAttempt::operator=(
    AdmissionAttempt&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  finish_abandoned_attempt();
  recorder_ = other.recorder_;
  base_event_ = std::move(other.base_event_);
  started_ns_ = other.started_ns_;
  terminal_emitted_ = other.terminal_emitted_;
  other.recorder_ = nullptr;
  other.terminal_emitted_ = true;
  return *this;
}

AdmissionAttempt::~AdmissionAttempt() { finish_abandoned_attempt(); }

RecordStatus AdmissionAttempt::terminal(xllm::proto::EventResult result,
                                        xllm::proto::ErrorStage error_stage,
                                        xllm::proto::EventReason reason) {
  if (terminal_emitted_) {
    if (recorder_ != nullptr) {
      recorder_->note_duplicate_terminal();
    }
    return RecordStatus::kDuplicateTerminal;
  }
  if (!valid_admission_terminal(result, error_stage, reason)) {
    return RecordStatus::kInvalid;
  }
  return emit_terminal(result, error_stage, reason);
}

RecordStatus AdmissionAttempt::emit_terminal(
    xllm::proto::EventResult result,
    xllm::proto::ErrorStage error_stage,
    xllm::proto::EventReason reason) {
  terminal_emitted_ = true;
  recorder_->note_admission_terminal();
  xllm::proto::RequestEvent event = base_event_;
  if (event.event_seq() != std::numeric_limits<uint64_t>::max()) {
    event.set_event_seq(event.event_seq() + 1);
  }
  event.set_result(result);
  event.set_error_stage(error_stage);
  event.set_reason(reason);

  const int64_t finished_ns = recorder_->clock_.now_ns();
  if (finished_ns >= started_ns_) {
    event.set_stage_duration_ns(
        static_cast<uint64_t>(finished_ns - started_ns_));
    event.set_stage_duration_validity(xllm::proto::METRIC_VALIDITY_VALID);
    event.set_stage_duration_invalid_reason(
        xllm::proto::INVALID_METRIC_REASON_NONE);
  } else {
    event.clear_stage_duration_ns();
    event.set_stage_duration_validity(xllm::proto::METRIC_VALIDITY_INVALID);
    event.set_stage_duration_invalid_reason(
        xllm::proto::INVALID_METRIC_REASON_CLOCK_REGRESSION);
  }
  return recorder_->record(std::move(event));
}

void AdmissionAttempt::finish_abandoned_attempt() {
  if (recorder_ == nullptr || terminal_emitted_) {
    return;
  }
  emit_terminal(xllm::proto::EVENT_RESULT_FAILED,
                xllm::proto::ERROR_STAGE_D_ADMISSION,
                xllm::proto::EVENT_REASON_MISSING_TERMINAL);
}

RequestLatencyTracker::RequestLatencyTracker(std::string measurement_boundary,
                                             const MonotonicClock& clock)
    : clock_(clock),
      measurement_boundary_(std::move(measurement_boundary)),
      started_ns_(clock_.now_ns()) {}

std::optional<xllm::proto::RequestMetric> RequestLatencyTracker::on_token() {
  const int64_t now_ns = clock_.now_ns();
  if (!first_token_ns_.has_value()) {
    first_token_ns_ = now_ns;
    last_token_ns_ = now_ns;
    return std::nullopt;
  }

  xllm::proto::RequestMetric metric = new_metric(
      xllm::proto::REQUEST_METRIC_KIND_SERVER_ITL, measurement_boundary_, 0);
  if (now_ns < *last_token_ns_) {
    set_invalid(&metric, xllm::proto::INVALID_METRIC_REASON_CLOCK_REGRESSION);
  } else {
    set_valid_duration(&metric, now_ns - *last_token_ns_);
  }
  last_token_ns_ = now_ns;
  return metric;
}

LatencyReport RequestLatencyTracker::finish(uint64_t output_tokens) const {
  const int64_t finished_ns = clock_.now_ns();
  LatencyReport report;
  report.ttft = new_metric(xllm::proto::REQUEST_METRIC_KIND_SERVER_TTFT,
                           measurement_boundary_,
                           output_tokens);
  report.tpot = new_metric(xllm::proto::REQUEST_METRIC_KIND_SERVER_TPOT,
                           measurement_boundary_,
                           output_tokens);
  report.e2e = new_metric(xllm::proto::REQUEST_METRIC_KIND_SERVER_E2E,
                          measurement_boundary_,
                          output_tokens);

  if (finished_ns < started_ns_) {
    set_invalid(&report.e2e,
                xllm::proto::INVALID_METRIC_REASON_CLOCK_REGRESSION);
  } else {
    set_valid_duration(&report.e2e, finished_ns - started_ns_);
  }

  if (!first_token_ns_.has_value()) {
    set_invalid(&report.ttft,
                xllm::proto::INVALID_METRIC_REASON_MISSING_FIRST_TOKEN);
  } else if (*first_token_ns_ < started_ns_) {
    set_invalid(&report.ttft,
                xllm::proto::INVALID_METRIC_REASON_CLOCK_REGRESSION);
  } else {
    set_valid_duration(&report.ttft, *first_token_ns_ - started_ns_);
  }

  if (output_tokens < 2) {
    report.tpot.set_validity(xllm::proto::METRIC_VALIDITY_NOT_APPLICABLE);
    report.tpot.set_invalid_reason(
        xllm::proto::INVALID_METRIC_REASON_INSUFFICIENT_OUTPUT_TOKENS);
  } else if (!first_token_ns_.has_value()) {
    set_invalid(&report.tpot,
                xllm::proto::INVALID_METRIC_REASON_MISSING_FIRST_TOKEN);
  } else if (finished_ns < *first_token_ns_) {
    set_invalid(&report.tpot,
                xllm::proto::INVALID_METRIC_REASON_CLOCK_REGRESSION);
  } else {
    const uint64_t tpot_ns =
        static_cast<uint64_t>(finished_ns - *first_token_ns_) /
        (output_tokens - 1);
    if (tpot_ns == 0) {
      set_invalid(&report.tpot, xllm::proto::INVALID_METRIC_REASON_ZERO_TPOT);
    } else {
      set_valid_duration(&report.tpot, static_cast<int64_t>(tpot_ns));
    }
  }
  return report;
}

}  // namespace xllm_service::observability
