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

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace xllm_service::observability {
namespace {

class FakeMonotonicClock final : public MonotonicClock {
 public:
  explicit FakeMonotonicClock(int64_t now_ns) : now_ns_(now_ns) {}

  int64_t now_ns() const override {
    return now_ns_.load(std::memory_order_relaxed);
  }
  void set(int64_t now_ns) { now_ns_.store(now_ns, std::memory_order_relaxed); }
  void advance(int64_t delta_ns) {
    now_ns_.fetch_add(delta_ns, std::memory_order_relaxed);
  }

 private:
  std::atomic<int64_t> now_ns_;
};

xllm::proto::RequestEvent make_event() {
  xllm::proto::RequestEvent event;
  event.set_schema_version(kRequestEventSchemaVersion);
  event.mutable_correlation()->set_global_request_id("global-1");
  event.mutable_correlation()->set_trace_id("trace-1");
  event.mutable_correlation()->set_request_uid(
      "01234567-89ab-7cde-bf01-23456789abcd");
  event.mutable_correlation()->set_attempt_seq(0);
  event.mutable_correlation()->set_global_request_id_source(
      xllm::proto::CORRELATION_ID_SOURCE_UPSTREAM);
  event.mutable_correlation()->set_trace_id_source(
      xllm::proto::CORRELATION_ID_SOURCE_UPSTREAM);
  event.set_event_seq(0);
  event.set_event_type(xllm::proto::REQUEST_EVENT_TYPE_INGRESS);
  event.set_owner_role(xllm::proto::EVENT_OWNER_ROLE_SERVICE);
  event.set_owner_incarnation_id("service-incarnation-1");
  event.set_result(xllm::proto::EVENT_RESULT_SUCCEEDED);
  event.set_error_stage(xllm::proto::ERROR_STAGE_NONE);
  event.set_reason(xllm::proto::EVENT_REASON_NONE);
  event.set_build_id("service-build-1");
  event.set_stage_duration_validity(
      xllm::proto::METRIC_VALIDITY_NOT_APPLICABLE);
  event.set_stage_duration_invalid_reason(
      xllm::proto::INVALID_METRIC_REASON_NONE);
  return event;
}

xllm::proto::RequestEvent make_admission_event() {
  xllm::proto::RequestEvent event = make_event();
  event.set_owner_role(xllm::proto::EVENT_OWNER_ROLE_DECODE);
  event.set_owner_incarnation_id("decode-incarnation-1");
  event.set_target_engine_uid("decode-1");
  event.set_target_incarnation_id("decode-incarnation-1");
  return event;
}

TEST(RequestEventRecorderTest, RecordsAndDrainsWithoutExceedingCapacity) {
  FakeMonotonicClock clock(100);
  RequestEventRecorder recorder(/*capacity=*/2, clock);
  xllm::proto::RequestEvent first = make_event();
  xllm::proto::RequestEvent second = make_event();
  second.set_event_seq(1);
  xllm::proto::RequestEvent third = make_event();
  third.set_event_seq(2);

  EXPECT_EQ(recorder.record(first), RecordStatus::kRecorded);
  EXPECT_EQ(recorder.record(second), RecordStatus::kRecorded);
  EXPECT_EQ(recorder.record(third), RecordStatus::kDroppedCapacity);
  EXPECT_EQ(recorder.size(), 2u);
  RecorderStats stats = recorder.stats();
  EXPECT_EQ(stats.recorded, 2u);
  EXPECT_EQ(stats.dropped_capacity, 1u);

  auto drained = recorder.drain(1);
  ASSERT_EQ(drained.size(), 1u);
  EXPECT_EQ(drained[0].event_seq(), 0u);
  EXPECT_EQ(recorder.size(), 1u);
  EXPECT_EQ(recorder.record(third), RecordStatus::kRecorded);

  drained = recorder.drain(2);
  ASSERT_EQ(drained.size(), 2u);
  EXPECT_EQ(drained[0].event_seq(), 1u);
  EXPECT_EQ(drained[1].event_seq(), 2u);
}

TEST(RequestEventRecorderTest, ZeroCapacityDropsWithoutAccessingRing) {
  FakeMonotonicClock clock(100);
  RequestEventRecorder recorder(/*capacity=*/0, clock);

  EXPECT_EQ(recorder.record(make_event()), RecordStatus::kDroppedCapacity);
  EXPECT_EQ(recorder.size(), 0u);
  EXPECT_TRUE(recorder.drain(1).empty());
  EXPECT_EQ(recorder.stats().dropped_capacity, 1u);
}

TEST(RequestEventRecorderTest, RejectsMissingCorrelationAndInvalidMetric) {
  FakeMonotonicClock clock(100);
  RequestEventRecorder recorder(/*capacity=*/4, clock);
  xllm::proto::RequestEvent event = make_event();
  event.mutable_correlation()->clear_attempt_seq();
  EXPECT_EQ(recorder.record(event), RecordStatus::kInvalid);

  event = make_event();
  event.set_event_type(xllm::proto::REQUEST_EVENT_TYPE_METRIC_SAMPLE);
  auto* metric = event.mutable_metric();
  metric->set_kind(xllm::proto::REQUEST_METRIC_KIND_SERVER_TTFT);
  metric->set_validity(xllm::proto::METRIC_VALIDITY_VALID);
  metric->set_measurement_boundary("service_response_write");
  EXPECT_EQ(recorder.record(event), RecordStatus::kInvalid);

  event = make_event();
  event.set_owner_incarnation_id(std::string(257, 'x'));
  EXPECT_EQ(recorder.record(event), RecordStatus::kInvalid);

  event = make_event();
  event.mutable_correlation()->set_trace_id("invalid trace id");
  EXPECT_EQ(recorder.record(event), RecordStatus::kInvalid);

  event = make_event();
  event.set_event_type(xllm::proto::REQUEST_EVENT_TYPE_METRIC_SAMPLE);
  metric = event.mutable_metric();
  metric->set_kind(xllm::proto::REQUEST_METRIC_KIND_SERVER_TPOT);
  metric->set_validity(xllm::proto::METRIC_VALIDITY_VALID);
  metric->set_invalid_reason(xllm::proto::INVALID_METRIC_REASON_NONE);
  metric->set_duration_ns(0);
  metric->set_measurement_boundary("service_response_write");
  EXPECT_EQ(recorder.record(event), RecordStatus::kInvalid);
  EXPECT_EQ(recorder.stats().invalid_events, 5u);
}

TEST(RequestEventRecorderTest, ConcurrentProducersRemainBounded) {
  FakeMonotonicClock clock(100);
  constexpr size_t kCapacity = 64;
  constexpr size_t kProducerCount = 8;
  constexpr size_t kEventsPerProducer = 100;
  RequestEventRecorder recorder(kCapacity, clock);
  std::vector<std::thread> producers;
  for (size_t producer = 0; producer < kProducerCount; ++producer) {
    producers.emplace_back([producer, &recorder]() {
      for (size_t i = 0; i < kEventsPerProducer; ++i) {
        xllm::proto::RequestEvent event = make_event();
        event.set_event_seq(producer * kEventsPerProducer + i);
        recorder.record(std::move(event));
      }
    });
  }
  for (std::thread& producer : producers) {
    producer.join();
  }

  const RecorderStats stats = recorder.stats();
  EXPECT_EQ(recorder.size(), kCapacity);
  EXPECT_EQ(stats.recorded, kCapacity);
  EXPECT_EQ(stats.recorded + stats.dropped_capacity + stats.dropped_contention,
            kProducerCount * kEventsPerProducer);
}

TEST(RequestEventRecorderTest, AdmissionAttemptEmitsExactlyOneTerminal) {
  FakeMonotonicClock clock(100);
  RequestEventRecorder recorder(/*capacity=*/4, clock);
  auto attempt = recorder.begin_admission(make_admission_event());
  clock.advance(50);
  EXPECT_EQ(attempt.terminal(xllm::proto::EVENT_RESULT_ACCEPTED,
                             xllm::proto::ERROR_STAGE_NONE,
                             xllm::proto::EVENT_REASON_NONE),
            RecordStatus::kRecorded);
  EXPECT_EQ(attempt.terminal(xllm::proto::EVENT_RESULT_ACCEPTED,
                             xllm::proto::ERROR_STAGE_NONE,
                             xllm::proto::EVENT_REASON_NONE),
            RecordStatus::kDuplicateTerminal);

  auto events = recorder.drain(4);
  ASSERT_EQ(events.size(), 2u);
  EXPECT_EQ(events[0].result(), xllm::proto::EVENT_RESULT_STARTED);
  EXPECT_EQ(events[1].result(), xllm::proto::EVENT_RESULT_ACCEPTED);
  EXPECT_EQ(events[1].stage_duration_ns(), 50u);
  EXPECT_EQ(events[1].stage_duration_validity(),
            xllm::proto::METRIC_VALIDITY_VALID);
  EXPECT_EQ(recorder.stats().admission_terminals_generated, 1u);
  EXPECT_EQ(recorder.stats().duplicate_terminal_calls, 1u);
}

TEST(RequestEventRecorderTest, InvalidAdmissionTerminalDoesNotSealAttempt) {
  FakeMonotonicClock clock(100);
  RequestEventRecorder recorder(/*capacity=*/4, clock);
  auto attempt = recorder.begin_admission(make_admission_event());

  EXPECT_EQ(attempt.terminal(xllm::proto::EVENT_RESULT_ACCEPTED,
                             xllm::proto::ERROR_STAGE_D_ADMISSION,
                             xllm::proto::EVENT_REASON_CAPACITY_EXHAUSTED),
            RecordStatus::kInvalid);
  clock.advance(10);
  EXPECT_EQ(attempt.terminal(xllm::proto::EVENT_RESULT_ACCEPTED,
                             xllm::proto::ERROR_STAGE_NONE,
                             xllm::proto::EVENT_REASON_NONE),
            RecordStatus::kRecorded);

  auto events = recorder.drain(4);
  ASSERT_EQ(events.size(), 2u);
  EXPECT_EQ(events[1].result(), xllm::proto::EVENT_RESULT_ACCEPTED);
  EXPECT_EQ(events[1].stage_duration_ns(), 10u);
  EXPECT_EQ(recorder.stats().admission_terminals_generated, 1u);
}

TEST(RequestEventRecorderTest, AbandonedAdmissionGetsFailedTerminal) {
  FakeMonotonicClock clock(100);
  RequestEventRecorder recorder(/*capacity=*/4, clock);
  {
    auto attempt = recorder.begin_admission(make_admission_event());
    clock.advance(25);
  }

  auto events = recorder.drain(4);
  ASSERT_EQ(events.size(), 2u);
  EXPECT_EQ(events[1].result(), xllm::proto::EVENT_RESULT_FAILED);
  EXPECT_EQ(events[1].error_stage(), xllm::proto::ERROR_STAGE_D_ADMISSION);
  EXPECT_EQ(events[1].reason(), xllm::proto::EVENT_REASON_MISSING_TERMINAL);
  EXPECT_EQ(events[1].stage_duration_ns(), 25u);
}

TEST(RequestEventRecorderTest, ClockRegressionMarksDurationInvalid) {
  FakeMonotonicClock clock(100);
  RequestEventRecorder recorder(/*capacity=*/4, clock);
  auto attempt = recorder.begin_admission(make_admission_event());
  clock.set(90);
  EXPECT_EQ(attempt.terminal(xllm::proto::EVENT_RESULT_REJECTED,
                             xllm::proto::ERROR_STAGE_D_ADMISSION,
                             xllm::proto::EVENT_REASON_CAPACITY_EXHAUSTED),
            RecordStatus::kRecorded);

  auto events = recorder.drain(4);
  ASSERT_EQ(events.size(), 2u);
  EXPECT_FALSE(events[1].has_stage_duration_ns());
  EXPECT_EQ(events[1].stage_duration_validity(),
            xllm::proto::METRIC_VALIDITY_INVALID);
  EXPECT_EQ(events[1].stage_duration_invalid_reason(),
            xllm::proto::INVALID_METRIC_REASON_CLOCK_REGRESSION);
  EXPECT_EQ(recorder.stats().invalid_metric_samples, 1u);
}

TEST(RequestLatencyTrackerTest, ComputesLocalTtftTpotE2eAndItl) {
  FakeMonotonicClock clock(100);
  RequestLatencyTracker tracker("service_response_write", clock);
  clock.set(150);
  EXPECT_FALSE(tracker.on_token().has_value());
  clock.set(170);
  auto itl = tracker.on_token();
  ASSERT_TRUE(itl.has_value());
  EXPECT_EQ(itl->validity(), xllm::proto::METRIC_VALIDITY_VALID);
  EXPECT_EQ(itl->duration_ns(), 20u);
  clock.set(250);
  LatencyReport report = tracker.finish(/*output_tokens=*/3);
  EXPECT_EQ(report.ttft.duration_ns(), 50u);
  EXPECT_EQ(report.tpot.duration_ns(), 50u);
  EXPECT_EQ(report.e2e.duration_ns(), 150u);
}

TEST(RequestLatencyTrackerTest, ZeroTpotIsInvalidNotAValidZeroSample) {
  FakeMonotonicClock clock(100);
  RequestLatencyTracker tracker("service_response_write", clock);
  clock.set(150);
  tracker.on_token();
  LatencyReport report = tracker.finish(/*output_tokens=*/2);
  EXPECT_EQ(report.tpot.validity(), xllm::proto::METRIC_VALIDITY_INVALID);
  EXPECT_EQ(report.tpot.invalid_reason(),
            xllm::proto::INVALID_METRIC_REASON_ZERO_TPOT);
  EXPECT_FALSE(report.tpot.has_duration_ns());
}

TEST(RequestLatencyTrackerTest, OneTokenTpotIsExplicitlyNotApplicable) {
  FakeMonotonicClock clock(100);
  RequestLatencyTracker tracker("service_response_write", clock);
  clock.set(150);
  tracker.on_token();
  clock.set(175);
  LatencyReport report = tracker.finish(/*output_tokens=*/1);
  EXPECT_EQ(report.tpot.validity(),
            xllm::proto::METRIC_VALIDITY_NOT_APPLICABLE);
  EXPECT_EQ(report.tpot.invalid_reason(),
            xllm::proto::INVALID_METRIC_REASON_INSUFFICIENT_OUTPUT_TOKENS);
}

TEST(RequestLatencyTrackerTest, MissingFirstTokenInvalidatesTtftAndTpot) {
  FakeMonotonicClock clock(100);
  RequestLatencyTracker tracker("service_response_write", clock);
  clock.set(200);
  LatencyReport report = tracker.finish(/*output_tokens=*/2);
  EXPECT_EQ(report.ttft.validity(), xllm::proto::METRIC_VALIDITY_INVALID);
  EXPECT_EQ(report.tpot.validity(), xllm::proto::METRIC_VALIDITY_INVALID);
  EXPECT_EQ(report.e2e.validity(), xllm::proto::METRIC_VALIDITY_VALID);
  EXPECT_EQ(report.e2e.duration_ns(), 100u);
}

TEST(RequestLatencyTrackerTest, ClockRegressionNeverProducesNegativeSample) {
  FakeMonotonicClock clock(100);
  RequestLatencyTracker tracker("service_response_write", clock);
  clock.set(90);
  tracker.on_token();
  clock.set(80);
  auto itl = tracker.on_token();
  ASSERT_TRUE(itl.has_value());
  EXPECT_EQ(itl->validity(), xllm::proto::METRIC_VALIDITY_INVALID);
  EXPECT_FALSE(itl->has_duration_ns());

  LatencyReport report = tracker.finish(/*output_tokens=*/2);
  EXPECT_EQ(report.ttft.validity(), xllm::proto::METRIC_VALIDITY_INVALID);
  EXPECT_EQ(report.tpot.validity(), xllm::proto::METRIC_VALIDITY_INVALID);
  EXPECT_EQ(report.e2e.validity(), xllm::proto::METRIC_VALIDITY_INVALID);
}

}  // namespace
}  // namespace xllm_service::observability
