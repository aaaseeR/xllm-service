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

#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <thread>
#include <vector>

namespace xllm_service {
namespace {

using namespace std::chrono_literals;

FlowControlConfig config() {
  return FlowControlConfig{
      .max_queued_requests = 16,
      .max_dispatched_request_contexts = 4,
      .max_queued_prompt_tokens = 1000,
      .max_queued_bytes = 100000,
      .max_queue_wait_ms = 10000,
      .max_queued_requests_per_tenant = 8,
      .max_queued_tokens_per_tenant = 800,
      .max_model_queued_requests = 8,
      .max_model_dispatched_request_contexts = 2,
      .max_model_queued_prompt_tokens = 800,
      .max_model_queued_bytes = 80000,
      .service_crash_request_budget = 20,
      .service_memory_budget_bytes = 104000,
      .dispatched_context_bytes = 1000,
      .dispatch_rate_lb_per_second = 100.0,
      .probe_round_ub_ms = 5,
      .blind_dispatch_probe_concurrency = 1,
      .starvation_dispatch_bound = 2,
      .flow_order = FlowOrder::FCFS,
  };
}

FlowControlWork work(std::string uid,
                     FlowControlQueue::TimePoint now,
                     std::string tenant = "tenant-a",
                     std::string model = "model-a",
                     int32_t priority = 2,
                     uint64_t tokens = 10,
                     uint64_t bytes = 100) {
  return FlowControlWork{
      .request_uid = std::move(uid),
      .model_pool = std::move(model),
      .tenant_id = tenant,
      .flow_id = std::move(tenant),
      .priority_band = priority,
      .prompt_tokens = tokens,
      .request_bytes = bytes,
      .deadline = now + 30s,
  };
}

TEST(FlowControlQueueTest, RejectsInvalidCrashAndMemoryBudgets) {
  FlowControlConfig invalid = config();
  invalid.service_crash_request_budget = 19;
  EXPECT_FALSE(FlowControlQueue(invalid).valid());

  invalid = config();
  invalid.service_memory_budget_bytes = 103999;
  EXPECT_FALSE(FlowControlQueue(invalid).valid());
}

TEST(FlowControlQueueTest, CapacityRejectionDoesNotPartiallyReserve) {
  FlowControlConfig limits = config();
  limits.max_queued_requests_per_tenant = 1;
  FlowControlQueue queue(limits);
  const auto now = FlowControlQueue::Clock::now();

  EXPECT_EQ(queue.admit(work("a", now), now, SaturationState::AVAILABLE).status,
            FlowControlStatus::OK);
  EXPECT_EQ(queue.admit(work("b", now), now, SaturationState::AVAILABLE).status,
            FlowControlStatus::QUEUE_CAPACITY_EXHAUSTED);
  EXPECT_EQ(queue.snapshot().queued_requests, 1);
  EXPECT_EQ(queue.snapshot().queued_prompt_tokens, 10);

  EXPECT_EQ(queue.cancel("a"), FlowControlStatus::OK);
  EXPECT_EQ(queue.admit(work("b", now), now, SaturationState::AVAILABLE).status,
            FlowControlStatus::OK);
}

TEST(FlowControlQueueTest, RejectsUnsatisfiableDeadlineFromWorkAhead) {
  FlowControlConfig limits = config();
  limits.dispatch_rate_lb_per_second = 1.0;
  FlowControlQueue queue(limits);
  const auto now = FlowControlQueue::Clock::now();

  EXPECT_EQ(
      queue.admit(work("ahead", now), now, SaturationState::SATURATED).status,
      FlowControlStatus::OK);
  FlowControlWork late = work("late", now);
  late.deadline = now + 500ms;
  EXPECT_EQ(queue.admit(late, now, SaturationState::AVAILABLE).status,
            FlowControlStatus::QUEUE_DEADLINE_UNSATISFIABLE);
  EXPECT_EQ(queue.snapshot().queued_requests, 1);
}

TEST(FlowControlQueueTest, PriorityThenTenantRoundRobinIsDeterministic) {
  FlowControlQueue queue(config());
  const auto now = FlowControlQueue::Clock::now();
  ASSERT_EQ(
      queue.admit(work("a1", now, "a"), now, SaturationState::SATURATED).status,
      FlowControlStatus::OK);
  ASSERT_EQ(
      queue.admit(work("a2", now, "a"), now, SaturationState::SATURATED).status,
      FlowControlStatus::OK);
  ASSERT_EQ(
      queue.admit(work("b1", now, "b"), now, SaturationState::SATURATED).status,
      FlowControlStatus::OK);
  ASSERT_EQ(queue
                .admit(work("high", now, "c", "model-a", 1),
                       now,
                       SaturationState::SATURATED)
                .status,
            FlowControlStatus::OK);

  auto dispatch = queue.take_next(now, SaturationState::AVAILABLE);
  ASSERT_TRUE(dispatch.work.has_value());
  EXPECT_EQ(dispatch.work->request_uid, "high");
  ASSERT_EQ(queue.complete("high"), FlowControlStatus::OK);

  dispatch = queue.take_next(now, SaturationState::AVAILABLE);
  ASSERT_TRUE(dispatch.work.has_value());
  EXPECT_EQ(dispatch.work->request_uid, "a1");
  ASSERT_EQ(queue.complete("a1"), FlowControlStatus::OK);

  dispatch = queue.take_next(now, SaturationState::AVAILABLE);
  ASSERT_TRUE(dispatch.work.has_value());
  EXPECT_EQ(dispatch.work->request_uid, "b1");
}

TEST(FlowControlQueueTest, EdfOrdersWithinOneFlow) {
  FlowControlConfig limits = config();
  limits.flow_order = FlowOrder::EDF;
  FlowControlQueue queue(limits);
  const auto now = FlowControlQueue::Clock::now();
  FlowControlWork later = work("later", now);
  later.deadline = now + 20s;
  FlowControlWork earlier = work("earlier", now);
  earlier.deadline = now + 10s;
  ASSERT_EQ(queue.admit(later, now, SaturationState::SATURATED).status,
            FlowControlStatus::OK);
  ASSERT_EQ(queue.admit(earlier, now, SaturationState::SATURATED).status,
            FlowControlStatus::OK);

  const FlowControlDispatch dispatch =
      queue.take_next(now, SaturationState::AVAILABLE);
  ASSERT_TRUE(dispatch.work.has_value());
  EXPECT_EQ(dispatch.work->request_uid, "earlier");
}

TEST(FlowControlQueueTest, UnknownUsesBoundedBestEffortProbeOnly) {
  FlowControlQueue queue(config());
  const auto now = FlowControlQueue::Clock::now();
  FlowControlWork strict = work("strict", now);
  strict.strict = true;
  ASSERT_EQ(queue.admit(strict, now, SaturationState::AVAILABLE).status,
            FlowControlStatus::OK);
  ASSERT_EQ(
      queue.admit(work("best", now, "b"), now, SaturationState::UNKNOWN).status,
      FlowControlStatus::OK);

  FlowControlDispatch dispatch = queue.take_next(now, SaturationState::UNKNOWN);
  ASSERT_TRUE(dispatch.work.has_value());
  EXPECT_EQ(dispatch.work->request_uid, "best");
  EXPECT_TRUE(dispatch.blind_probe);
  EXPECT_EQ(queue.take_next(now, SaturationState::UNKNOWN).status,
            FlowControlStatus::QUEUE_CAPACITY_EXHAUSTED);
  EXPECT_EQ(queue.snapshot().blind_probes_inflight, 1);
}

TEST(FlowControlQueueTest, CancellationAndWaitExpiryReleaseExactAccounting) {
  FlowControlConfig limits = config();
  limits.max_queue_wait_ms = 50;
  FlowControlQueue queue(limits);
  const auto now = FlowControlQueue::Clock::now();
  ASSERT_EQ(queue.admit(work("a", now), now, SaturationState::SATURATED).status,
            FlowControlStatus::OK);
  ASSERT_EQ(
      queue.admit(work("b", now, "b"), now, SaturationState::SATURATED).status,
      FlowControlStatus::OK);
  ASSERT_EQ(queue.cancel("a"), FlowControlStatus::OK);

  const std::vector<FlowControlWork> expired =
      queue.take_expired(now + 51ms, 8);
  ASSERT_EQ(expired.size(), 1);
  EXPECT_EQ(expired.front().request_uid, "b");
  EXPECT_EQ(queue.snapshot().queued_requests, 0);
  EXPECT_EQ(queue.snapshot().queued_bytes, 0);
}

TEST(FlowControlQueueTest, FullModelDoesNotBlockAnotherModel) {
  FlowControlConfig limits = config();
  limits.max_model_dispatched_request_contexts = 1;
  FlowControlQueue queue(limits);
  const auto now = FlowControlQueue::Clock::now();
  ASSERT_EQ(
      queue.admit(work("m1-a", now), now, SaturationState::SATURATED).status,
      FlowControlStatus::OK);
  ASSERT_EQ(queue.admit(work("m1-b", now, "b"), now, SaturationState::SATURATED)
                .status,
            FlowControlStatus::OK);
  ASSERT_EQ(
      queue
          .admit(
              work("m2", now, "c", "model-b"), now, SaturationState::SATURATED)
          .status,
      FlowControlStatus::OK);

  FlowControlDispatch dispatch =
      queue.take_next(now, SaturationState::AVAILABLE);
  ASSERT_TRUE(dispatch.work.has_value());
  EXPECT_EQ(dispatch.work->request_uid, "m1-a");
  dispatch = queue.take_next(now, SaturationState::AVAILABLE);
  ASSERT_TRUE(dispatch.work.has_value());
  EXPECT_EQ(dispatch.work->request_uid, "m2");
}

TEST(FlowControlQueueTest, FailedRouteCanReturnWithoutLosingReservation) {
  FlowControlQueue queue(config());
  const auto now = FlowControlQueue::Clock::now();
  ASSERT_EQ(
      queue.admit(work("request", now), now, SaturationState::SATURATED).status,
      FlowControlStatus::OK);
  ASSERT_EQ(queue.take_next(now, SaturationState::AVAILABLE).status,
            FlowControlStatus::OK);
  EXPECT_EQ(queue.snapshot().dispatched_requests, 1);

  ASSERT_EQ(queue.return_to_queue("request"), FlowControlStatus::OK);
  const FlowControlSnapshot snapshot = queue.snapshot();
  EXPECT_EQ(snapshot.queued_requests, 1);
  EXPECT_EQ(snapshot.dispatched_requests, 0);
  EXPECT_EQ(snapshot.queued_prompt_tokens, 10);
  EXPECT_EQ(snapshot.queued_bytes, 100);
}

TEST(FlowControlQueueTest, DrainRetryReturnsOnlyUndispatchedWork) {
  FlowControlQueue queue(config());
  const auto now = FlowControlQueue::Clock::now();
  ASSERT_EQ(
      queue.admit(work("dispatched", now), now, SaturationState::SATURATED)
          .status,
      FlowControlStatus::OK);
  ASSERT_EQ(
      queue.admit(work("queued", now, "b"), now, SaturationState::SATURATED)
          .status,
      FlowControlStatus::OK);
  const FlowControlDispatch dispatch =
      queue.take_next(now, SaturationState::AVAILABLE);
  ASSERT_TRUE(dispatch.work.has_value());
  EXPECT_EQ(dispatch.work->request_uid, "dispatched");

  const std::vector<FlowControlWork> retried = queue.retry_undispatched(8);
  ASSERT_EQ(retried.size(), 1);
  EXPECT_EQ(retried.front().request_uid, "queued");
  EXPECT_EQ(queue.snapshot().queued_requests, 0);
  EXPECT_EQ(queue.snapshot().dispatched_requests, 1);
  EXPECT_EQ(queue.complete("dispatched"), FlowControlStatus::OK);
}

TEST(FlowControlQueueTest, StarvationBoundAllowsLowerPriorityDispatch) {
  FlowControlQueue queue(config());
  const auto now = FlowControlQueue::Clock::now();
  ASSERT_EQ(queue
                .admit(work("low", now, "low", "model-a", 3),
                       now,
                       SaturationState::SATURATED)
                .status,
            FlowControlStatus::OK);
  for (int index = 0; index < 2; ++index) {
    const std::string uid = "high-" + std::to_string(index);
    ASSERT_EQ(queue
                  .admit(work(uid, now, "high", "model-a", 1),
                         now,
                         SaturationState::SATURATED)
                  .status,
              FlowControlStatus::OK);
    const FlowControlDispatch dispatch =
        queue.take_next(now, SaturationState::AVAILABLE);
    ASSERT_TRUE(dispatch.work.has_value());
    EXPECT_EQ(dispatch.work->request_uid, uid);
    ASSERT_EQ(queue.complete(uid), FlowControlStatus::OK);
  }
  ASSERT_EQ(queue
                .admit(work("high-2", now, "high", "model-a", 1),
                       now,
                       SaturationState::SATURATED)
                .status,
            FlowControlStatus::OK);
  const FlowControlDispatch dispatch =
      queue.take_next(now, SaturationState::AVAILABLE);
  ASSERT_TRUE(dispatch.work.has_value());
  EXPECT_EQ(dispatch.work->request_uid, "low");
}

TEST(FlowControlQueueTest, ConcurrentAdmissionNeverExceedsHardLimit) {
  FlowControlConfig limits = config();
  limits.max_queued_requests = 8;
  limits.max_model_queued_requests = 8;
  limits.max_queued_requests_per_tenant = 8;
  limits.service_crash_request_budget = 12;
  FlowControlQueue queue(limits);
  const auto now = FlowControlQueue::Clock::now();
  std::vector<std::thread> threads;
  for (int index = 0; index < 32; ++index) {
    threads.emplace_back([&, index] {
      queue.admit(work("request-" + std::to_string(index), now),
                  now,
                  SaturationState::SATURATED);
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(queue.snapshot().queued_requests, 8);
}

}  // namespace
}  // namespace xllm_service
