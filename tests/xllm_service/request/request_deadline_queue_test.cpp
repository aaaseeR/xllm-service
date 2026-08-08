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

#include "request/request_deadline_queue.h"

#include <gtest/gtest.h>

#include <chrono>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "request/request.h"

namespace xllm_service {
namespace {

TEST(RequestDeadlineQueueTest, ExpiresInDeadlineOrderWithBoundedBatches) {
  const RequestDeadlineQueue::TimePoint now(std::chrono::milliseconds(100));
  RequestDeadlineQueue queue(/*capacity=*/3);
  auto first = std::make_shared<Request>();
  auto second = std::make_shared<Request>();
  auto third = std::make_shared<Request>();

  EXPECT_EQ(queue.insert("second", second, now + std::chrono::milliseconds(20)),
            RequestDeadlineQueueStatus::kOk);
  EXPECT_EQ(queue.insert("first", first, now + std::chrono::milliseconds(10)),
            RequestDeadlineQueueStatus::kOk);
  EXPECT_EQ(queue.insert("third", third, now + std::chrono::milliseconds(30)),
            RequestDeadlineQueueStatus::kOk);

  EXPECT_TRUE(
      queue.take_expired(now + std::chrono::milliseconds(9), 3).empty());
  EXPECT_FALSE(queue.has_expired(now + std::chrono::milliseconds(9)));
  EXPECT_TRUE(queue.has_expired(now + std::chrono::milliseconds(10)));
  const auto first_batch =
      queue.take_expired(now + std::chrono::milliseconds(25), 1);
  ASSERT_EQ(first_batch.size(), 1);
  EXPECT_EQ(first_batch[0], first);
  const auto second_batch =
      queue.take_expired(now + std::chrono::milliseconds(25), 3);
  ASSERT_EQ(second_batch.size(), 1);
  EXPECT_EQ(second_batch[0], second);
  EXPECT_EQ(queue.size(), 1);
}

TEST(RequestDeadlineQueueTest, RejectsDuplicateAndCapacityOverflow) {
  const RequestDeadlineQueue::TimePoint now(std::chrono::milliseconds(100));
  RequestDeadlineQueue queue(/*capacity=*/1);
  auto request = std::make_shared<Request>();

  EXPECT_EQ(queue.insert("request", request, now),
            RequestDeadlineQueueStatus::kOk);
  EXPECT_EQ(queue.insert("request", request, now),
            RequestDeadlineQueueStatus::kDuplicate);
  EXPECT_EQ(queue.insert("another", std::make_shared<Request>(), now),
            RequestDeadlineQueueStatus::kCapacityExceeded);
  EXPECT_EQ(queue.insert("", request, now),
            RequestDeadlineQueueStatus::kInvalid);
}

TEST(RequestDeadlineQueueTest, EraseAndExpiredWeakReferencesReleaseCapacity) {
  const RequestDeadlineQueue::TimePoint now(std::chrono::milliseconds(100));
  RequestDeadlineQueue queue(/*capacity=*/2);
  auto erased = std::make_shared<Request>();
  auto expired = std::make_shared<Request>();
  ASSERT_EQ(queue.insert("erased", erased, now),
            RequestDeadlineQueueStatus::kOk);
  ASSERT_EQ(queue.insert("expired", expired, now),
            RequestDeadlineQueueStatus::kOk);

  EXPECT_TRUE(queue.erase("erased"));
  EXPECT_FALSE(queue.erase("erased"));
  expired.reset();
  EXPECT_TRUE(queue.take_expired(now, 2).empty());
  EXPECT_EQ(queue.size(), 0);
  EXPECT_EQ(queue.insert("replacement", std::make_shared<Request>(), now),
            RequestDeadlineQueueStatus::kOk);
}

TEST(RequestDeadlineQueueTest, ZeroCapacityAndZeroBatchStayBounded) {
  const RequestDeadlineQueue::TimePoint now(std::chrono::milliseconds(100));
  RequestDeadlineQueue queue(/*capacity=*/0);
  EXPECT_EQ(queue.insert("request", std::make_shared<Request>(), now),
            RequestDeadlineQueueStatus::kCapacityExceeded);

  RequestDeadlineQueue nonempty(/*capacity=*/1);
  ASSERT_EQ(nonempty.insert("request", std::make_shared<Request>(), now),
            RequestDeadlineQueueStatus::kOk);
  EXPECT_TRUE(nonempty.take_expired(now, 0).empty());
  EXPECT_EQ(nonempty.size(), 1);
}

TEST(RequestDeadlineQueueTest, HugeBatchIsCappedByCurrentQueueSize) {
  const RequestDeadlineQueue::TimePoint now(std::chrono::milliseconds(100));
  RequestDeadlineQueue queue(/*capacity=*/1);
  auto request = std::make_shared<Request>();
  ASSERT_EQ(queue.insert("request", request, now),
            RequestDeadlineQueueStatus::kOk);

  const auto expired =
      queue.take_expired(now, std::numeric_limits<size_t>::max());
  ASSERT_EQ(expired.size(), 1);
  EXPECT_EQ(expired[0], request);
}

TEST(RequestDeadlineQueueTest, ConcurrentInsertPreservesEveryEntry) {
  constexpr size_t kThreadCount = 8;
  constexpr size_t kRequestsPerThread = 32;
  constexpr size_t kRequestCount = kThreadCount * kRequestsPerThread;
  const RequestDeadlineQueue::TimePoint now(std::chrono::milliseconds(100));
  RequestDeadlineQueue queue(/*capacity=*/kRequestCount);
  std::vector<std::shared_ptr<Request>> requests;
  requests.reserve(kRequestCount);
  for (size_t i = 0; i < kRequestCount; ++i) {
    requests.emplace_back(std::make_shared<Request>());
  }
  std::vector<RequestDeadlineQueueStatus> statuses(
      kRequestCount, RequestDeadlineQueueStatus::kInvalid);
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);

  for (size_t thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    threads.emplace_back([&, thread_index]() {
      const size_t begin = thread_index * kRequestsPerThread;
      const size_t end = begin + kRequestsPerThread;
      for (size_t i = begin; i < end; ++i) {
        statuses[i] = queue.insert(std::to_string(i), requests[i], now);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  for (const auto status : statuses) {
    EXPECT_EQ(status, RequestDeadlineQueueStatus::kOk);
  }
  EXPECT_EQ(queue.size(), kRequestCount);
  size_t expired_count = 0;
  while (queue.size() != 0) {
    const auto batch = queue.take_expired(now, /*max_entries=*/7);
    ASSERT_FALSE(batch.empty());
    EXPECT_LE(batch.size(), 7);
    expired_count += batch.size();
  }
  EXPECT_EQ(expired_count, kRequestCount);
}

}  // namespace
}  // namespace xllm_service
