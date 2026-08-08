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

#include "request/client_disconnect_monitor.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace xllm_service {
namespace {

TEST(ClientDisconnectMonitorTest, ReservesCapacityBeforeNotification) {
  ClientDisconnectMonitor monitor(2);
  auto first = std::make_shared<Request>();
  auto second = std::make_shared<Request>();
  auto overflow = std::make_shared<Request>();

  EXPECT_EQ(monitor.register_request("first", first),
            ClientDisconnectMonitorStatus::OK);
  EXPECT_EQ(monitor.register_request("second", second),
            ClientDisconnectMonitorStatus::OK);
  EXPECT_EQ(monitor.register_request("first", first),
            ClientDisconnectMonitorStatus::DUPLICATE);
  EXPECT_EQ(monitor.register_request("overflow", overflow),
            ClientDisconnectMonitorStatus::CAPACITY_EXCEEDED);
  EXPECT_EQ(monitor.size(), 2u);
}

TEST(ClientDisconnectMonitorTest, NotificationIsIdempotentAndBounded) {
  ClientDisconnectMonitor monitor(2);
  auto first = std::make_shared<Request>();
  auto second = std::make_shared<Request>();
  ASSERT_EQ(monitor.register_request("first", first),
            ClientDisconnectMonitorStatus::OK);
  ASSERT_EQ(monitor.register_request("second", second),
            ClientDisconnectMonitorStatus::OK);

  monitor.notify_disconnected("first");
  monitor.notify_disconnected("first");
  monitor.notify_disconnected("unknown");
  monitor.notify_disconnected("second");
  EXPECT_TRUE(first->client_disconnected.load(std::memory_order_acquire));
  EXPECT_TRUE(second->client_disconnected.load(std::memory_order_acquire));
  EXPECT_EQ(monitor.pending(), 2u);

  std::vector<std::shared_ptr<Request>> first_batch =
      monitor.take_disconnected(1);
  ASSERT_EQ(first_batch.size(), 1u);
  EXPECT_EQ(first_batch.front(), first);
  std::vector<std::shared_ptr<Request>> second_batch =
      monitor.take_disconnected(8);
  ASSERT_EQ(second_batch.size(), 1u);
  EXPECT_EQ(second_batch.front(), second);
  EXPECT_EQ(monitor.pending(), 0u);
}

TEST(ClientDisconnectMonitorTest, EraseAndExpiredWeakEntryReleaseCapacity) {
  ClientDisconnectMonitor monitor(1);
  auto request = std::make_shared<Request>();
  ASSERT_EQ(monitor.register_request("request", request),
            ClientDisconnectMonitorStatus::OK);
  EXPECT_TRUE(monitor.erase("request"));
  EXPECT_FALSE(monitor.erase("request"));

  ASSERT_EQ(monitor.register_request("expired", request),
            ClientDisconnectMonitorStatus::OK);
  request.reset();
  monitor.notify_disconnected("expired");
  EXPECT_EQ(monitor.size(), 0u);
  EXPECT_TRUE(monitor.take_disconnected(1).empty());
}

TEST(ClientDisconnectMonitorTest, ConcurrentNotificationsDeliverOnce) {
  ClientDisconnectMonitor monitor(1);
  auto request = std::make_shared<Request>();
  ASSERT_EQ(monitor.register_request("request", request),
            ClientDisconnectMonitorStatus::OK);
  std::vector<std::thread> threads;
  for (size_t index = 0; index < 16; ++index) {
    threads.emplace_back([&]() { monitor.notify_disconnected("request"); });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(monitor.pending(), 1u);
  EXPECT_EQ(monitor.take_disconnected(16).size(), 1u);
}

TEST(ClientDisconnectMonitorTest, CloseRejectsLateRegistrationAndSignal) {
  ClientDisconnectMonitor monitor(1);
  auto request = std::make_shared<Request>();
  ASSERT_EQ(monitor.register_request("request", request),
            ClientDisconnectMonitorStatus::OK);
  monitor.close();
  monitor.notify_disconnected("request");

  EXPECT_EQ(monitor.size(), 0u);
  EXPECT_TRUE(monitor.take_disconnected(1).empty());
  EXPECT_EQ(monitor.register_request("late", request),
            ClientDisconnectMonitorStatus::CLOSED);
}

}  // namespace
}  // namespace xllm_service
