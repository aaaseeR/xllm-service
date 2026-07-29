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

#include "transport/channel_pool.h"

#include <gtest/gtest.h>

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace xllm_service {
namespace {

TEST(ChannelPoolTest, ReusesChannelForActiveIncarnation) {
  int32_t create_count = 0;
  ChannelPool pool([&create_count](const std::string&) {
    ++create_count;
    return std::make_shared<brpc::Channel>();
  });

  ASSERT_TRUE(pool.activate("127.0.0.1:8000", "incarnation-1"));
  const auto first =
      pool.get_or_create("127.0.0.1:8000", "incarnation-1");
  const auto second =
      pool.get_or_create("127.0.0.1:8000", "incarnation-1");

  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first, second);
  EXPECT_EQ(create_count, 1);
  EXPECT_EQ(pool.channel_count(), 1);
  EXPECT_EQ(pool.endpoint_count(), 1);
}

TEST(ChannelPoolTest, ReplacesIncarnationAndIgnoresStaleRemoval) {
  ChannelPool pool([](const std::string&) {
    return std::make_shared<brpc::Channel>();
  });

  ASSERT_TRUE(pool.activate("127.0.0.1:8000", "incarnation-1"));
  const auto old_channel =
      pool.get_or_create("127.0.0.1:8000", "incarnation-1");
  ASSERT_NE(old_channel, nullptr);

  ASSERT_TRUE(pool.activate("127.0.0.1:8000", "incarnation-2"));
  EXPECT_EQ(pool.get_or_create("127.0.0.1:8000", "incarnation-1"),
            nullptr);
  EXPECT_FALSE(pool.remove("127.0.0.1:8000", "incarnation-1"));
  const auto new_channel =
      pool.get_or_create("127.0.0.1:8000", "incarnation-2");

  ASSERT_NE(new_channel, nullptr);
  EXPECT_NE(old_channel, new_channel);
  EXPECT_TRUE(pool.remove("127.0.0.1:8000", "incarnation-2"));
  EXPECT_EQ(pool.endpoint_count(), 0);
}

TEST(ChannelPoolTest, DiscardsChannelCreatedForReplacedIncarnation) {
  std::mutex mutex;
  std::condition_variable condition;
  bool old_factory_started = false;
  bool release_old_factory = false;
  int32_t create_count = 0;
  ChannelPool pool([&](const std::string&) {
    std::unique_lock<std::mutex> lock(mutex);
    ++create_count;
    if (create_count == 1) {
      old_factory_started = true;
      condition.notify_all();
      condition.wait(lock, [&release_old_factory]() {
        return release_old_factory;
      });
    }
    return std::make_shared<brpc::Channel>();
  });
  ASSERT_TRUE(pool.activate("127.0.0.1:8000", "incarnation-1"));

  std::shared_ptr<brpc::Channel> stale_channel;
  std::thread creator([&]() {
    stale_channel =
        pool.get_or_create("127.0.0.1:8000", "incarnation-1");
  });
  {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&old_factory_started]() {
      return old_factory_started;
    });
  }

  ASSERT_TRUE(pool.activate("127.0.0.1:8000", "incarnation-2"));
  {
    std::lock_guard<std::mutex> lock(mutex);
    release_old_factory = true;
  }
  condition.notify_all();
  creator.join();

  EXPECT_EQ(stale_channel, nullptr);
  EXPECT_EQ(pool.channel_count(), 0);
  EXPECT_NE(pool.get_or_create("127.0.0.1:8000", "incarnation-2"),
            nullptr);
  EXPECT_EQ(create_count, 2);
}

}  // namespace
}  // namespace xllm_service
