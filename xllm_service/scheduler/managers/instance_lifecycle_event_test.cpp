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

#include "scheduler/managers/instance_lifecycle_event.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace xllm_service {
namespace {

TEST(InstanceLifecycleEventDispatcherTest, PublishesImmutableEventToHandlers) {
  std::vector<std::string> observed;
  InstanceLifecycleEventDispatcher dispatcher(
      std::vector<InstanceLifecycleEventDispatcher::Handler>{
          [&observed](const InstanceLifecycleEvent& event) {
            observed.push_back(event.instance.name);
          },
          [&observed](const InstanceLifecycleEvent& event) {
            observed.push_back(event.instance.incarnation_id);
          }});

  InstanceLifecycleEvent event;
  event.type = InstanceLifecycleEventType::DEREGISTERED;
  event.instance.name = "prefill-0";
  event.instance.incarnation_id = "incarnation-2";
  dispatcher.publish(event);

  EXPECT_EQ(observed,
            (std::vector<std::string>{"prefill-0", "incarnation-2"}));
  EXPECT_EQ(event.type, InstanceLifecycleEventType::DEREGISTERED);
  EXPECT_EQ(event.instance.name, "prefill-0");
}

TEST(InstanceLifecycleEventDispatcherTest, SkipsEmptyHandlers) {
  int invocation_count = 0;
  InstanceLifecycleEventDispatcher dispatcher(
      std::vector<InstanceLifecycleEventDispatcher::Handler>{
          {},
          [&invocation_count](const InstanceLifecycleEvent&) {
            ++invocation_count;
          }});

  dispatcher.publish(InstanceLifecycleEvent{});

  EXPECT_EQ(invocation_count, 1);
}

TEST(InstanceLifecycleEventDispatcherTest, IgnoresEventsAfterClose) {
  int invocation_count = 0;
  InstanceLifecycleEventDispatcher dispatcher(
      [&invocation_count](const InstanceLifecycleEvent&) {
        ++invocation_count;
      });

  dispatcher.publish(InstanceLifecycleEvent{});
  dispatcher.close();
  dispatcher.publish(InstanceLifecycleEvent{});

  EXPECT_EQ(invocation_count, 1);
}

TEST(InstanceLifecycleEventDispatcherTest, CloseWaitsForInflightHandler) {
  std::mutex mutex;
  std::condition_variable condition;
  bool handler_started = false;
  bool release_handler = false;
  InstanceLifecycleEventDispatcher dispatcher(
      [&](const InstanceLifecycleEvent&) {
        std::unique_lock<std::mutex> lock(mutex);
        handler_started = true;
        condition.notify_all();
        condition.wait(lock, [&release_handler]() { return release_handler; });
      });

  std::thread publish_thread(
      [&dispatcher]() { dispatcher.publish(InstanceLifecycleEvent{}); });
  bool started = false;
  {
    std::unique_lock<std::mutex> lock(mutex);
    started = condition.wait_for(lock,
                                 std::chrono::seconds(5),
                                 [&handler_started]() {
                                   return handler_started;
                                 });
  }
  EXPECT_TRUE(started);
  if (!started) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      release_handler = true;
    }
    condition.notify_all();
    publish_thread.join();
    dispatcher.close();
    return;
  }

  std::atomic<bool> close_returned{false};
  std::thread close_thread([&dispatcher, &close_returned]() {
    dispatcher.close();
    close_returned.store(true);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(close_returned.load());

  {
    std::lock_guard<std::mutex> lock(mutex);
    release_handler = true;
  }
  condition.notify_all();
  publish_thread.join();
  close_thread.join();

  EXPECT_TRUE(close_returned.load());
}

}  // namespace
}  // namespace xllm_service
