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

#include <string>
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

}  // namespace
}  // namespace xllm_service
