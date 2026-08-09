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

#include "rpc_service/kv_state_stream_client.h"

#include <brpc/channel.h>
#include <brpc/controller.h>

#include <limits>
#include <memory>

#include "provider/provider_contract.h"
#include "xllm_rpc_service.pb.h"

namespace xllm_service {
namespace {

struct PushContext {
  brpc::Channel channel;
  brpc::Controller controller;
  proto::Status response;
  bool issued = false;
};

std::string validate_push(const KVStateStreamPush& push) {
  if (push.subscriber.empty()) {
    return "KV State Stream subscriber is empty";
  }
  if (push.timeout_ms == 0 ||
      push.timeout_ms >
          static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return "KV State Stream timeout is outside int32 range";
  }
  if (push.batch.contract_version() != provider::kProviderContractVersion ||
      push.batch.master_incarnation().empty() || push.batch.stream_seq() == 0 ||
      push.batch.stream_epoch() == 0 || push.batch.engine_batches().empty()) {
    return "KV State Stream batch identity is incomplete";
  }
  return "";
}

}  // namespace

std::vector<KVStateStreamPushResult> push_kv_state_stream_batches(
    const std::vector<KVStateStreamPush>& pushes) {
  std::vector<KVStateStreamPushResult> results(pushes.size());
  std::vector<std::unique_ptr<PushContext>> contexts;
  contexts.reserve(pushes.size());

  for (size_t index = 0; index < pushes.size(); ++index) {
    const KVStateStreamPush& push = pushes[index];
    std::unique_ptr<PushContext> context = std::make_unique<PushContext>();
    results[index].message = validate_push(push);
    if (results[index].message.empty()) {
      brpc::ChannelOptions options;
      options.timeout_ms = static_cast<int32_t>(push.timeout_ms);
      options.connect_timeout_ms = static_cast<int32_t>(push.timeout_ms);
      options.max_retry = 0;
      if (context->channel.Init(push.subscriber.c_str(), "", &options) != 0) {
        results[index].message =
            "KV State Stream channel initialization failed";
      } else {
        context->controller.set_timeout_ms(
            static_cast<int32_t>(push.timeout_ms));
        proto::XllmRpcService_Stub stub(&context->channel);
        stub.PushKVState(&context->controller,
                         &push.batch,
                         &context->response,
                         brpc::DoNothing());
        context->issued = true;
      }
    }
    contexts.emplace_back(std::move(context));
  }

  for (const std::unique_ptr<PushContext>& context : contexts) {
    if (context->issued) {
      brpc::Join(context->controller.call_id());
    }
  }
  for (size_t index = 0; index < pushes.size(); ++index) {
    const PushContext& context = *contexts[index];
    if (!context.issued) {
      continue;
    }
    if (context.controller.Failed()) {
      results[index].timed_out =
          context.controller.ErrorCode() == brpc::ERPCTIMEDOUT;
      results[index].message = context.controller.ErrorText();
      continue;
    }
    results[index].ok = context.response.ok();
    results[index].message = context.response.status_msg();
  }
  return results;
}

}  // namespace xllm_service
