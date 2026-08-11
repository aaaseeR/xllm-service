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

#include "rpc_service/kv_snapshot_client.h"

#include <brpc/controller.h>

#include <limits>
#include <memory>
#include <utility>

#include "disagg_pd.pb.h"
#include "provider/identity_key.h"
#include "provider/provider_contract.h"

namespace xllm_service {
namespace {

struct QueryContext {
  brpc::Controller controller;
  xllm::proto::KVCacheSnapshotPage response;
  bool issued = false;
};

std::string validate_query(const KVSnapshotPageQuery& query) {
  if (query.channel == nullptr) {
    return "KV snapshot Engine channel is unavailable";
  }
  if (query.timeout_ms == 0 ||
      query.timeout_ms >
          static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return "KV snapshot timeout is outside int32 range";
  }
  if (query.max_response_bytes == 0 ||
      query.request.contract_version() != provider::kProviderContractVersion ||
      !query.request.has_identity() ||
      query.request.identity().engine().engine_uid().empty() ||
      query.request.identity().engine().incarnation_id().empty() ||
      query.request.max_entries() == 0 || query.request.max_bytes() == 0 ||
      query.request.max_generation_time_ms() == 0) {
    return "KV snapshot request is incomplete";
  }
  return "";
}

std::string validate_response(const KVSnapshotPageQuery& query,
                              const xllm::proto::KVCacheSnapshotPage& page) {
  if (page.ByteSizeLong() > query.max_response_bytes) {
    return "KV snapshot response exceeds byte capacity";
  }
  if (page.contract_version() != provider::kProviderContractVersion ||
      !xllm::proto::KVSnapshotStatus_IsValid(page.status()) ||
      page.status() == xllm::proto::KV_SNAPSHOT_STATUS_UNSPECIFIED ||
      !provider::same_kv_stream_identity(page.identity(),
                                         query.request.identity())) {
    return "KV snapshot response identity is invalid";
  }
  if (page.status() == xllm::proto::KV_SNAPSHOT_STATUS_OK &&
      (page.snapshot_id().empty() ||
       (!query.request.snapshot_id().empty() &&
        page.snapshot_id() != query.request.snapshot_id()) ||
       page.next_cursor() < query.request.cursor())) {
    return "KV snapshot response pagination is invalid";
  }
  return "";
}

}  // namespace

std::vector<KVSnapshotPageResult> query_kv_snapshot_pages(
    const std::vector<KVSnapshotPageQuery>& queries) {
  std::vector<KVSnapshotPageResult> results(queries.size());
  std::vector<std::unique_ptr<QueryContext>> contexts;
  contexts.reserve(queries.size());
  for (size_t index = 0; index < queries.size(); ++index) {
    const KVSnapshotPageQuery& query = queries[index];
    std::unique_ptr<QueryContext> context = std::make_unique<QueryContext>();
    results[index].message = validate_query(query);
    if (results[index].message.empty()) {
      context->controller.set_timeout_ms(
          static_cast<int32_t>(query.timeout_ms));
      xllm::proto::DisaggPDService_Stub stub(query.channel.get());
      stub.GetKVCacheSnapshot(&context->controller,
                              &query.request,
                              &context->response,
                              brpc::DoNothing());
      context->issued = true;
    }
    contexts.emplace_back(std::move(context));
  }

  for (const std::unique_ptr<QueryContext>& context : contexts) {
    if (context->issued) {
      brpc::Join(context->controller.call_id());
    }
  }
  for (size_t index = 0; index < queries.size(); ++index) {
    const QueryContext& context = *contexts[index];
    if (!context.issued) {
      continue;
    }
    if (context.controller.Failed()) {
      results[index].timed_out =
          context.controller.ErrorCode() == brpc::ERPCTIMEDOUT;
      results[index].message = context.controller.ErrorText();
      continue;
    }
    results[index].message =
        validate_response(queries[index], context.response);
    if (!results[index].message.empty()) {
      continue;
    }
    results[index].ok = true;
    results[index].page = context.response;
  }
  return results;
}

}  // namespace xllm_service
