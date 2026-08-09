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

#include <brpc/closure_guard.h>
#include <brpc/server.h>
#include <bthread/bthread.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include "disagg_pd.pb.h"
#include "provider/provider_contract.h"

namespace xllm_service {
namespace {

class FakeSnapshotService final : public xllm::proto::DisaggPDService {
 public:
  void set_delay_us(uint64_t delay_us) { delay_us_.store(delay_us); }
  int32_t query_count() const { return query_count_.load(); }

  void GetKVCacheSnapshot(google::protobuf::RpcController* controller,
                          const xllm::proto::KVCacheSnapshotRequest* request,
                          xllm::proto::KVCacheSnapshotPage* response,
                          google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    static_cast<void>(controller);
    query_count_.fetch_add(1);
    bthread_usleep(delay_us_.load());
    response->set_contract_version(provider::kProviderContractVersion);
    response->set_status(xllm::proto::KV_SNAPSHOT_STATUS_OK);
    *response->mutable_identity() = request->identity();
    response->set_snapshot_id(
        request->snapshot_id().empty() ? "snapshot-a" : request->snapshot_id());
    response->set_base_event_seq(7);
    response->set_next_cursor(request->cursor());
    response->set_done(true);
  }

 private:
  std::atomic<uint64_t> delay_us_{0};
  std::atomic<int32_t> query_count_{0};
};

class KVSnapshotClientTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(server_.AddService(&service_, brpc::SERVER_DOESNT_OWN_SERVICE),
              0);
    ASSERT_EQ(server_.Start("127.0.0.1:0", nullptr), 0);
    channel_ = std::make_shared<brpc::Channel>();
    const std::string address =
        "127.0.0.1:" + std::to_string(server_.listen_address().port);
    ASSERT_EQ(channel_->Init(address.c_str(), "", nullptr), 0);
  }

  void TearDown() override {
    ASSERT_EQ(server_.Stop(0), 0);
    ASSERT_EQ(server_.Join(), 0);
  }

  KVSnapshotPageQuery make_query(uint64_t timeout_ms = 200) {
    KVSnapshotPageQuery query;
    query.channel = channel_;
    query.timeout_ms = timeout_ms;
    query.max_response_bytes = 4096;
    query.request.set_contract_version(provider::kProviderContractVersion);
    query.request.mutable_identity()->mutable_engine()->set_provider_id(
        xllm::proto::PROVIDER_ID_XLLM_NATIVE);
    query.request.mutable_identity()->mutable_engine()->set_profile_digest(
        "profile");
    query.request.mutable_identity()->mutable_engine()->set_engine_uid(
        "engine");
    query.request.mutable_identity()->mutable_engine()->set_incarnation_id(
        "incarnation");
    query.request.mutable_identity()->set_model_revision("model");
    query.request.mutable_identity()->set_kv_namespace("namespace");
    query.request.mutable_identity()->set_cache_epoch(1);
    query.request.set_max_entries(16);
    query.request.set_max_bytes(2048);
    query.request.set_max_generation_time_ms(20);
    return query;
  }

  FakeSnapshotService service_;
  brpc::Server server_;
  std::shared_ptr<brpc::Channel> channel_;
};

TEST_F(KVSnapshotClientTest, QueriesValidatedPageOverLoopback) {
  const std::vector<KVSnapshotPageResult> results =
      query_kv_snapshot_pages({make_query()});
  ASSERT_EQ(results.size(), 1u);
  ASSERT_TRUE(results[0].ok) << results[0].message;
  ASSERT_TRUE(results[0].page.has_value());
  EXPECT_EQ(results[0].page->snapshot_id(), "snapshot-a");
  EXPECT_EQ(results[0].page->base_event_seq(), 7);
}

TEST_F(KVSnapshotClientTest, ReportsTimeoutAndRejectsInvalidWithoutRpc) {
  service_.set_delay_us(80000);
  const std::vector<KVSnapshotPageResult> timeout =
      query_kv_snapshot_pages({make_query(5)});
  ASSERT_EQ(timeout.size(), 1u);
  EXPECT_FALSE(timeout[0].ok);
  EXPECT_TRUE(timeout[0].timed_out);

  const int32_t queries_before_invalid = service_.query_count();
  KVSnapshotPageQuery invalid = make_query();
  invalid.request.set_max_entries(0);
  const std::vector<KVSnapshotPageResult> rejected =
      query_kv_snapshot_pages({invalid});
  ASSERT_EQ(rejected.size(), 1u);
  EXPECT_FALSE(rejected[0].ok);
  EXPECT_EQ(service_.query_count(), queries_before_invalid);
}

TEST_F(KVSnapshotClientTest, RejectsOversizedResponse) {
  KVSnapshotPageQuery query = make_query();
  query.max_response_bytes = 1;
  const std::vector<KVSnapshotPageResult> results =
      query_kv_snapshot_pages({query});
  ASSERT_EQ(results.size(), 1u);
  EXPECT_FALSE(results[0].ok);
  EXPECT_EQ(service_.query_count(), 1);
}

}  // namespace
}  // namespace xllm_service
