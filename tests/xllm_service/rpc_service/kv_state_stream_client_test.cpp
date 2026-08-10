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

#include <brpc/closure_guard.h>
#include <brpc/server.h>
#include <bthread/bthread.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "provider/provider_contract.h"
#include "xllm_rpc_service.pb.h"

namespace xllm_service {
namespace {

class FakeKVStateStreamService final : public proto::XllmRpcService {
 public:
  void set_delay_us(uint64_t delay_us) { delay_us_.store(delay_us); }
  int32_t peak_in_flight() const { return peak_in_flight_.load(); }
  int32_t push_count() const { return push_count_.load(); }

  void PushKVState(google::protobuf::RpcController* controller,
                   const xllm::proto::KVStateBatch* request,
                   proto::Status* response,
                   google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    static_cast<void>(controller);
    push_count_.fetch_add(1);
    const int32_t in_flight = in_flight_.fetch_add(1) + 1;
    int32_t peak = peak_in_flight_.load();
    while (peak < in_flight &&
           !peak_in_flight_.compare_exchange_weak(peak, in_flight)) {
    }
    bthread_usleep(delay_us_.load());
    response->set_ok(request->contract_version() ==
                     provider::kProviderContractVersion);
    in_flight_.fetch_sub(1);
  }

 private:
  std::atomic<uint64_t> delay_us_{0};
  std::atomic<int32_t> in_flight_{0};
  std::atomic<int32_t> peak_in_flight_{0};
  std::atomic<int32_t> push_count_{0};
};

class KVStateStreamClientTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(server_.AddService(&service_, brpc::SERVER_DOESNT_OWN_SERVICE),
              0);
    ASSERT_EQ(server_.Start("127.0.0.1:0", nullptr), 0);
    address_ = "127.0.0.1:" + std::to_string(server_.listen_address().port);
  }

  void TearDown() override {
    ASSERT_EQ(server_.Stop(0), 0);
    ASSERT_EQ(server_.Join(), 0);
  }

  KVStateStreamPush make_push(uint64_t sequence, uint64_t timeout_ms = 200) {
    KVStateStreamPush push;
    push.subscriber = address_;
    push.timeout_ms = timeout_ms;
    push.batch.set_contract_version(provider::kProviderContractVersion);
    push.batch.set_master_incarnation("master-inc");
    push.batch.set_stream_seq(sequence);
    push.batch.set_stream_epoch(1);
    xllm::proto::KVEventBatch* engine = push.batch.add_engine_batches();
    engine->set_contract_version(provider::kProviderContractVersion);
    return push;
  }

  FakeKVStateStreamService service_;
  brpc::Server server_;
  std::string address_;
};

TEST_F(KVStateStreamClientTest, PushesBatchOverLoopback) {
  const std::vector<KVStateStreamPushResult> results =
      push_kv_state_stream_batches({make_push(1)});
  ASSERT_EQ(results.size(), 1u);
  EXPECT_TRUE(results[0].ok) << results[0].message;
  EXPECT_EQ(service_.push_count(), 1);
}

TEST_F(KVStateStreamClientTest, IssuesSubscriberCallsConcurrently) {
  service_.set_delay_us(40000);
  std::vector<KVStateStreamPush> pushes;
  for (uint64_t sequence = 1; sequence <= 8; ++sequence) {
    pushes.emplace_back(make_push(sequence));
  }
  const std::vector<KVStateStreamPushResult> results =
      push_kv_state_stream_batches(pushes);
  for (const KVStateStreamPushResult& result : results) {
    EXPECT_TRUE(result.ok) << result.message;
  }
  EXPECT_EQ(service_.push_count(), 8);
  EXPECT_GE(service_.peak_in_flight(), 2);
}

TEST_F(KVStateStreamClientTest, ReportsRpcTimeout) {
  service_.set_delay_us(80000);
  const std::vector<KVStateStreamPushResult> timeout =
      push_kv_state_stream_batches({make_push(1, 5)});
  ASSERT_EQ(timeout.size(), 1u);
  EXPECT_FALSE(timeout[0].ok);
  EXPECT_TRUE(timeout[0].timed_out);
}

TEST_F(KVStateStreamClientTest, RejectsInvalidBatchWithoutNetwork) {
  KVStateStreamPush invalid = make_push(2);
  invalid.batch.clear_master_incarnation();
  const std::vector<KVStateStreamPushResult> rejected =
      push_kv_state_stream_batches({invalid});
  ASSERT_EQ(rejected.size(), 1u);
  EXPECT_FALSE(rejected[0].ok);
  EXPECT_EQ(service_.push_count(), 0);
}

}  // namespace
}  // namespace xllm_service
