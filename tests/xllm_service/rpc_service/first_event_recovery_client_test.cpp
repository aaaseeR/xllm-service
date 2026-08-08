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

#include "rpc_service/first_event_recovery_client.h"

#include <brpc/closure_guard.h>
#include <brpc/server.h>
#include <bthread/bthread.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "disagg_pd.pb.h"
#include "request/output_event_sequencer.h"
#include "xllm_rpc_service.pb.h"

namespace xllm_service {
namespace {

constexpr size_t kMaxPayloadBytes = 4 * 1024 * 1024;

class FakeDecodeService final : public xllm::proto::DisaggPDService {
 public:
  void set_delay_us(uint64_t delay_us) { delay_us_.store(delay_us); }

  void set_decode_incarnation(std::string incarnation_id) {
    std::lock_guard<std::mutex> lock(incarnation_mutex_);
    decode_incarnation_id_ = std::move(incarnation_id);
  }

  int32_t peak_in_flight() const { return peak_in_flight_.load(); }
  int32_t query_count() const { return query_count_.load(); }

  void QueryRequest(google::protobuf::RpcController* controller,
                    const xllm::proto::AttemptControlRequest* request,
                    xllm::proto::AttemptControlResponse* response,
                    google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    static_cast<void>(controller);
    query_count_.fetch_add(1);
    const int32_t in_flight = in_flight_.fetch_add(1) + 1;
    int32_t peak = peak_in_flight_.load();
    while (peak < in_flight &&
           !peak_in_flight_.compare_exchange_weak(peak, in_flight)) {
    }
    bthread_usleep(delay_us_.load());

    std::string decode_incarnation_id;
    {
      std::lock_guard<std::mutex> lock(incarnation_mutex_);
      decode_incarnation_id = decode_incarnation_id_;
    }
    response->set_ok(true);
    xllm::proto::AttemptStatus* status = response->mutable_status();
    status->mutable_key()->CopyFrom(request->key());
    status->mutable_key()->set_incarnation_id(decode_incarnation_id);
    status->set_state(xllm::proto::ATTEMPT_LIFECYCLE_STATE_RUNNING);

    proto::DisaggStreamGeneration generation;
    generation.set_req_id("engine-request-" + request->key().request_uid());
    generation.set_service_req_id(request->key().request_uid());
    generation.set_output_event_seq(0);
    generation.set_attempt_seq(request->key().attempt_seq());
    generation.set_sender_engine_uid("prefill-1");
    generation.set_sender_incarnation_id("prefill-incarnation-1");
    generation.set_finished_on_prefill_instance(true);
    proto::SequenceOutput* output = generation.add_outputs();
    output->set_index(0);
    output->set_text("exact-first-event");
    output->add_token_ids(101);
    status->set_first_event_payload(generation.SerializeAsString());
    in_flight_.fetch_sub(1);
  }

 private:
  std::atomic<uint64_t> delay_us_{0};
  std::atomic<int32_t> in_flight_{0};
  std::atomic<int32_t> peak_in_flight_{0};
  std::atomic<int32_t> query_count_{0};
  mutable std::mutex incarnation_mutex_;
  std::string decode_incarnation_id_ = "decode-incarnation-1";
};

class FirstEventRecoveryClientTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(server_.AddService(&service_, brpc::SERVER_DOESNT_OWN_SERVICE),
              0);
    ASSERT_EQ(server_.Start("127.0.0.1:0", nullptr), 0);
    channel_ = std::make_shared<brpc::Channel>();
    ASSERT_EQ(
        channel_->Init("127.0.0.1", server_.listen_address().port, nullptr), 0);
  }

  void TearDown() override {
    ASSERT_EQ(server_.Stop(0), 0);
    ASSERT_EQ(server_.Join(), 0);
  }

  void restart_decode(std::string incarnation_id) {
    const int32_t port = server_.listen_address().port;
    ASSERT_EQ(server_.Stop(0), 0);
    ASSERT_EQ(server_.Join(), 0);
    service_.set_decode_incarnation(std::move(incarnation_id));
    ASSERT_EQ(
        server_.Start(("127.0.0.1:" + std::to_string(port)).c_str(), nullptr),
        0);
  }

  void reconnect_channel() {
    channel_ = std::make_shared<brpc::Channel>();
    ASSERT_EQ(
        channel_->Init("127.0.0.1", server_.listen_address().port, nullptr), 0);
  }

  FirstEventRecoveryQuery make_query(std::string request_uid,
                                     uint64_t attempt_seq = 7,
                                     uint64_t timeout_ms = 200) const {
    FirstEventRecoveryQuery query;
    query.channel = channel_;
    query.attempt.set_request_uid(std::move(request_uid));
    query.attempt.set_attempt_seq(attempt_seq);
    query.decode.set_engine_uid("decode-1");
    query.decode.set_incarnation_id("decode-incarnation-1");
    query.prefill.set_engine_uid("prefill-1");
    query.prefill.set_incarnation_id("prefill-incarnation-1");
    query.max_payload_bytes = kMaxPayloadBytes;
    query.timeout_ms = timeout_ms;
    return query;
  }

  FakeDecodeService service_;
  brpc::Server server_;
  std::shared_ptr<brpc::Channel> channel_;
};

TEST_F(FirstEventRecoveryClientTest, RecoversExactFirstEventOverLoopback) {
  std::vector<FirstEventRecoveryResult> results =
      query_first_output_events({make_query("request-1")});

  ASSERT_EQ(results.size(), 1u);
  ASSERT_TRUE(results[0].status.ok()) << results[0].status.message();
  ASSERT_TRUE(results[0].output.has_value());
  EXPECT_EQ(results[0].output->service_request_id, "request-1");
  EXPECT_EQ(results[0].output->outputs[0].text, "exact-first-event");
  EXPECT_EQ(results[0].output->outputs[0].token_ids,
            (std::vector<int32_t>{101}));
  EXPECT_EQ(service_.query_count(), 1);
}

TEST_F(FirstEventRecoveryClientTest, IssuesBatchQueriesConcurrently) {
  service_.set_delay_us(40000);
  std::vector<FirstEventRecoveryQuery> queries;
  for (size_t index = 0; index < 8; ++index) {
    queries.emplace_back(make_query("request-" + std::to_string(index)));
  }

  const std::vector<FirstEventRecoveryResult> results =
      query_first_output_events(queries);

  ASSERT_EQ(results.size(), queries.size());
  for (const FirstEventRecoveryResult& result : results) {
    EXPECT_TRUE(result.status.ok()) << result.status.message();
    EXPECT_TRUE(result.output.has_value());
  }
  EXPECT_EQ(service_.query_count(), 8);
  EXPECT_GE(service_.peak_in_flight(), 2);
}

TEST_F(FirstEventRecoveryClientTest, ReturnsDeadlineExceededOnRpcTimeout) {
  service_.set_delay_us(80000);

  const std::vector<FirstEventRecoveryResult> results =
      query_first_output_events({make_query("request-timeout", 7, 5)});

  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].status.code(), llm::StatusCode::DEADLINE_EXCEEDED);
  EXPECT_FALSE(results[0].output.has_value());
}

TEST_F(FirstEventRecoveryClientTest, RejectsResponseFromRestartedDecode) {
  std::vector<FirstEventRecoveryResult> before_restart =
      query_first_output_events({make_query("request-before-restart")});
  ASSERT_EQ(before_restart.size(), 1u);
  ASSERT_TRUE(before_restart[0].status.ok());
  restart_decode("decode-incarnation-2");

  const std::vector<FirstEventRecoveryResult> first_results =
      query_first_output_events({make_query("request-restart")});

  ASSERT_EQ(first_results.size(), 1u);
  EXPECT_FALSE(first_results[0].status.ok());
  EXPECT_FALSE(first_results[0].output.has_value());
  reconnect_channel();
  const std::vector<FirstEventRecoveryResult> reconnected_results =
      query_first_output_events({make_query("request-after-reconnect")});
  ASSERT_EQ(reconnected_results.size(), 1u);
  EXPECT_EQ(reconnected_results[0].status.code(),
            llm::StatusCode::INVALID_ARGUMENT);
  EXPECT_FALSE(reconnected_results[0].output.has_value());
}

TEST_F(FirstEventRecoveryClientTest, RecoveryWinsRaceWithLiveFirstEvent) {
  std::vector<FirstEventRecoveryResult> results =
      query_first_output_events({make_query("request-race")});
  ASSERT_EQ(results.size(), 1u);
  ASSERT_TRUE(results[0].status.ok());
  ASSERT_TRUE(results[0].output.has_value());

  OutputEventSequencer sequencer(OutputEventSequencer::Config{
      .max_buffered_events = 8,
      .max_buffered_bytes = 4096,
      .max_sequence_gap = 8,
  });
  llm::RequestOutput live_second = *results[0].output;
  live_second.output_event_seq = 1;
  live_second.finished_on_prefill_instance = false;
  live_second.sender_engine_uid = "decode-1";
  live_second.sender_incarnation_id = "decode-incarnation-1";
  EXPECT_EQ(sequencer.push(live_second, true).status,
            OutputEventSequenceStatus::kBuffered);

  OutputEventSequenceResult recovered =
      sequencer.push(*results[0].output, true);
  ASSERT_EQ(recovered.status, OutputEventSequenceStatus::kReady);
  ASSERT_EQ(recovered.ready_outputs.size(), 2u);
  EXPECT_EQ(recovered.ready_outputs[0].output_event_seq, 0);
  EXPECT_EQ(recovered.ready_outputs[1].output_event_seq, 1);

  OutputEventSequenceResult late_live_first =
      sequencer.push(*results[0].output, true);
  EXPECT_EQ(late_live_first.status, OutputEventSequenceStatus::kDuplicate);
  EXPECT_TRUE(late_live_first.ready_outputs.empty());
}

TEST(FirstEventRecoveryClientValidationTest, RejectsWithoutNetworkSideEffects) {
  FirstEventRecoveryQuery no_channel;
  no_channel.timeout_ms = 10;
  FirstEventRecoveryQuery no_time;
  no_time.channel = std::make_shared<brpc::Channel>();

  const std::vector<FirstEventRecoveryResult> results =
      query_first_output_events({no_channel, no_time});

  ASSERT_EQ(results.size(), 2u);
  EXPECT_EQ(results[0].status.code(), llm::StatusCode::UNAVAILABLE);
  EXPECT_EQ(results[1].status.code(), llm::StatusCode::DEADLINE_EXCEEDED);
}

TEST(FirstEventRecoveryClientValidationTest, RejectsInvalidBoundaries) {
  FirstEventRecoveryQuery incomplete;
  incomplete.channel = std::make_shared<brpc::Channel>();
  incomplete.timeout_ms = 10;
  incomplete.max_payload_bytes = kMaxPayloadBytes;

  FirstEventRecoveryQuery overflowing_timeout = incomplete;
  overflowing_timeout.attempt.set_request_uid("request-overflow");
  overflowing_timeout.attempt.set_attempt_seq(1);
  overflowing_timeout.decode.set_engine_uid("decode-1");
  overflowing_timeout.decode.set_incarnation_id("decode-incarnation-1");
  overflowing_timeout.prefill.set_engine_uid("prefill-1");
  overflowing_timeout.prefill.set_incarnation_id("prefill-incarnation-1");
  overflowing_timeout.timeout_ms =
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) + 1;

  const std::vector<FirstEventRecoveryResult> results =
      query_first_output_events({incomplete, overflowing_timeout});

  ASSERT_EQ(results.size(), 2u);
  EXPECT_EQ(results[0].status.code(), llm::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(results[1].status.code(), llm::StatusCode::INVALID_ARGUMENT);
  EXPECT_TRUE(query_first_output_events({}).empty());
}

}  // namespace
}  // namespace xllm_service
