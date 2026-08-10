/* Copyright 2025-2026 The xLLM Authors.

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

#include <brpc/channel.h>
#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <brpc/server.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "completion.pb.h"
#include "disagg_pd.pb.h"
#include "provider.pb.h"
#include "xllm_rpc_service.pb.h"
#include "xllm_service.pb.h"

DEFINE_string(listen_address,
              "127.0.0.1:19000",
              "Dialable address of the mock Native Engine.");
DEFINE_string(engine_uid, "", "Immutable logical Engine UID.");
DEFINE_string(incarnation_id, "", "Immutable Engine process incarnation.");
DEFINE_int32(worker_threads, 8, "Bounded async completion worker count.");
DEFINE_int32(queue_capacity, 4096, "Bounded pending completion queue.");
DEFINE_int32(completion_delay_ms, 2, "Simulated CPU execution duration.");

namespace {

std::atomic<bool> g_stop{false};

void handle_signal(int) { g_stop.store(true, std::memory_order_release); }

struct SelectedDecode {
  std::string engine_uid;
  std::string incarnation_id;
};

bool selected_decode(const xllm::proto::CompletionRequest& request,
                     SelectedDecode* decode) {
  if (decode == nullptr || !request.has_execution_plan() ||
      request.execution_plan().mode() !=
          xllm::proto::EXECUTION_MODE_REMOTE_PD ||
      request.execution_plan().request_uid() !=
          request.correlation().request_uid()) {
    return false;
  }
  for (const auto& role : request.execution_plan().selected_roles()) {
    if (role.role() == xllm::proto::ENGINE_ROLE_DECODE) {
      decode->engine_uid = role.engine_uid();
      decode->incarnation_id = role.incarnation_id();
      return !decode->engine_uid.empty() && !decode->incarnation_id.empty();
    }
  }
  return false;
}

xllm_service::proto::DisaggStreamGeneration make_generation(
    const xllm::proto::CompletionRequest& request,
    uint64_t output_event_seq,
    const std::string& sender_uid,
    const std::string& sender_incarnation,
    bool finished_on_prefill,
    bool finished) {
  xllm_service::proto::DisaggStreamGeneration generation;
  generation.set_req_id(request.request_id());
  generation.set_service_req_id(request.correlation().request_uid());
  generation.set_output_event_seq(output_event_seq);
  generation.set_attempt_seq(request.correlation().attempt_seq());
  generation.set_sender_engine_uid(sender_uid);
  generation.set_sender_incarnation_id(sender_incarnation);
  generation.set_finished_on_prefill_instance(finished_on_prefill);
  generation.set_finished(finished);

  auto* usage = generation.mutable_usage();
  usage->set_num_prompt_tokens(static_cast<int32_t>(request.token_ids_size()));
  usage->set_num_generated_tokens(finished ? 1 : 0);
  usage->set_num_total_tokens(usage->num_prompt_tokens() +
                              usage->num_generated_tokens());
  usage->set_num_cached_tokens(0);
  usage->set_num_decode_cached_tokens(0);

  generation.set_decode_admission_disposition(
      xllm::proto::ADMISSION_DISPOSITION_ACCEPTED);
  generation.set_decode_admission_reason(xllm::proto::ADMISSION_REASON_NONE);
  generation.set_decode_admission_attempts(1);
  generation.set_decode_admission_rpc_duration_ns(100000);

  if (finished) {
    auto* output = generation.add_outputs();
    output->set_index(0);
    output->set_text("mock-native route=" + FLAGS_engine_uid + "->" +
                     sender_uid);
    output->add_token_ids(42);
    output->set_finish_reason("stop");
  }
  return generation;
}

bool deliver(const std::string& service_address,
             const xllm_service::proto::DisaggStreamGeneration& generation) {
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.connect_timeout_ms = 500;
  options.timeout_ms = 2000;
  if (channel.Init(service_address.c_str(), &options) != 0) {
    LOG(ERROR) << "mock_native_callback_channel_failed service_address="
               << service_address;
    return false;
  }

  xllm_service::proto::DisaggStreamGenerations request;
  *request.add_gens() = generation;
  xllm_service::proto::StatusSet response;
  brpc::Controller controller;
  xllm_service::proto::XllmRpcService_Stub stub(&channel);
  stub.Generations(&controller, &request, &response, nullptr);
  if (controller.Failed() || response.all_status_size() != 1 ||
      !response.all_status(0).ok()) {
    LOG(WARNING) << "mock_native_callback_rejected request_uid="
                 << generation.service_req_id()
                 << " output_event_seq=" << generation.output_event_seq()
                 << " transport_error=" << controller.ErrorText()
                 << " response=" << response.ShortDebugString();
    return false;
  }
  return true;
}

class CompletionWorkers {
 public:
  CompletionWorkers(size_t threads, size_t capacity) : capacity_(capacity) {
    workers_.reserve(threads);
    for (size_t index = 0; index < threads; ++index) {
      workers_.emplace_back([this]() { run(); });
    }
  }

  ~CompletionWorkers() { stop(); }

  bool submit(const xllm::proto::CompletionRequest& request) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (stopping_ || pending_.size() >= capacity_) {
      rejected_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    pending_.push_back(request);
    high_watermark_.store(
        std::max<uint64_t>(high_watermark_.load(std::memory_order_relaxed),
                           pending_.size()),
        std::memory_order_relaxed);
    condition_.notify_one();
    return true;
  }

  void stop() {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (stopping_) {
        return;
      }
      stopping_ = true;
    }
    condition_.notify_all();
    for (std::thread& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
    LOG(INFO) << "mock_native_engine_summary engine_uid=" << FLAGS_engine_uid
              << " accepted=" << accepted_.load()
              << " completed=" << completed_.load()
              << " callback_failures=" << callback_failures_.load()
              << " rejected=" << rejected_.load()
              << " queue_high_watermark=" << high_watermark_.load();
  }

 private:
  void run() {
    while (true) {
      xllm::proto::CompletionRequest request;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock,
                        [this]() { return stopping_ || !pending_.empty(); });
        if (pending_.empty()) {
          if (stopping_) {
            return;
          }
          continue;
        }
        request = std::move(pending_.front());
        pending_.pop_front();
      }
      accepted_.fetch_add(1, std::memory_order_relaxed);
      execute(request);
    }
  }

  void execute(const xllm::proto::CompletionRequest& request) {
    SelectedDecode decode;
    if (!selected_decode(request, &decode) ||
        !request.has_source_xservice_addr() ||
        request.source_xservice_addr().empty() || !request.has_correlation() ||
        !request.correlation().has_attempt_seq()) {
      LOG(ERROR) << "mock_native_invalid_v2_request request="
                 << request.ShortDebugString();
      callback_failures_.fetch_add(1, std::memory_order_relaxed);
      return;
    }

    if (FLAGS_completion_delay_ms > 0) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(FLAGS_completion_delay_ms));
    }
    const auto prefill = make_generation(
        request, 0, FLAGS_engine_uid, FLAGS_incarnation_id, true, false);
    const auto terminal = make_generation(
        request, 1, decode.engine_uid, decode.incarnation_id, false, true);

    const uint64_t ordinal = accepted_.load(std::memory_order_relaxed);
    bool ok = true;
    if (ordinal % 3 == 0) {
      ok = deliver(request.source_xservice_addr(), terminal) && ok;
      ok = deliver(request.source_xservice_addr(), prefill) && ok;
    } else {
      ok = deliver(request.source_xservice_addr(), prefill) && ok;
      if (ordinal % 3 == 1) {
        ok = deliver(request.source_xservice_addr(), prefill) && ok;
      }
      ok = deliver(request.source_xservice_addr(), terminal) && ok;
    }
    if (!ok) {
      callback_failures_.fetch_add(1, std::memory_order_relaxed);
    }
    completed_.fetch_add(1, std::memory_order_relaxed);
  }

  const size_t capacity_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<xllm::proto::CompletionRequest> pending_;
  std::vector<std::thread> workers_;
  bool stopping_ = false;
  std::atomic<uint64_t> accepted_{0};
  std::atomic<uint64_t> completed_{0};
  std::atomic<uint64_t> callback_failures_{0};
  std::atomic<uint64_t> rejected_{0};
  std::atomic<uint64_t> high_watermark_{0};
};

class MockApiService final : public xllm::proto::XllmAPIService {
 public:
  explicit MockApiService(CompletionWorkers* workers) : workers_(workers) {}

  void Completions(google::protobuf::RpcController* controller,
                   const xllm::proto::CompletionRequest* request,
                   xllm::proto::CompletionResponse*,
                   google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    if (request == nullptr || !workers_->submit(*request)) {
      controller->SetFailed("mock Native Engine completion queue is full");
    }
  }

 private:
  CompletionWorkers* workers_;
};

class MockPdService final : public xllm::proto::DisaggPDService {
 public:
  void LinkInstance(google::protobuf::RpcController*,
                    const xllm::proto::InstanceClusterInfo*,
                    xllm::proto::Status* response,
                    google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    response->set_ok(true);
  }

  void UnlinkInstance(google::protobuf::RpcController*,
                      const xllm::proto::InstanceClusterInfo*,
                      xllm::proto::Status* response,
                      google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    response->set_ok(true);
  }

  void QueryRequest(google::protobuf::RpcController*,
                    const xllm::proto::AttemptControlRequest* request,
                    xllm::proto::AttemptControlResponse* response,
                    google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    response->set_ok(true);
    response->set_reason(xllm::proto::ADMISSION_REASON_NONE);
    *response->mutable_status()->mutable_key() = request->key();
    response->mutable_status()->set_state(
        xllm::proto::ATTEMPT_LIFECYCLE_STATE_RUNNING);
    response->mutable_status()->set_reason(xllm::proto::ADMISSION_REASON_NONE);
  }

  void CancelRequest(google::protobuf::RpcController*,
                     const xllm::proto::AttemptControlRequest* request,
                     xllm::proto::AttemptControlResponse* response,
                     google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    response->set_ok(true);
    response->set_reason(xllm::proto::ADMISSION_REASON_CANCELLED);
    *response->mutable_status()->mutable_key() = request->key();
    response->mutable_status()->set_state(
        xllm::proto::ATTEMPT_LIFECYCLE_STATE_CANCELLED);
    response->mutable_status()->set_reason(
        xllm::proto::ADMISSION_REASON_CANCELLED);
  }
};

}  // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;

  if (FLAGS_engine_uid.empty() || FLAGS_incarnation_id.empty() ||
      FLAGS_worker_threads <= 0 || FLAGS_queue_capacity <= 0) {
    LOG(ERROR) << "engine_uid, incarnation_id, positive worker_threads and "
                  "queue_capacity are required";
    return 2;
  }

  CompletionWorkers workers(static_cast<size_t>(FLAGS_worker_threads),
                            static_cast<size_t>(FLAGS_queue_capacity));
  MockApiService api_service(&workers);
  MockPdService pd_service;
  brpc::Server server;
  if (server.AddService(&api_service, brpc::SERVER_DOESNT_OWN_SERVICE) != 0 ||
      server.AddService(&pd_service, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    LOG(ERROR) << "failed to add mock Native Engine services";
    return 3;
  }
  if (server.Start(FLAGS_listen_address.c_str(), nullptr) != 0) {
    LOG(ERROR) << "failed to start mock Native Engine address="
               << FLAGS_listen_address;
    return 4;
  }

  LOG(INFO) << "mock_native_engine_started engine_uid=" << FLAGS_engine_uid
            << " incarnation_id=" << FLAGS_incarnation_id
            << " listen_address=" << FLAGS_listen_address;
  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);
  while (!g_stop.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  server.Stop(0);
  server.Join();
  workers.stop();
  return 0;
}
