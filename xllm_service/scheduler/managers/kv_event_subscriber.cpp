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

#include "kv_event_subscriber.h"

#include <glog/logging.h>
#include <zmq.hpp>

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include "common/metrics.h"

namespace xllm_service {
namespace {

std::string message_to_string(const zmq::message_t& msg) {
  return std::string(static_cast<const char*>(msg.data()), msg.size());
}

}  // namespace

KvEventSubscriber::KvEventSubscriber(Options options)
    : options_(std::move(options)) {}

KvEventSubscriber::~KvEventSubscriber() { stop(); }

bool KvEventSubscriber::start() {
  if (!options_.enabled()) {
    return true;
  }
  if (started_.exchange(true)) {
    return true;
  }
  exited_.store(false);
  subscriber_thread_ =
      std::make_unique<std::thread>(&KvEventSubscriber::run_loop, this);
  return true;
}

void KvEventSubscriber::stop() {
  exited_.store(true);
  if (subscriber_thread_ && subscriber_thread_->joinable()) {
    subscriber_thread_->join();
  }
  subscriber_thread_.reset();
  started_.store(false);
}

void KvEventSubscriber::add_or_update_source(const InstanceMetaInfo& info) {
  if (!options_.enabled() || info.zmq_endpoint.empty()) {
    return;
  }
  Command command;
  command.type = CommandType::ADD_OR_UPDATE;
  command.info = info;
  enqueue(std::move(command));
}

void KvEventSubscriber::remove_source(const std::string& instance_name,
                                      const std::string& incarnation_id) {
  if (!options_.enabled() || instance_name.empty()) {
    return;
  }
  Command command;
  command.type = CommandType::REMOVE;
  command.instance_name = instance_name;
  command.incarnation_id = incarnation_id;
  enqueue(std::move(command));
}

nlohmann::json KvEventSubscriber::debug_summary() const {
  nlohmann::json summary;
  nlohmann::json sources_json = nlohmann::json::array();

  std::lock_guard<std::mutex> lock(mutex_);
  summary["enabled"] = options_.enabled();
  summary["started"] = started_.load();
  summary["source_count"] = sources_.size();
  summary["received_events"] = received_events_;
  summary["seq_gap_events"] = seq_gap_events_;
  summary["stale_events"] = stale_events_;
  for (const auto& [name, source] : sources_) {
    sources_json.push_back({{"name", name},
                            {"endpoint", source.endpoint},
                            {"incarnation_id", source.incarnation_id},
                            {"last_seq_no", source.last_seq_no},
                            {"has_seq", source.has_seq},
                            {"suspect", source.suspect}});
  }
  summary["sources"] = std::move(sources_json);
  return summary;
}

void KvEventSubscriber::enqueue(Command command) {
  std::lock_guard<std::mutex> lock(mutex_);
  commands_.push_back(std::move(command));
}

void KvEventSubscriber::run_loop() {
  zmq::context_t context(1);
  zmq::socket_t subscriber(context, zmq::socket_type::sub);
  subscriber.set(zmq::sockopt::linger, 0);
  subscriber.set(zmq::sockopt::reconnect_ivl,
                 std::max<int32_t>(1, options_.reconnect_interval_ms()));
  subscriber.set(
      zmq::sockopt::reconnect_ivl_max,
      std::max<int32_t>(1, options_.reconnect_interval_max_ms()));
  subscriber.set(zmq::sockopt::subscribe, "");

  const auto poll_interval = std::chrono::milliseconds(
      std::max<int32_t>(1, options_.poll_interval_ms()));

  while (!exited_.load()) {
    std::vector<ReceivedEvent> received_events;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      while (!commands_.empty()) {
        Command command = std::move(commands_.front());
        commands_.pop_front();

        if (command.type == CommandType::REMOVE) {
          auto source_it = sources_.find(command.instance_name);
          if (source_it == sources_.end()) {
            continue;
          }
          if (!command.incarnation_id.empty() &&
              source_it->second.incarnation_id != command.incarnation_id) {
            continue;
          }
          try {
            subscriber.disconnect(source_it->second.endpoint);
          } catch (const zmq::error_t& e) {
            LOG(WARNING) << "Failed to disconnect KV event source, instance: "
                         << command.instance_name
                         << ", endpoint: " << source_it->second.endpoint
                         << ", error: " << e.what();
          }
          sources_.erase(source_it);
          GAUGE_SET(kv_event_zmq_source_count, sources_.size());
          if (options_.clear_callback()) {
            options_.clear_callback()(command.instance_name);
          }
          continue;
        }

        const auto& info = command.info;
        if (info.name.empty() || info.zmq_endpoint.empty()) {
          continue;
        }

        auto source_it = sources_.find(info.name);
        if (source_it != sources_.end()) {
          if (source_it->second.endpoint == info.zmq_endpoint &&
              source_it->second.incarnation_id == info.incarnation_id) {
            continue;
          }
          try {
            subscriber.disconnect(source_it->second.endpoint);
          } catch (const zmq::error_t& e) {
            LOG(WARNING) << "Failed to disconnect old KV event source, "
                         << "instance: " << info.name
                         << ", endpoint: " << source_it->second.endpoint
                         << ", error: " << e.what();
          }
          sources_.erase(source_it);
          GAUGE_SET(kv_event_zmq_source_count, sources_.size());
          if (options_.clear_callback()) {
            options_.clear_callback()(info.name);
          }
        }

        try {
          subscriber.connect(info.zmq_endpoint);
        } catch (const zmq::error_t& e) {
          LOG(ERROR) << "Failed to connect KV event source, instance: "
                     << info.name << ", endpoint: " << info.zmq_endpoint
                     << ", error: " << e.what();
          continue;
        }

        SourceState source;
        source.endpoint = info.zmq_endpoint;
        source.incarnation_id = info.incarnation_id;
        sources_.insert_or_assign(info.name, std::move(source));
        GAUGE_SET(kv_event_zmq_source_count, sources_.size());
        LOG(INFO) << "Connected KV event source, instance: " << info.name
                  << ", endpoint: " << info.zmq_endpoint
                  << ", incarnation_id: " << info.incarnation_id;
      }

      while (!exited_.load()) {
        zmq::message_t topic_msg;
        auto topic_result =
            subscriber.recv(topic_msg, zmq::recv_flags::dontwait);
        if (!topic_result.has_value()) {
          break;
        }

        zmq::message_t body_msg;
        auto body_result =
            subscriber.recv(body_msg, zmq::recv_flags::dontwait);
        if (!body_result.has_value()) {
          LOG(ERROR) << "Received KV event topic without payload.";
          COUNTER_INC(kv_event_zmq_parse_failure_total);
          continue;
        }

        const std::string instance_name = message_to_string(topic_msg);
        proto::KvCacheEventEnvelope envelope;
        if (!envelope.ParseFromArray(body_msg.data(),
                                     static_cast<int>(body_msg.size()))) {
          LOG(ERROR) << "Failed to parse KV event envelope, instance: "
                     << instance_name;
          COUNTER_INC(kv_event_zmq_parse_failure_total);
          continue;
        }

        auto source_it = sources_.find(instance_name);
        if (source_it == sources_.end()) {
          ++stale_events_;
          COUNTER_INC(kv_event_zmq_stale_total);
          continue;
        }

        SourceState& source = source_it->second;
        if (source.incarnation_id != envelope.incarnation_id()) {
          ++stale_events_;
          COUNTER_INC(kv_event_zmq_stale_total);
          LOG(WARNING) << "Ignore stale KV event, instance: " << instance_name
                       << ", current incarnation_id: " << source.incarnation_id
                       << ", event incarnation_id: "
                       << envelope.incarnation_id();
          continue;
        }

        if (source.has_seq && envelope.seq_no() != source.last_seq_no + 1) {
          ++seq_gap_events_;
          source.suspect = true;
          COUNTER_INC(kv_event_zmq_gap_total);
          LOG(WARNING) << "KV event sequence gap, instance: " << instance_name
                       << ", last_seq_no: " << source.last_seq_no
                       << ", received_seq_no: " << envelope.seq_no();
        }
        source.last_seq_no = envelope.seq_no();
        source.has_seq = true;
        ++received_events_;
        COUNTER_INC(kv_event_zmq_received_total);

        ReceivedEvent event;
        event.instance_name = instance_name;
        event.cache_event = envelope.cache_event();
        received_events.emplace_back(std::move(event));
      }
    }

    for (const auto& event : received_events) {
      if (options_.record_callback()) {
        options_.record_callback()(event.instance_name, event.cache_event);
      }
    }

    std::this_thread::sleep_for(poll_interval);
  }

  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& [name, source] : sources_) {
    try {
      subscriber.disconnect(source.endpoint);
    } catch (const zmq::error_t& e) {
      LOG(WARNING) << "Failed to disconnect KV event source during shutdown, "
                   << "instance: " << name << ", endpoint: " << source.endpoint
                   << ", error: " << e.what();
    }
  }
  sources_.clear();
  GAUGE_SET(kv_event_zmq_source_count, 0);
  subscriber.close();
  context.close();
}

}  // namespace xllm_service
