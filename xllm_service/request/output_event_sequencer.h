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

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include "common/xllm/output.h"

namespace xllm_service {

enum class OutputEventSequenceStatus {
  kReady = 0,
  kBuffered,
  kDuplicate,
  kMissingSequence,
  kGapTooLarge,
  kCapacityExceeded,
  kTerminalConflict,
  kInvalidSequence,
  kClosed,
};

struct OutputEventSequenceResult {
  OutputEventSequenceStatus status = OutputEventSequenceStatus::kReady;
  std::vector<llm::RequestOutput> ready_outputs;
};

struct RemotePdOutputBinding {
  std::optional<uint64_t> attempt_seq;
  std::string prefill_engine_uid;
  std::string prefill_incarnation_id;
  std::string decode_engine_uid;
  std::string decode_incarnation_id;
};

bool matches_remote_pd_output_identity(const llm::RequestOutput& output,
                                       const RemotePdOutputBinding& binding);

// Per-request, in-memory ordering for output arriving from multiple senders.
// The owning Request serializes push() with dispatch to its affinity thread.
class OutputEventSequencer final {
 public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  struct Config {
    size_t max_buffered_events = 64;
    size_t max_buffered_bytes = 4 * 1024 * 1024;
    uint64_t max_sequence_gap = 64;
  };

  explicit OutputEventSequencer(Config config);

  OutputEventSequenceResult push(llm::RequestOutput output,
                                 bool require_sequence,
                                 TimePoint now = Clock::now());

  bool gap_expired(TimePoint now, std::chrono::milliseconds timeout) const;
  void close() { closed_ = true; }

  uint64_t next_expected_seq() const { return next_expected_seq_; }
  size_t buffered_events() const { return buffered_.size(); }
  size_t buffered_bytes() const { return buffered_bytes_; }

 private:
  struct BufferedOutput {
    llm::RequestOutput output;
    size_t bytes = 0;
  };

  static bool terminal_output(const llm::RequestOutput& output);
  static size_t estimated_bytes(const llm::RequestOutput& output);

  Config config_;
  uint64_t next_expected_seq_ = 0;
  size_t buffered_bytes_ = 0;
  bool closed_ = false;
  std::optional<uint64_t> terminal_seq_;
  std::optional<TimePoint> gap_started_at_;
  std::map<uint64_t, BufferedOutput> buffered_;
};

}  // namespace xllm_service
