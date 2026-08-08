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

#include "request/output_event_sequencer.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace xllm_service {
namespace {

size_t saturated_add(size_t left, size_t right) {
  if (right > std::numeric_limits<size_t>::max() - left) {
    return std::numeric_limits<size_t>::max();
  }
  return left + right;
}

size_t saturated_multiply(size_t left, size_t right) {
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
    return std::numeric_limits<size_t>::max();
  }
  return left * right;
}

void add_string_capacity(size_t* bytes, const std::string& value) {
  *bytes = saturated_add(*bytes, value.capacity());
}

}  // namespace

bool matches_remote_pd_output_identity(const llm::RequestOutput& output,
                                       const RemotePdOutputBinding& binding) {
  if (!output.attempt_seq.has_value() || !binding.attempt_seq.has_value() ||
      output.attempt_seq != binding.attempt_seq) {
    return false;
  }
  const bool matches_prefill =
      output.sender_engine_uid == binding.prefill_engine_uid &&
      output.sender_incarnation_id == binding.prefill_incarnation_id;
  const bool matches_decode =
      output.sender_engine_uid == binding.decode_engine_uid &&
      output.sender_incarnation_id == binding.decode_incarnation_id;
  const bool status_error = output.status.has_value() && !output.status->ok();
  if (status_error) {
    return matches_prefill || matches_decode;
  }
  return output.finished_on_prefill_instance ? matches_prefill : matches_decode;
}

OutputEventSequencer::OutputEventSequencer(Config config) : config_(config) {
  if (config_.max_buffered_events == 0 || config_.max_buffered_bytes == 0 ||
      config_.max_sequence_gap == 0) {
    throw std::invalid_argument("invalid output event reorder capacity");
  }
}

bool OutputEventSequencer::terminal_output(const llm::RequestOutput& output) {
  return output.finished || (output.status.has_value() && !output.status->ok());
}

size_t OutputEventSequencer::estimated_bytes(const llm::RequestOutput& output) {
  size_t bytes = sizeof(output);
  add_string_capacity(&bytes, output.request_id);
  add_string_capacity(&bytes, output.service_request_id);
  add_string_capacity(&bytes, output.sender_engine_uid);
  add_string_capacity(&bytes, output.sender_incarnation_id);
  if (output.prompt.has_value()) {
    add_string_capacity(&bytes, *output.prompt);
  }
  if (output.status.has_value()) {
    add_string_capacity(&bytes, output.status->message());
  }
  bytes = saturated_add(bytes,
                        saturated_multiply(output.outputs.capacity(),
                                           sizeof(llm::SequenceOutput)));
  for (const llm::SequenceOutput& sequence : output.outputs) {
    add_string_capacity(&bytes, sequence.text);
    bytes = saturated_add(
        bytes,
        saturated_multiply(sequence.token_ids.capacity(), sizeof(int32_t)));
    if (sequence.finish_reason.has_value()) {
      add_string_capacity(&bytes, *sequence.finish_reason);
    }
    if (!sequence.logprobs.has_value()) {
      continue;
    }
    bytes = saturated_add(bytes,
                          saturated_multiply(sequence.logprobs->capacity(),
                                             sizeof(llm::LogProb)));
    for (const llm::LogProb& logprob : *sequence.logprobs) {
      add_string_capacity(&bytes, logprob.token);
      if (!logprob.top_logprobs.has_value()) {
        continue;
      }
      bytes = saturated_add(bytes,
                            saturated_multiply(logprob.top_logprobs->capacity(),
                                               sizeof(llm::LogProbData)));
      for (const llm::LogProbData& top_logprob : *logprob.top_logprobs) {
        add_string_capacity(&bytes, top_logprob.token);
      }
    }
  }
  return bytes;
}

OutputEventSequenceResult OutputEventSequencer::push(llm::RequestOutput output,
                                                     bool require_sequence,
                                                     TimePoint now) {
  if (closed_) {
    return {.status = OutputEventSequenceStatus::kClosed};
  }
  if (!output.output_event_seq.has_value()) {
    if (require_sequence) {
      return {.status = OutputEventSequenceStatus::kMissingSequence};
    }
    OutputEventSequenceResult result;
    result.ready_outputs.emplace_back(std::move(output));
    return result;
  }

  const uint64_t sequence = *output.output_event_seq;
  if (sequence == std::numeric_limits<uint64_t>::max()) {
    return {.status = OutputEventSequenceStatus::kInvalidSequence};
  }
  if (sequence < next_expected_seq_) {
    return {.status = OutputEventSequenceStatus::kDuplicate};
  }
  if (terminal_seq_.has_value() && sequence > *terminal_seq_) {
    return {.status = OutputEventSequenceStatus::kTerminalConflict};
  }
  const bool terminal = terminal_output(output);
  if (terminal && terminal_seq_.has_value() && sequence != *terminal_seq_) {
    return {.status = OutputEventSequenceStatus::kTerminalConflict};
  }
  if (terminal && buffered_.upper_bound(sequence) != buffered_.end()) {
    return {.status = OutputEventSequenceStatus::kTerminalConflict};
  }
  if (sequence - next_expected_seq_ > config_.max_sequence_gap) {
    return {.status = OutputEventSequenceStatus::kGapTooLarge};
  }
  if (buffered_.find(sequence) != buffered_.end()) {
    return {.status = OutputEventSequenceStatus::kDuplicate};
  }

  if (sequence != next_expected_seq_) {
    const size_t bytes = estimated_bytes(output);
    if (buffered_.size() >= config_.max_buffered_events ||
        bytes > config_.max_buffered_bytes - buffered_bytes_) {
      return {.status = OutputEventSequenceStatus::kCapacityExceeded};
    }
    if (terminal) {
      terminal_seq_ = sequence;
    }
    if (!gap_started_at_.has_value()) {
      gap_started_at_ = now;
    }
    buffered_bytes_ += bytes;
    buffered_.emplace(
        sequence, BufferedOutput{.output = std::move(output), .bytes = bytes});
    return {.status = OutputEventSequenceStatus::kBuffered};
  }

  if (terminal) {
    terminal_seq_ = sequence;
  }
  OutputEventSequenceResult result;
  result.ready_outputs.emplace_back(std::move(output));
  ++next_expected_seq_;
  while (true) {
    auto it = buffered_.find(next_expected_seq_);
    if (it == buffered_.end()) {
      break;
    }
    buffered_bytes_ -= it->second.bytes;
    result.ready_outputs.emplace_back(std::move(it->second.output));
    buffered_.erase(it);
    ++next_expected_seq_;
  }
  if (buffered_.empty()) {
    gap_started_at_.reset();
  }
  return result;
}

bool OutputEventSequencer::gap_expired(
    TimePoint now,
    std::chrono::milliseconds timeout) const {
  return !buffered_.empty() && gap_started_at_.has_value() &&
         now >= *gap_started_at_ && now - *gap_started_at_ >= timeout;
}

}  // namespace xllm_service
