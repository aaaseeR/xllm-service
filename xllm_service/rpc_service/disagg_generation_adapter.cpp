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

#include "rpc_service/disagg_generation_adapter.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace xllm_service {
namespace {

RequestOutputConversionResult invalid_usage(std::string message) {
  return {llm::Status(llm::StatusCode::INVALID_ARGUMENT, std::move(message)),
          std::nullopt};
}

RequestOutputConversionResult validate_usage(const proto::OutputUsage& usage) {
  if (usage.num_prompt_tokens() < 0 || usage.num_generated_tokens() < 0 ||
      usage.num_total_tokens() < 0 || usage.num_cached_tokens() < 0) {
    return invalid_usage("token counts must be non-negative");
  }
  if (usage.num_cached_tokens() > usage.num_prompt_tokens()) {
    return invalid_usage(
        "prefix cache hit tokens must not exceed prompt tokens");
  }
  const int64_t expected_total =
      static_cast<int64_t>(usage.num_prompt_tokens()) +
      static_cast<int64_t>(usage.num_generated_tokens());
  if (static_cast<int64_t>(usage.num_total_tokens()) != expected_total) {
    return invalid_usage(
        "total tokens must equal prompt tokens plus generated tokens");
  }
  return {};
}

bool recoverable_first_event_state(xllm::proto::AttemptLifecycleState state) {
  return state == xllm::proto::ATTEMPT_LIFECYCLE_STATE_GENERATION_COMMITTED ||
         state == xllm::proto::ATTEMPT_LIFECYCLE_STATE_RUNNING ||
         state == xllm::proto::ATTEMPT_LIFECYCLE_STATE_DONE;
}

}  // namespace

RequestOutputConversionResult request_output_from_disagg_generation(
    const proto::DisaggStreamGeneration& generation) {
  if (generation.has_usage()) {
    RequestOutputConversionResult validation =
        validate_usage(generation.usage());
    if (!validation.status.ok()) {
      return validation;
    }
  }

  llm::RequestOutput request_output;
  request_output.request_id = generation.req_id();
  request_output.service_request_id = generation.service_req_id();
  if (generation.has_gen_status()) {
    request_output.status = llm::Status(
        static_cast<llm::StatusCode>(generation.gen_status().status_code()),
        generation.gen_status().status_msg());
  }
  if (generation.has_usage()) {
    llm::Usage usage;
    usage.num_prompt_tokens =
        static_cast<size_t>(generation.usage().num_prompt_tokens());
    usage.num_generated_tokens =
        static_cast<size_t>(generation.usage().num_generated_tokens());
    usage.num_total_tokens =
        static_cast<size_t>(generation.usage().num_total_tokens());
    usage.num_cached_tokens =
        static_cast<size_t>(generation.usage().num_cached_tokens());
    request_output.usage = std::move(usage);
  }
  request_output.finished_on_prefill_instance =
      generation.finished_on_prefill_instance();
  request_output.finished = generation.finished();
  if (generation.has_output_event_seq()) {
    request_output.output_event_seq = generation.output_event_seq();
  }
  if (generation.has_attempt_seq()) {
    request_output.attempt_seq = generation.attempt_seq();
  }
  request_output.sender_engine_uid = generation.sender_engine_uid();
  request_output.sender_incarnation_id = generation.sender_incarnation_id();
  request_output.outputs.reserve(generation.outputs_size());
  for (const proto::SequenceOutput& output : generation.outputs()) {
    llm::SequenceOutput sequence_output;
    sequence_output.index = static_cast<size_t>(output.index());
    sequence_output.text = output.text();
    sequence_output.token_ids = std::vector<int32_t>(output.token_ids().begin(),
                                                     output.token_ids().end());
    if (!output.finish_reason().empty()) {
      sequence_output.finish_reason = output.finish_reason();
    }
    if (!output.logprobs().empty()) {
      std::vector<llm::LogProb> logprobs;
      logprobs.reserve(output.logprobs_size());
      for (const proto::LogProb& logprob : output.logprobs()) {
        llm::LogProb converted_logprob;
        converted_logprob.token = logprob.log_prob_data().token();
        converted_logprob.token_id = logprob.log_prob_data().token_id();
        converted_logprob.logprob = logprob.log_prob_data().logprob();
        converted_logprob.finished_token =
            logprob.log_prob_data().finished_token();
        if (!logprob.top_logprobs().empty()) {
          std::vector<llm::LogProbData> top_logprobs;
          top_logprobs.reserve(logprob.top_logprobs_size());
          for (const proto::LogProbData& top_logprob : logprob.top_logprobs()) {
            llm::LogProbData converted_top_logprob;
            converted_top_logprob.token = top_logprob.token();
            converted_top_logprob.token_id = top_logprob.token_id();
            converted_top_logprob.logprob = top_logprob.logprob();
            converted_top_logprob.finished_token = top_logprob.finished_token();
            top_logprobs.emplace_back(std::move(converted_top_logprob));
          }
          converted_logprob.top_logprobs = std::move(top_logprobs);
        }
        logprobs.emplace_back(std::move(converted_logprob));
      }
      sequence_output.logprobs = std::move(logprobs);
    }
    request_output.outputs.emplace_back(std::move(sequence_output));
  }
  return {llm::Status(), std::move(request_output)};
}

RequestOutputConversionResult first_output_from_query_response(
    const xllm::proto::AttemptControlResponse& response,
    const xllm::proto::ExecutionAttemptId& expected_attempt,
    const xllm::proto::ExecutionHolder& expected_decode,
    const xllm::proto::ExecutionHolder& expected_prefill,
    size_t max_payload_bytes) {
  if (!response.ok() || !response.has_status() ||
      expected_attempt.request_uid().empty() ||
      !expected_attempt.has_attempt_seq() ||
      expected_decode.engine_uid().empty() ||
      expected_decode.incarnation_id().empty() ||
      expected_prefill.engine_uid().empty() ||
      expected_prefill.incarnation_id().empty() || max_payload_bytes == 0) {
    return invalid_usage("query response does not match the execution hold");
  }
  const xllm::proto::AttemptStatus& status = response.status();
  const xllm::proto::RequestAttemptKey& key = status.key();
  if (!recoverable_first_event_state(status.state()) ||
      key.request_uid() != expected_attempt.request_uid() ||
      !key.has_attempt_seq() ||
      key.attempt_seq() != expected_attempt.attempt_seq() ||
      key.incarnation_id() != expected_decode.incarnation_id()) {
    return invalid_usage("query response does not match the execution hold");
  }

  proto::DisaggStreamGeneration generation;
  if (status.first_event_payload().empty() ||
      status.first_event_payload().size() > max_payload_bytes ||
      !generation.ParseFromString(status.first_event_payload()) ||
      generation.req_id().empty() ||
      generation.service_req_id() != expected_attempt.request_uid() ||
      !generation.has_output_event_seq() ||
      generation.output_event_seq() != 0 || !generation.has_attempt_seq() ||
      generation.attempt_seq() != expected_attempt.attempt_seq() ||
      generation.sender_engine_uid() != expected_prefill.engine_uid() ||
      generation.sender_incarnation_id() != expected_prefill.incarnation_id() ||
      !generation.finished_on_prefill_instance() ||
      generation.outputs().empty()) {
    return invalid_usage("query response has no valid canonical seq=0 event");
  }
  for (const proto::SequenceOutput& output : generation.outputs()) {
    if (output.index() < 0) {
      return invalid_usage("query response has an invalid output index");
    }
  }
  return request_output_from_disagg_generation(generation);
}

}  // namespace xllm_service
