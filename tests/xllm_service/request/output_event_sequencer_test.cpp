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

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace xllm_service {
namespace {

llm::RequestOutput make_output(std::optional<uint64_t> sequence,
                               std::string text,
                               bool finished = false) {
  llm::RequestOutput output;
  output.service_request_id = "request-1";
  output.output_event_seq = sequence;
  output.finished = finished;
  llm::SequenceOutput sequence_output;
  sequence_output.index = 0;
  sequence_output.text = std::move(text);
  output.outputs.emplace_back(std::move(sequence_output));
  return output;
}

OutputEventSequencer::Config make_config() {
  return {.max_buffered_events = 4,
          .max_buffered_bytes = 4096,
          .max_sequence_gap = 4};
}

RemotePdOutputBinding make_binding() {
  return {.attempt_seq = 3,
          .prefill_engine_uid = "prefill-1",
          .prefill_incarnation_id = "prefill-incarnation-1",
          .decode_engine_uid = "decode-1",
          .decode_incarnation_id = "decode-incarnation-1"};
}

TEST(OutputEventSequencerTest, RemotePdIdentityFencesAttemptAndIncarnation) {
  llm::RequestOutput prefill = make_output(0, "prefill");
  prefill.attempt_seq = 3;
  prefill.sender_engine_uid = "prefill-1";
  prefill.sender_incarnation_id = "prefill-incarnation-1";
  prefill.finished_on_prefill_instance = true;
  EXPECT_TRUE(matches_remote_pd_output_identity(prefill, make_binding()));

  llm::RequestOutput decode = make_output(1, "decode");
  decode.attempt_seq = 3;
  decode.sender_engine_uid = "decode-1";
  decode.sender_incarnation_id = "decode-incarnation-1";
  EXPECT_TRUE(matches_remote_pd_output_identity(decode, make_binding()));

  decode.attempt_seq = 2;
  EXPECT_FALSE(matches_remote_pd_output_identity(decode, make_binding()));
  decode.attempt_seq = 3;
  decode.sender_incarnation_id = "old-incarnation";
  EXPECT_FALSE(matches_remote_pd_output_identity(decode, make_binding()));
  EXPECT_FALSE(
      matches_remote_pd_output_identity(prefill, RemotePdOutputBinding{}));
}

TEST(OutputEventSequencerTest, ErrorCanComeFromEitherBoundSender) {
  llm::RequestOutput error = make_output(0, "error");
  error.attempt_seq = 3;
  error.sender_engine_uid = "prefill-1";
  error.sender_incarnation_id = "prefill-incarnation-1";
  error.status = llm::Status(llm::StatusCode::UNKNOWN, "failed");
  EXPECT_TRUE(matches_remote_pd_output_identity(error, make_binding()));

  error.sender_engine_uid = "other";
  EXPECT_FALSE(matches_remote_pd_output_identity(error, make_binding()));
}

TEST(OutputEventSequencerTest, SingleEngineModeAcceptsOnlyBoundSender) {
  RemotePdOutputBinding binding = make_binding();
  binding.execution_mode = xllm::proto::EXECUTION_MODE_LOCAL_PREFILL_DECODE;
  binding.prefill_engine_uid = "decode-1";
  binding.prefill_incarnation_id = "decode-incarnation-1";
  binding.decode_engine_uid.clear();
  binding.decode_incarnation_id.clear();

  llm::RequestOutput output = make_output(0, "local");
  output.attempt_seq = 3;
  output.sender_engine_uid = "decode-1";
  output.sender_incarnation_id = "decode-incarnation-1";
  EXPECT_TRUE(matches_remote_pd_output_identity(output, binding));
  output.sender_engine_uid = "prefill-1";
  EXPECT_FALSE(matches_remote_pd_output_identity(output, binding));
}

TEST(OutputEventSequencerTest, LegacyOutputPassesOnlyWhenSequenceIsOptional) {
  OutputEventSequencer legacy(make_config());
  auto ready = legacy.push(make_output(std::nullopt, "legacy"), false);
  ASSERT_EQ(ready.status, OutputEventSequenceStatus::kReady);
  ASSERT_EQ(ready.ready_outputs.size(), 1);
  EXPECT_EQ(ready.ready_outputs[0].outputs[0].text, "legacy");

  OutputEventSequencer v2(make_config());
  auto missing = v2.push(make_output(std::nullopt, "missing"), true);
  EXPECT_EQ(missing.status, OutputEventSequenceStatus::kMissingSequence);
  EXPECT_TRUE(missing.ready_outputs.empty());
}

TEST(OutputEventSequencerTest, ReordersCrossSenderEventsAndReleasesCapacity) {
  OutputEventSequencer sequencer(make_config());
  auto two = sequencer.push(make_output(2, "two"), true);
  EXPECT_EQ(two.status, OutputEventSequenceStatus::kBuffered);
  EXPECT_EQ(sequencer.buffered_events(), 1);

  auto zero = sequencer.push(make_output(0, "zero"), true);
  ASSERT_EQ(zero.status, OutputEventSequenceStatus::kReady);
  ASSERT_EQ(zero.ready_outputs.size(), 1);
  EXPECT_EQ(zero.ready_outputs[0].outputs[0].text, "zero");

  auto one = sequencer.push(make_output(1, "one"), true);
  ASSERT_EQ(one.status, OutputEventSequenceStatus::kReady);
  ASSERT_EQ(one.ready_outputs.size(), 2);
  EXPECT_EQ(one.ready_outputs[0].outputs[0].text, "one");
  EXPECT_EQ(one.ready_outputs[1].outputs[0].text, "two");
  EXPECT_EQ(sequencer.next_expected_seq(), 3);
  EXPECT_EQ(sequencer.buffered_events(), 0);
  EXPECT_EQ(sequencer.buffered_bytes(), 0);
}

TEST(OutputEventSequencerTest, DropsDeliveredAndBufferedDuplicates) {
  OutputEventSequencer sequencer(make_config());
  EXPECT_EQ(sequencer.push(make_output(1, "first"), true).status,
            OutputEventSequenceStatus::kBuffered);
  EXPECT_EQ(sequencer.push(make_output(1, "duplicate"), true).status,
            OutputEventSequenceStatus::kDuplicate);
  EXPECT_EQ(sequencer.push(make_output(0, "zero"), true).status,
            OutputEventSequenceStatus::kReady);
  EXPECT_EQ(sequencer.push(make_output(1, "late"), true).status,
            OutputEventSequenceStatus::kDuplicate);
}

TEST(OutputEventSequencerTest, RejectsGapEventAndByteCapacityOverflow) {
  OutputEventSequencer::Config gap_config = make_config();
  gap_config.max_sequence_gap = 1;
  OutputEventSequencer gap_limited(gap_config);
  EXPECT_EQ(gap_limited.push(make_output(2, "far"), true).status,
            OutputEventSequenceStatus::kGapTooLarge);

  OutputEventSequencer::Config byte_config = make_config();
  byte_config.max_buffered_bytes = 1;
  OutputEventSequencer byte_limited(byte_config);
  EXPECT_EQ(byte_limited.push(make_output(1, "payload"), true).status,
            OutputEventSequenceStatus::kCapacityExceeded);
}

TEST(OutputEventSequencerTest, RejectsEventCountCapacityOverflow) {
  OutputEventSequencer::Config config = make_config();
  config.max_buffered_events = 1;
  OutputEventSequencer sequencer(config);
  EXPECT_EQ(sequencer.push(make_output(2, "two"), true).status,
            OutputEventSequenceStatus::kBuffered);
  EXPECT_EQ(sequencer.push(make_output(1, "one"), true).status,
            OutputEventSequenceStatus::kCapacityExceeded);
}

TEST(OutputEventSequencerTest, TerminalSequenceFencesLaterEvents) {
  OutputEventSequencer sequencer(make_config());
  EXPECT_EQ(sequencer.push(make_output(2, "done", true), true).status,
            OutputEventSequenceStatus::kBuffered);
  EXPECT_EQ(sequencer.push(make_output(3, "after"), true).status,
            OutputEventSequenceStatus::kTerminalConflict);
  EXPECT_EQ(sequencer.push(make_output(0, "zero"), true).status,
            OutputEventSequenceStatus::kReady);
  auto ready = sequencer.push(make_output(1, "one"), true);
  ASSERT_EQ(ready.status, OutputEventSequenceStatus::kReady);
  ASSERT_EQ(ready.ready_outputs.size(), 2);
  EXPECT_TRUE(ready.ready_outputs[1].finished);
  EXPECT_EQ(sequencer.push(make_output(3, "late"), true).status,
            OutputEventSequenceStatus::kTerminalConflict);
}

TEST(OutputEventSequencerTest, TerminalCannotOvertakeBufferedOutput) {
  OutputEventSequencer sequencer(make_config());
  EXPECT_EQ(sequencer.push(make_output(2, "two"), true).status,
            OutputEventSequenceStatus::kBuffered);
  EXPECT_EQ(
      sequencer.push(make_output(1, "premature-terminal", true), true).status,
      OutputEventSequenceStatus::kTerminalConflict);
}

TEST(OutputEventSequencerTest, RejectsSequenceSpaceSentinel) {
  OutputEventSequencer sequencer(make_config());
  EXPECT_EQ(
      sequencer
          .push(make_output(std::numeric_limits<uint64_t>::max(), "max"), true)
          .status,
      OutputEventSequenceStatus::kInvalidSequence);
}

TEST(OutputEventSequencerTest, GapTimeoutUsesLocalMonotonicTime) {
  OutputEventSequencer sequencer(make_config());
  const auto start = OutputEventSequencer::TimePoint{};
  EXPECT_EQ(sequencer.push(make_output(1, "one"), true, start).status,
            OutputEventSequenceStatus::kBuffered);
  EXPECT_FALSE(sequencer.gap_expired(start + std::chrono::milliseconds(99),
                                     std::chrono::milliseconds(100)));
  EXPECT_FALSE(sequencer.gap_expired(start - std::chrono::milliseconds(1),
                                     std::chrono::milliseconds(100)));
  EXPECT_TRUE(sequencer.gap_expired(start + std::chrono::milliseconds(100),
                                    std::chrono::milliseconds(100)));
  sequencer.close();
  EXPECT_EQ(sequencer.push(make_output(0, "zero"), true, start).status,
            OutputEventSequenceStatus::kClosed);
}

TEST(OutputEventSequencerTest, FilledGapNeverTimesOut) {
  OutputEventSequencer sequencer(make_config());
  const auto start = OutputEventSequencer::TimePoint{};
  EXPECT_EQ(sequencer.push(make_output(1, "one"), true, start).status,
            OutputEventSequenceStatus::kBuffered);
  EXPECT_EQ(sequencer.push(make_output(0, "zero"), true, start).status,
            OutputEventSequenceStatus::kReady);
  EXPECT_FALSE(sequencer.gap_expired(start + std::chrono::seconds(1),
                                     std::chrono::milliseconds(100)));
}

}  // namespace
}  // namespace xllm_service
