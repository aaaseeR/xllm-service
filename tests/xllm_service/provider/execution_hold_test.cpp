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

#include "provider/execution_hold.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace xllm_service::provider {
namespace {

using xllm::proto::ExecutionAttemptId;
using xllm::proto::ExecutionHolder;
using xllm::proto::ExecutionHoldProof;
using xllm::proto::ExecutionResourceHold;
using xllm::proto::HolderConvergenceProof;

constexpr char kRequestUid[] = "0197f0a1-1234-7abc-8def-0123456789ab";

ExecutionHoldCleanupTable::Config make_config(size_t record_capacity = 4) {
  return ExecutionHoldCleanupTable::Config{
      .record_capacity = record_capacity,
      .byte_capacity = record_capacity * 1024,
      .max_cleanup_record_bytes = 1024,
      .max_potential_holders = 4,
      .max_identifier_bytes = 128,
      .allow_hard_time_bound_proof = false,
  };
}

ExecutionAttemptId make_attempt(uint64_t attempt_seq = 0,
                                const std::string& request_uid = kRequestUid) {
  ExecutionAttemptId attempt;
  attempt.set_request_uid(request_uid);
  attempt.set_attempt_seq(attempt_seq);
  return attempt;
}

ExecutionHolder make_holder(const std::string& suffix) {
  ExecutionHolder holder;
  holder.set_engine_uid("engine-" + suffix);
  holder.set_incarnation_id("incarnation-" + suffix);
  return holder;
}

ExecutionResourceHold make_hold(
    uint64_t attempt_seq = 0,
    size_t holder_count = 2,
    xllm::proto::ExecutionHoldKind kind =
        xllm::proto::EXECUTION_HOLD_KIND_REMOTE_D_RESERVATION) {
  ExecutionResourceHold hold;
  hold.set_kind(kind);
  *hold.mutable_attempt() = make_attempt(attempt_seq);
  hold.set_coordinator_incarnation_id("coordinator-1");
  for (size_t index = 0; index < holder_count; ++index) {
    *hold.add_potential_holders() = make_holder(std::to_string(index));
  }
  hold.set_proof(xllm::proto::EXECUTION_HOLD_PROOF_OUTCOME_UNKNOWN);
  return hold;
}

TEST(ExecutionHoldTest, ReservationIsBoundedAndReleasedByRaii) {
  ExecutionHoldCleanupTable table(make_config(/*record_capacity=*/2));
  auto first = table.try_reserve();
  auto second = table.try_reserve();
  EXPECT_TRUE(first.has_value());
  EXPECT_TRUE(second.has_value());
  EXPECT_FALSE(table.try_reserve().has_value());

  auto stats = table.stats();
  EXPECT_EQ(stats.reserved_records, 2);
  EXPECT_EQ(stats.reserved_bytes, 2048);
  first.reset();
  EXPECT_EQ(table.stats().reserved_records, 1);
  EXPECT_TRUE(table.try_reserve().has_value());
}

TEST(ExecutionHoldTest, ByteCapacityCanBackpressureBeforeRecordCapacity) {
  auto config = make_config(/*record_capacity=*/4);
  config.byte_capacity = 2048;
  ExecutionHoldCleanupTable table(config);
  auto first = table.try_reserve();
  auto second = table.try_reserve();
  EXPECT_TRUE(first.has_value());
  EXPECT_TRUE(second.has_value());
  EXPECT_FALSE(table.try_reserve().has_value());
  EXPECT_EQ(table.stats().reserved_records, 2);
}

TEST(ExecutionHoldTest, InvalidInstallFailsClosedAndReleasesCapacity) {
  ExecutionHoldCleanupTable table(make_config());
  RequestExecutionHold request_hold;
  ExecutionResourceHold hold = make_hold();
  hold.mutable_attempt()->clear_attempt_seq();

  auto reservation = table.try_reserve();
  ASSERT_TRUE(reservation.has_value());
  EXPECT_EQ(request_hold.install(std::move(*reservation), std::move(hold)),
            ExecutionHoldStatus::kInvalidArgument);
  EXPECT_EQ(table.stats().reserved_records, 0);
}

TEST(ExecutionHoldTest, ConcurrentInstallAllowsExactlyOneDispatchGate) {
  constexpr size_t kThreads = 32;
  ExecutionHoldCleanupTable table(make_config(/*record_capacity=*/kThreads));
  RequestExecutionHold request_hold;
  std::atomic<bool> start = false;
  std::atomic<size_t> installed = 0;
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (size_t index = 0; index < kThreads; ++index) {
    threads.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      auto reservation = table.try_reserve();
      ASSERT_TRUE(reservation.has_value());
      if (request_hold.install(std::move(*reservation), make_hold()) ==
          ExecutionHoldStatus::kOk) {
        installed.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (std::thread& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(installed.load(), 1);
  EXPECT_EQ(table.stats().reserved_records, 1);
}

TEST(ExecutionHoldTest, LikelyHolderNeverNarrowsSafeCandidates) {
  ExecutionHoldCleanupTable table(make_config());
  RequestExecutionHold request_hold;
  auto reservation = table.try_reserve();
  ASSERT_TRUE(reservation.has_value());
  ASSERT_EQ(request_hold.install(std::move(*reservation), make_hold()),
            ExecutionHoldStatus::kOk);

  EXPECT_EQ(request_hold.set_likely_holder(make_holder("1")),
            ExecutionHoldStatus::kOk);
  auto snapshot = request_hold.snapshot();
  ASSERT_TRUE(snapshot.has_value());
  EXPECT_EQ(snapshot->potential_holders_size(), 2);
  ASSERT_TRUE(snapshot->has_likely_holder());
  EXPECT_EQ(snapshot->likely_holder().engine_uid(), "engine-1");
  EXPECT_EQ(request_hold.set_likely_holder(make_holder("unknown")),
            ExecutionHoldStatus::kHolderMismatch);
  EXPECT_EQ(request_hold.snapshot()->potential_holders_size(), 2);
}

TEST(ExecutionHoldTest, ConfirmedHolderRequiresEvidenceAndSafelyNarrows) {
  ExecutionHoldCleanupTable table(make_config());
  RequestExecutionHold request_hold;
  auto reservation = table.try_reserve();
  ASSERT_TRUE(reservation.has_value());
  ASSERT_EQ(request_hold.install(std::move(*reservation), make_hold()),
            ExecutionHoldStatus::kOk);

  EXPECT_EQ(
      request_hold.confirm_holder(
          make_holder("1"), xllm::proto::EXECUTION_HOLD_PROOF_OUTCOME_UNKNOWN),
      ExecutionHoldStatus::kUnsafeProof);
  EXPECT_EQ(request_hold.confirm_holder(
                make_holder("unknown"),
                xllm::proto::EXECUTION_HOLD_PROOF_PROVEN_PRECOMMIT),
            ExecutionHoldStatus::kHolderMismatch);
  EXPECT_EQ(
      request_hold.confirm_holder(
          make_holder("1"), xllm::proto::EXECUTION_HOLD_PROOF_PROVEN_PRECOMMIT),
      ExecutionHoldStatus::kOk);
  auto snapshot = request_hold.snapshot();
  ASSERT_TRUE(snapshot.has_value());
  EXPECT_EQ(snapshot->potential_holders_size(), 1);
  EXPECT_TRUE(snapshot->has_confirmed_holder());
  EXPECT_EQ(snapshot->confirmed_holder().incarnation_id(), "incarnation-1");
  EXPECT_EQ(request_hold.advance_proof(
                xllm::proto::EXECUTION_HOLD_PROOF_GENERATION_COMMITTED),
            ExecutionHoldStatus::kOk);
  EXPECT_EQ(request_hold.advance_proof(
                xllm::proto::EXECUTION_HOLD_PROOF_PROVEN_PRECOMMIT),
            ExecutionHoldStatus::kUnsafeProof);
}

TEST(ExecutionHoldTest, QueryAbsentCanNeverClearARequestHold) {
  ExecutionHoldCleanupTable table(make_config());
  RequestExecutionHold request_hold;
  auto reservation = table.try_reserve();
  ASSERT_TRUE(reservation.has_value());
  ASSERT_EQ(request_hold.install(std::move(*reservation), make_hold()),
            ExecutionHoldStatus::kOk);

  EXPECT_EQ(request_hold.apply_convergence_proof(
                make_attempt(),
                make_holder("0"),
                xllm::proto::HOLDER_CONVERGENCE_PROOF_QUERY_ABSENT),
            ExecutionHoldStatus::kUnsafeProof);
  EXPECT_TRUE(request_hold.has_hold());
  EXPECT_EQ(table.stats().reserved_records, 1);
}

TEST(ExecutionHoldTest, ProofMustMatchAttemptAndHolderIncarnation) {
  ExecutionHoldCleanupTable table(make_config());
  RequestExecutionHold request_hold;
  auto reservation = table.try_reserve();
  ASSERT_TRUE(reservation.has_value());
  ASSERT_EQ(request_hold.install(std::move(*reservation), make_hold()),
            ExecutionHoldStatus::kOk);

  EXPECT_EQ(request_hold.apply_convergence_proof(
                make_attempt(/*attempt_seq=*/1),
                make_holder("0"),
                xllm::proto::HOLDER_CONVERGENCE_PROOF_CANCEL_FENCE_ACK),
            ExecutionHoldStatus::kAttemptMismatch);
  ExecutionHolder stale = make_holder("0");
  stale.set_incarnation_id("old-incarnation");
  EXPECT_EQ(request_hold.apply_convergence_proof(
                make_attempt(),
                stale,
                xllm::proto::HOLDER_CONVERGENCE_PROOF_PROCESS_TERMINATED),
            ExecutionHoldStatus::kHolderMismatch);
  EXPECT_TRUE(request_hold.has_hold());
}

TEST(ExecutionHoldTest, EveryPotentialHolderMustConverge) {
  ExecutionHoldCleanupTable table(make_config());
  RequestExecutionHold request_hold;
  auto reservation = table.try_reserve();
  ASSERT_TRUE(reservation.has_value());
  ASSERT_EQ(request_hold.install(std::move(*reservation), make_hold()),
            ExecutionHoldStatus::kOk);

  EXPECT_EQ(request_hold.apply_convergence_proof(
                make_attempt(),
                make_holder("0"),
                xllm::proto::HOLDER_CONVERGENCE_PROOF_CANCEL_FENCE_ACK),
            ExecutionHoldStatus::kOk);
  EXPECT_TRUE(request_hold.has_hold());
  EXPECT_EQ(request_hold.apply_convergence_proof(
                make_attempt(),
                make_holder("1"),
                xllm::proto::HOLDER_CONVERGENCE_PROOF_PROCESS_TERMINATED),
            ExecutionHoldStatus::kResolved);
  EXPECT_FALSE(request_hold.has_hold());
  EXPECT_EQ(table.stats().reserved_records, 0);
}

TEST(ExecutionHoldTest, ResolvedRequestCanInstallNextAttempt) {
  ExecutionHoldCleanupTable table(make_config());
  RequestExecutionHold request_hold;
  auto first = table.try_reserve();
  ASSERT_TRUE(first.has_value());
  ASSERT_EQ(request_hold.install(std::move(*first), make_hold(0, 1)),
            ExecutionHoldStatus::kOk);
  ASSERT_EQ(request_hold.apply_convergence_proof(
                make_attempt(0),
                make_holder("0"),
                xllm::proto::HOLDER_CONVERGENCE_PROOF_TERMINAL_OUTCOME),
            ExecutionHoldStatus::kResolved);

  auto second = table.try_reserve();
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(request_hold.install(std::move(*second), make_hold(1, 1)),
            ExecutionHoldStatus::kOk);
  EXPECT_EQ(request_hold.snapshot()->attempt().attempt_seq(), 1);
}

TEST(ExecutionHoldTest, EarlyRequestEndTransfersSameTokenToCleanup) {
  ExecutionHoldCleanupTable table(make_config());
  RequestExecutionHold request_hold;
  auto reservation = table.try_reserve();
  ASSERT_TRUE(reservation.has_value());
  ASSERT_EQ(request_hold.install(std::move(*reservation), make_hold()),
            ExecutionHoldStatus::kOk);
  ASSERT_EQ(table.stats().reserved_records, 1);

  EXPECT_EQ(table.adopt(&request_hold), ExecutionHoldStatus::kOk);
  EXPECT_FALSE(request_hold.has_hold());
  EXPECT_TRUE(table.contains(make_attempt()));
  auto stats = table.stats();
  EXPECT_EQ(stats.reserved_records, 1);
  EXPECT_EQ(stats.cleanup_records, 1);
  auto snapshot = table.query(make_attempt());
  ASSERT_TRUE(snapshot.has_value());
  EXPECT_EQ(snapshot->potential_holders_size(), 2);
  EXPECT_EQ(table.adopt(&request_hold), ExecutionHoldStatus::kAlreadyDetached);
}

TEST(ExecutionHoldTest, CleanupRejectsAbsentAndReleasesAfterAllProofs) {
  ExecutionHoldCleanupTable table(make_config());
  RequestExecutionHold request_hold;
  auto reservation = table.try_reserve();
  ASSERT_TRUE(reservation.has_value());
  ASSERT_EQ(request_hold.install(std::move(*reservation), make_hold()),
            ExecutionHoldStatus::kOk);
  ASSERT_EQ(table.adopt(&request_hold), ExecutionHoldStatus::kOk);

  EXPECT_EQ(table.apply_convergence_proof(
                make_attempt(),
                make_holder("0"),
                xllm::proto::HOLDER_CONVERGENCE_PROOF_QUERY_ABSENT),
            ExecutionHoldStatus::kUnsafeProof);
  EXPECT_EQ(table.apply_convergence_proof(
                make_attempt(),
                make_holder("0"),
                xllm::proto::HOLDER_CONVERGENCE_PROOF_SELF_FENCED),
            ExecutionHoldStatus::kOk);
  EXPECT_EQ(table.apply_convergence_proof(
                make_attempt(),
                make_holder("1"),
                xllm::proto::HOLDER_CONVERGENCE_PROOF_PROCESS_TERMINATED),
            ExecutionHoldStatus::kResolved);
  EXPECT_FALSE(table.contains(make_attempt()));
  auto stats = table.stats();
  EXPECT_EQ(stats.cleanup_records, 0);
  EXPECT_EQ(stats.reserved_records, 0);
}

TEST(ExecutionHoldTest, ConfirmedHolderNarrowsCleanupFanout) {
  ExecutionHoldCleanupTable table(make_config());
  RequestExecutionHold request_hold;
  auto reservation = table.try_reserve();
  ASSERT_TRUE(reservation.has_value());
  ASSERT_EQ(request_hold.install(std::move(*reservation), make_hold()),
            ExecutionHoldStatus::kOk);
  ASSERT_EQ(
      request_hold.confirm_holder(
          make_holder("1"), xllm::proto::EXECUTION_HOLD_PROOF_PROVEN_PRECOMMIT),
      ExecutionHoldStatus::kOk);
  ASSERT_EQ(table.adopt(&request_hold), ExecutionHoldStatus::kOk);

  EXPECT_EQ(table.apply_convergence_proof(
                make_attempt(),
                make_holder("0"),
                xllm::proto::HOLDER_CONVERGENCE_PROOF_CANCEL_FENCE_ACK),
            ExecutionHoldStatus::kHolderMismatch);
  EXPECT_EQ(table.apply_convergence_proof(
                make_attempt(),
                make_holder("1"),
                xllm::proto::HOLDER_CONVERGENCE_PROOF_CANCEL_FENCE_ACK),
            ExecutionHoldStatus::kResolved);
}

TEST(ExecutionHoldTest, HardTimeProofRequiresExplicitProfileEnablement) {
  ExecutionHoldCleanupTable disabled(make_config());
  RequestExecutionHold disabled_hold;
  auto disabled_reservation = disabled.try_reserve();
  ASSERT_TRUE(disabled_reservation.has_value());
  ASSERT_EQ(
      disabled_hold.install(std::move(*disabled_reservation), make_hold(0, 1)),
      ExecutionHoldStatus::kOk);
  EXPECT_EQ(disabled_hold.apply_convergence_proof(
                make_attempt(),
                make_holder("0"),
                xllm::proto::HOLDER_CONVERGENCE_PROOF_HARD_TIME_BOUND),
            ExecutionHoldStatus::kUnsafeProof);
  EXPECT_TRUE(disabled_hold.has_hold());

  auto enabled_config = make_config();
  enabled_config.allow_hard_time_bound_proof = true;
  ExecutionHoldCleanupTable enabled(enabled_config);
  RequestExecutionHold enabled_hold;
  auto enabled_reservation = enabled.try_reserve();
  ASSERT_TRUE(enabled_reservation.has_value());
  ASSERT_EQ(
      enabled_hold.install(std::move(*enabled_reservation), make_hold(0, 1)),
      ExecutionHoldStatus::kOk);
  EXPECT_EQ(enabled_hold.apply_convergence_proof(
                make_attempt(),
                make_holder("0"),
                xllm::proto::HOLDER_CONVERGENCE_PROOF_HARD_TIME_BOUND),
            ExecutionHoldStatus::kResolved);
}

TEST(ExecutionHoldTest, ReservationCannotMoveAcrossCleanupTables) {
  ExecutionHoldCleanupTable source(make_config());
  ExecutionHoldCleanupTable target(make_config());
  RequestExecutionHold request_hold;
  auto reservation = source.try_reserve();
  ASSERT_TRUE(reservation.has_value());
  ASSERT_EQ(request_hold.install(std::move(*reservation), make_hold()),
            ExecutionHoldStatus::kOk);

  EXPECT_EQ(target.adopt(&request_hold), ExecutionHoldStatus::kInvalidArgument);
  EXPECT_TRUE(request_hold.has_hold());
  EXPECT_EQ(source.stats().reserved_records, 1);
  EXPECT_EQ(target.stats().reserved_records, 0);
}

TEST(ExecutionHoldTest, ConcurrentReservationExhaustionIsExact) {
  constexpr size_t kCapacity = 8;
  constexpr size_t kThreads = 64;
  ExecutionHoldCleanupTable table(make_config(kCapacity));
  std::atomic<bool> start = false;
  std::vector<std::optional<ExecutionHoldCleanupTable::Reservation>> tokens(
      kThreads);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (size_t index = 0; index < kThreads; ++index) {
    threads.emplace_back([&, index] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      tokens[index] = table.try_reserve();
    });
  }
  start.store(true, std::memory_order_release);
  for (std::thread& thread : threads) {
    thread.join();
  }

  size_t admitted = 0;
  for (const auto& token : tokens) {
    admitted += token.has_value() ? 1 : 0;
  }
  EXPECT_EQ(admitted, kCapacity);
  EXPECT_EQ(table.stats().reserved_records, kCapacity);
  tokens.clear();
  EXPECT_EQ(table.stats().reserved_records, 0);
}

TEST(ExecutionHoldTest, OldAttemptProofCannotClearNewCleanupRecord) {
  ExecutionHoldCleanupTable table(make_config());
  RequestExecutionHold request_hold;
  auto reservation = table.try_reserve();
  ASSERT_TRUE(reservation.has_value());
  ASSERT_EQ(request_hold.install(std::move(*reservation), make_hold(1, 1)),
            ExecutionHoldStatus::kOk);
  ASSERT_EQ(table.adopt(&request_hold), ExecutionHoldStatus::kOk);

  EXPECT_EQ(table.apply_convergence_proof(
                make_attempt(0),
                make_holder("0"),
                xllm::proto::HOLDER_CONVERGENCE_PROOF_TERMINAL_OUTCOME),
            ExecutionHoldStatus::kNoHold);
  EXPECT_TRUE(table.contains(make_attempt(1)));
}

TEST(ExecutionHoldTest, InvalidUuidAndDuplicateCandidatesAreRejected) {
  ExecutionHoldCleanupTable table(make_config());
  RequestExecutionHold invalid_uuid;
  auto first = table.try_reserve();
  ASSERT_TRUE(first.has_value());
  ExecutionResourceHold hold = make_hold();
  hold.mutable_attempt()->set_request_uid("not-a-uuid-v7");
  EXPECT_EQ(invalid_uuid.install(std::move(*first), std::move(hold)),
            ExecutionHoldStatus::kInvalidArgument);

  RequestExecutionHold duplicate;
  auto second = table.try_reserve();
  ASSERT_TRUE(second.has_value());
  hold = make_hold();
  *hold.add_potential_holders() = make_holder("0");
  EXPECT_EQ(duplicate.install(std::move(*second), std::move(hold)),
            ExecutionHoldStatus::kInvalidArgument);
  EXPECT_EQ(table.stats().reserved_records, 0);
}

TEST(ExecutionHoldTest, OversizedMinimalRecordIsRejectedBeforeDispatch) {
  auto config = make_config();
  config.max_cleanup_record_bytes = 64;
  ExecutionHoldCleanupTable table(config);
  RequestExecutionHold request_hold;
  auto reservation = table.try_reserve();
  ASSERT_TRUE(reservation.has_value());
  EXPECT_EQ(request_hold.install(std::move(*reservation), make_hold()),
            ExecutionHoldStatus::kRecordTooLarge);
  EXPECT_FALSE(request_hold.has_hold());
  EXPECT_EQ(table.stats().reserved_records, 0);
}

}  // namespace
}  // namespace xllm_service::provider
