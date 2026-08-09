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

#include "provider/kv_state_replica.h"

#include <algorithm>
#include <limits>
#include <utility>

#include "provider/provider_contract.h"

namespace xllm_service::provider {

KVStateReplica::KVStateReplica(KVStateReplicaConfig config,
                               KVShadowIndex* shadow_index)
    : config_(std::move(config)), shadow_index_(shadow_index) {
  config_valid_ = config_.max_engine_batches > 0 &&
                  config_.max_serialized_bytes > 0 && shadow_index_ != nullptr;
}

KVApplyResult KVStateReplica::result(KVApplyCode code,
                                     std::string reason,
                                     uint64_t accepted_through_event_seq) {
  return KVApplyResult{
      .code = code,
      .reason = std::move(reason),
      .accepted_through_event_seq = accepted_through_event_seq,
  };
}

bool KVStateReplica::set_master(std::string master_incarnation) {
  if (!config_valid_) {
    return false;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (master_incarnation_ == master_incarnation) {
    return true;
  }
  master_incarnation_ = std::move(master_incarnation);
  stream_epoch_ = 0;
  last_stream_seq_ = 0;
  shadow_index_->reset_all();
  return true;
}

KVApplyResult KVStateReplica::apply(const xllm::proto::KVStateBatch& batch,
                                    uint64_t received_monotonic_ms) {
  if (!config_valid_ || received_monotonic_ms == 0 ||
      batch.contract_version() != kProviderContractVersion ||
      batch.master_incarnation().empty() || batch.stream_epoch() == 0 ||
      batch.stream_seq() == 0 || batch.engine_batches().empty() ||
      static_cast<size_t>(batch.engine_batches_size()) >
          config_.max_engine_batches ||
      batch.ByteSizeLong() > config_.max_serialized_bytes) {
    return result(KVApplyCode::REJECTED, "KV State batch is invalid", 0);
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (batch.master_incarnation() != master_incarnation_) {
    return result(KVApplyCode::REJECTED,
                  "KV State batch master is stale",
                  last_stream_seq_);
  }
  if (batch.stream_epoch() < stream_epoch_) {
    return result(KVApplyCode::DUPLICATE,
                  "KV State batch stream epoch is stale",
                  last_stream_seq_);
  }
  if (batch.stream_epoch() > stream_epoch_) {
    stream_epoch_ = batch.stream_epoch();
    last_stream_seq_ = 0;
    shadow_index_->reset_all();
  }
  if (batch.stream_seq() <= last_stream_seq_) {
    return result(KVApplyCode::DUPLICATE,
                  "KV State batch was already applied",
                  last_stream_seq_);
  }
  const bool stream_gap =
      last_stream_seq_ == std::numeric_limits<uint64_t>::max() ||
      batch.stream_seq() != last_stream_seq_ + 1;
  if (stream_gap) {
    shadow_index_->reset_all();
  }
  last_stream_seq_ = batch.stream_seq();

  KVApplyCode aggregate =
      stream_gap ? KVApplyCode::SNAPSHOT_REQUIRED : KVApplyCode::DUPLICATE;
  uint64_t accepted = 0;
  for (const xllm::proto::KVEventBatch& engine_batch : batch.engine_batches()) {
    const KVApplyResult applied =
        shadow_index_->apply_event_batch(engine_batch, received_monotonic_ms);
    accepted = std::max(accepted, applied.accepted_through_event_seq);
    if (applied.code == KVApplyCode::REJECTED) {
      shadow_index_->reset_all();
      return result(KVApplyCode::SNAPSHOT_REQUIRED,
                    "KV State batch contained an invalid Engine batch",
                    accepted);
    }
    if (applied.code == KVApplyCode::SNAPSHOT_REQUIRED) {
      aggregate = KVApplyCode::SNAPSHOT_REQUIRED;
    } else if (aggregate != KVApplyCode::SNAPSHOT_REQUIRED &&
               applied.code == KVApplyCode::RECOVERING) {
      aggregate = KVApplyCode::RECOVERING;
    } else if (aggregate == KVApplyCode::DUPLICATE &&
               applied.code == KVApplyCode::APPLIED) {
      aggregate = KVApplyCode::APPLIED;
    }
  }
  return result(aggregate,
                stream_gap ? "KV State stream gap discarded all prior credit"
                           : "KV State batch applied",
                accepted);
}

std::string KVStateReplica::master_incarnation() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return master_incarnation_;
}

uint64_t KVStateReplica::stream_epoch() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stream_epoch_;
}

uint64_t KVStateReplica::last_stream_seq() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return last_stream_seq_;
}

}  // namespace xllm_service::provider
