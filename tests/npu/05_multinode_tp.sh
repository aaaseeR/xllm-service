#!/usr/bin/env bash
set -u -o pipefail

TEST_NAME=multinode_tp
FAIL_COUNT=0
source "$(dirname "$0")/lib.sh"
load_ascend_env
TEST_START_EPOCH=$(date +%s)
TEST_START_ISO=$(date '+%F %T %z')

NNODES=${NNODES:-}
NODE_RANK=${NODE_RANK:-}
MASTER_NODE_ADDR=${MASTER_NODE_ADDR:-}
HOST_IP=${HOST_IP:-}
NPU_ID=${NPU_ID:-}
START_SERVICE=${START_SERVICE:-0}
RUN_REQUEST=${RUN_REQUEST:-0}

if [[ -z "$NNODES" || -z "$NODE_RANK" || -z "$MASTER_NODE_ADDR" ||
      -z "$HOST_IP" || -z "$NPU_ID" ]]; then
  fail "NNODES, NODE_RANK, MASTER_NODE_ADDR, HOST_IP, and NPU_ID are required"
  finish
  exit $?
fi
if (( NNODES < 2 || NODE_RANK < 0 || NODE_RANK >= NNODES )); then
  fail "invalid NNODES/NODE_RANK: NNODES=$NNODES NODE_RANK=$NODE_RANK"
  finish
  exit $?
fi
if ! etcd_alive; then
  fail "etcd is not reachable"
  finish
  exit $?
fi
if [[ "$START_SERVICE" == 1 ]]; then
  maybe_clean_namespace || fail "cannot clean test namespace"
fi
trap 'stop_named telemetry_multinode_tp; stop_named engine0; [[ "$START_SERVICE" == 1 ]] && stop_named service0' EXIT

export XLLM_NNODES="$NNODES"
export XLLM_NODE_RANK="$NODE_RANK"
export XLLM_MASTER_NODE_ADDR="$MASTER_NODE_ADDR"

if [[ "$START_SERVICE" == 1 ]]; then
  start_service 0
  if wait_code "http://127.0.0.1:$(service_http 0)/livez" 200 30; then
    pass "rank 0 Service is live"
  else
    fail "rank 0 Service is not live"
  fi
fi

start_engine engine0 DEFAULT "$NPU_ID" 0 "$HOST_IP"
if wait_role_count DEFAULT 1 "$REGISTER_TIMEOUT"; then
  pass "TP world registered node_rank=$NODE_RANK host=$HOST_IP"
else
  fail "TP world did not register node_rank=$NODE_RANK host=$HOST_IP"
fi
if [[ "$START_SERVICE" == 1 ]]; then
  start_telemetry_sampler multinode_tp "$(service_http 0)" "$(engine_http 0)"
else
  start_telemetry_sampler multinode_tp "$(engine_http 0)"
fi

if [[ "$START_SERVICE" == 1 ]]; then
  if wait_code "http://127.0.0.1:$(service_http 0)/readyz" 200 "$READY_TIMEOUT"; then
    pass "rank 0 Service is ready for TP engine"
  else
    fail "rank 0 Service is not ready for TP engine"
  fi
  if [[ "$RUN_REQUEST" == 1 ]]; then
    if request_once "$(service_http 0)" multinode_tp \
      'Return the exact text MULTINODE_TP.' 16; then
      pass "multi-node TP request succeeded"
    else
      fail "multi-node TP request failed"
    fi
  fi
fi

if [[ "$START_SERVICE" == 1 ]]; then
  snapshot "05_rank_${NODE_RANK}" "$(service_http 0)" "$(engine_http 0)"
else
  snapshot "05_rank_${NODE_RANK}" "$(engine_http 0)"
fi
record_process_snapshot "05_rank_${NODE_RANK}"
finish
