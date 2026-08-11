#!/usr/bin/env bash
set -u -o pipefail

TEST_NAME=cross_host_pd
FAIL_COUNT=0
source "$(dirname "$0")/lib.sh"
load_ascend_env
TEST_START_EPOCH=$(date +%s)
TEST_START_ISO=$(date '+%F %T %z')

ROLE=${ROLE:-}
HOST_IP=${HOST_IP:-}
NPU_ID=${NPU_ID:-}
START_SERVICE=${START_SERVICE:-0}
RUN_REQUEST=${RUN_REQUEST:-0}

if [[ "$ROLE" != PREFILL && "$ROLE" != DECODE ]]; then
  fail "ROLE must be PREFILL or DECODE"
  finish
  exit $?
fi
if [[ -z "$HOST_IP" || -z "$NPU_ID" ]]; then
  fail "HOST_IP and NPU_ID are required"
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
trap 'stop_named telemetry_cross_host_pd; stop_named engine0; [[ "$START_SERVICE" == 1 ]] && stop_named service0' EXIT

if [[ "$START_SERVICE" == 1 ]]; then
  start_service 0
  if wait_code "http://127.0.0.1:$(service_http 0)/livez" 200 30; then
    pass "controller Service is live"
  else
    fail "controller Service is not live"
  fi
fi

start_engine engine0 "$ROLE" "$NPU_ID" 0 "$HOST_IP"
if wait_role_count "$ROLE" 1 "$REGISTER_TIMEOUT"; then
  pass "$ROLE registered from host=$HOST_IP"
else
  fail "$ROLE did not register from host=$HOST_IP"
fi
if [[ "$START_SERVICE" == 1 ]]; then
  start_telemetry_sampler cross_host_pd "$(service_http 0)" "$(engine_http 0)"
else
  start_telemetry_sampler cross_host_pd "$(engine_http 0)"
fi

if [[ "$START_SERVICE" == 1 ]]; then
  log "waiting for remote PREFILL and DECODE registrations"
  if wait_role_count PREFILL 1 "$REGISTER_TIMEOUT" &&
     wait_role_count DECODE 1 "$REGISTER_TIMEOUT" &&
     wait_code "http://127.0.0.1:$(service_http 0)/readyz" 200 "$READY_TIMEOUT"; then
    pass "cross-host P/D view is ready"
  else
    fail "cross-host P/D view did not become ready"
  fi
  if [[ "$RUN_REQUEST" == 1 ]]; then
    if request_once "$(service_http 0)" cross_host_pd \
      'Return the exact text CROSS_HOST_PD.' 16; then
      pass "cross-host P/D request succeeded"
    else
      fail "cross-host P/D request failed"
    fi
  fi
fi

if [[ "$START_SERVICE" == 1 ]]; then
  snapshot "03_${ROLE}_final" "$(service_http 0)" "$(engine_http 0)"
else
  snapshot "03_${ROLE}_final" "$(engine_http 0)"
fi
record_process_snapshot "03_${ROLE}_final"
finish
