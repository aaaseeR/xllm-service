#!/usr/bin/env bash
set -u -o pipefail

TEST_NAME=single_pd
FAIL_COUNT=0
source "$(dirname "$0")/lib.sh"
load_ascend_env
TEST_START_EPOCH=$(date +%s)
TEST_START_ISO=$(date '+%F %T %z')

IFS=', ' read -r PREFILL_CARD DECODE_CARD _REST <<< "${NPU_IDS:-}"
if [[ -z "${PREFILL_CARD:-}" || -z "${DECODE_CARD:-}" ]]; then
  fail "NPU_IDS must contain two cards, e.g. NPU_IDS=0,1"
  finish
  exit $?
fi
if ! etcd_alive; then
  fail "etcd is not reachable"
  finish
  exit $?
fi
maybe_clean_namespace || fail "cannot clean test namespace"
trap 'stop_named telemetry_single_pd; stop_named prefill; stop_named decode; stop_named service0' EXIT

start_service 0
start_engine prefill PREFILL "$PREFILL_CARD" 0 127.0.0.1
start_engine decode DECODE "$DECODE_CARD" 1 127.0.0.1

if wait_role_count PREFILL 1 "$REGISTER_TIMEOUT"; then
  pass "PREFILL registered"
else
  fail "PREFILL did not register"
fi
if wait_role_count DECODE 1 "$REGISTER_TIMEOUT"; then
  pass "DECODE registered"
else
  fail "DECODE did not register"
fi
if wait_code "http://127.0.0.1:$(service_http 0)/readyz" 200 "$READY_TIMEOUT"; then
  pass "Service ready with P/D"
else
  fail "Service not ready with P/D"
fi
start_telemetry_sampler single_pd "$(service_http 0)" "$(engine_http 0)" "$(engine_http 1)"

snapshot 02_registered "$(service_http 0)" "$(engine_http 0)" "$(engine_http 1)"
record_process_snapshot 02_registered

if [[ "$(role_count PREFILL)" == 1 && "$(role_count DECODE)" == 1 && "$(role_count DEFAULT)" == 0 ]]; then
  pass "etcd contains exactly one P and one D, no DEFAULT"
else
  fail "unexpected role counts: P=$(role_count PREFILL) D=$(role_count DECODE) DEFAULT=$(role_count DEFAULT)"
fi
if metric_present "$(service_http 0)" xllm_service_v2_active_requests; then
  pass "Prometheus metrics endpoint is available"
else
  fail "Prometheus metrics endpoint is missing expected V2 metric"
fi

if request_once "$(service_http 0)" pd_short 'Return the word PD.' 8; then
  pass "P/D short request succeeded"
else
  fail "P/D short request failed"
fi
if request_once "$(service_http 0)" pd_long \
  'Explain in one short paragraph why a prefill stage and a decode stage can share a request.' 32; then
  pass "P/D longer request succeeded"
else
  fail "P/D longer request failed"
fi

stop_named prefill
if wait_role_count PREFILL 0 90 && [[ "$(role_count DECODE)" == 1 ]]; then
  pass "PREFILL stop removes only PREFILL"
else
  fail "PREFILL stop changed an unexpected role or lease remained"
fi
start_engine prefill PREFILL "$PREFILL_CARD" 0 127.0.0.1
if wait_role_count PREFILL 1 "$REGISTER_TIMEOUT"; then
  pass "PREFILL re-registered"
else
  fail "PREFILL did not re-register"
fi

stop_named decode
if wait_role_count DECODE 0 90 && [[ "$(role_count PREFILL)" == 1 ]]; then
  pass "DECODE stop removes only DECODE"
else
  fail "DECODE stop changed an unexpected role or lease remained"
fi
start_engine decode DECODE "$DECODE_CARD" 1 127.0.0.1
if wait_role_count DECODE 1 "$REGISTER_TIMEOUT"; then
  pass "DECODE re-registered"
else
  fail "DECODE did not re-register"
fi

if wait_code "http://127.0.0.1:$(service_http 0)/readyz" 200 "$READY_TIMEOUT" &&
   request_once "$(service_http 0)" pd_after_restart 'Return READY after P/D restart.' 8; then
  pass "P/D request succeeded after role restarts"
else
  fail "P/D request failed after role restarts"
fi

snapshot 02_final "$(service_http 0)" "$(engine_http 0)" "$(engine_http 1)"
record_process_snapshot 02_final
finish
