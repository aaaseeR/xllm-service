#!/usr/bin/env bash
set -u -o pipefail

TEST_NAME=v3_shadow
FAIL_COUNT=0
source "$(dirname "$0")/lib.sh"
load_ascend_env
TEST_START_EPOCH=$(date +%s)
TEST_START_ISO=$(date '+%F %T %z')

V3_CONFIG=${V3_CONFIG:-}
if [[ -z "$V3_CONFIG" || ! -f "$V3_CONFIG" ]]; then
  fail "V3_CONFIG must point to an existing strict placement config"
  finish
  exit $?
fi
IFS=', ' read -r PREFILL_CARD DECODE_CARD _REST <<< "${NPU_IDS:-}"
if [[ -z "${PREFILL_CARD:-}" || -z "${DECODE_CARD:-}" ]]; then
  fail "NPU_IDS must contain two cards"
  finish
  exit $?
fi
if ! etcd_alive; then
  fail "etcd is not reachable"
  finish
  exit $?
fi
maybe_clean_namespace || fail "cannot clean test namespace"
PLACEMENT_CONFIG_PATH=$V3_CONFIG
PLACEMENT_MODE_OVERRIDE=${PLACEMENT_MODE_OVERRIDE:-1}
export PLACEMENT_CONFIG_PATH PLACEMENT_MODE_OVERRIDE
trap 'stop_named telemetry_v3_shadow; stop_named prefill; stop_named decode; stop_named service0; stop_named service1' EXIT

start_engine prefill PREFILL "$PREFILL_CARD" 0 127.0.0.1
start_engine decode DECODE "$DECODE_CARD" 1 127.0.0.1
if wait_role_count PREFILL 1 "$REGISTER_TIMEOUT" && wait_role_count DECODE 1 "$REGISTER_TIMEOUT"; then
  pass "P/D engines registered before V3 Services"
else
  fail "P/D engines did not register before V3 Services"
fi

start_service 0
start_service 1
for port in "$(service_http 0)" "$(service_http 1)"; do
  if wait_code "http://127.0.0.1:$port/livez" 200 30; then
    pass "Service $port is live"
  else
    fail "Service $port is not live"
  fi
done

for port in "$(service_http 0)" "$(service_http 1)"; do
  if wait_code "http://127.0.0.1:$port/readyz" 200 "$READY_TIMEOUT"; then
    pass "Service $port is ready"
  else
    fail "Service $port is not ready"
  fi
done
start_telemetry_sampler v3_shadow "$(service_http 0)" "$(service_http 1)" "$(engine_http 0)" "$(engine_http 1)"

sleep "${V3_SETTLE_SECONDS:-15}"
snapshot 04_before_failover "$(service_http 0)" "$(service_http 1)" "$(engine_http 0)" "$(engine_http 1)"
record_process_snapshot 04_before_failover

mode0=$(metric_value "$(service_http 0)" xllm_service_v3_placement_mode)
mode1=$(metric_value "$(service_http 1)" xllm_service_v3_placement_mode)
pools0=$(metric_value "$(service_http 0)" xllm_service_v3_placement_pools)
pools1=$(metric_value "$(service_http 1)" xllm_service_v3_placement_pools)
[[ "$mode0" == "$PLACEMENT_MODE_OVERRIDE" ]] && pass "Service 0 placement mode=$mode0" || fail "Service 0 placement mode=$mode0"
[[ "$mode1" == "$PLACEMENT_MODE_OVERRIDE" ]] && pass "Service 1 placement mode=$mode1" || fail "Service 1 placement mode=$mode1"
[[ "${pools0:-0}" =~ ^[1-9][0-9]*$ ]] && pass "Service 0 has placement pools=$pools0" || fail "Service 0 has no placement pools"
[[ "${pools1:-0}" =~ ^[1-9][0-9]*$ ]] && pass "Service 1 has placement pools=$pools1" || fail "Service 1 has no placement pools"

leader0=$(metric_value "$(service_http 0)" xllm_service_v3_placement_leader)
leader1=$(metric_value "$(service_http 1)" xllm_service_v3_placement_leader)
if [[ "$leader0" == 1 && "$leader1" == 0 ]] || [[ "$leader0" == 0 && "$leader1" == 1 ]]; then
  pass "exactly one placement leader"
else
  fail "invalid leader values: service0=$leader0 service1=$leader1"
fi

command_keys=$(key_count 'XLLM:PLACEMENT:COMMAND/')
if [[ "$command_keys" == 0 ]]; then
  pass "SHADOW produced no placement command records"
else
  fail "SHADOW produced $command_keys placement command records"
fi
if metric_present "$(service_http 0)" xllm_service_v3_placement_cycles_total; then
  pass "placement cycle counter is exported"
else
  fail "placement cycle counter is missing"
fi
if request_once "$(service_http 0)" shadow_request \
  'Return the exact text SHADOW_REQUEST.' 16; then
  pass "request succeeded while placement was SHADOW"
else
  fail "request failed while placement was SHADOW"
fi

if [[ "$leader0" == 1 ]]; then
  stop_named service0
  survivor=1
else
  stop_named service1
  survivor=0
fi
survivor_port=$(service_http "$survivor")
if wait_code "http://127.0.0.1:$survivor_port/readyz" 200 "$READY_TIMEOUT"; then
  pass "surviving Service stayed ready after leader kill"
else
  fail "surviving Service did not stay ready after leader kill"
fi
leader_after=$(metric_value "$survivor_port" xllm_service_v3_placement_leader)
if [[ "$leader_after" == 1 ]]; then
  pass "surviving Service acquired placement leadership"
else
  fail "surviving Service did not acquire placement leadership"
fi
if request_once "$survivor_port" shadow_after_failover \
  'Return the exact text SHADOW_FAILOVER.' 16; then
  pass "request succeeded after placement leader failover"
else
  fail "request failed after placement leader failover"
fi
if [[ "$(key_count 'XLLM:PLACEMENT:COMMAND/')" == 0 ]]; then
  pass "no placement command records after leader failover"
else
  fail "placement command records appeared after leader failover"
fi

snapshot 04_final "$survivor_port" "$(engine_http 0)" "$(engine_http 1)"
record_process_snapshot 04_final
finish
