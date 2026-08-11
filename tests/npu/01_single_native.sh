#!/usr/bin/env bash
set -u -o pipefail

TEST_NAME=single_native
FAIL_COUNT=0
source "$(dirname "$0")/lib.sh"
load_ascend_env
TEST_START_EPOCH=$(date +%s)
TEST_START_ISO=$(date '+%F %T %z')

IFS=', ' read -r CARD _REST <<< "${NPU_IDS:-}"
if [[ -z "${CARD:-}" ]]; then
  fail "NPU_IDS must contain at least one card"
  finish
  exit $?
fi
if ! etcd_alive; then
  fail "etcd is not reachable"
  finish
  exit $?
fi
maybe_clean_namespace || fail "cannot clean test namespace"
trap 'stop_named telemetry_single_native; stop_named engine0; stop_named service0' EXIT

start_service 0
if wait_code "http://127.0.0.1:$(service_http 0)/livez" 200 30; then
  pass "Service liveness before engine registration"
else
  fail "Service liveness before engine registration"
fi

before_ready=$(http_code "http://127.0.0.1:$(service_http 0)/readyz")
printf '%s\n' "$before_ready" > "$SNAP/readyz_before.txt"
log "readyz before engine=$before_ready"

start_engine engine0 DEFAULT "$CARD" 0 127.0.0.1
if wait_role_count DEFAULT 1 "$REGISTER_TIMEOUT"; then
  pass "DEFAULT engine registered"
else
  fail "DEFAULT engine did not register"
fi
if wait_code "http://127.0.0.1:$(service_http 0)/readyz" 200 "$READY_TIMEOUT"; then
  pass "Service ready after DEFAULT registration"
else
  fail "Service did not become ready"
fi
start_telemetry_sampler single_native "$(service_http 0)" "$(engine_http 0)"

snapshot 01_registered "$(service_http 0)" "$(engine_http 0)"
record_process_snapshot 01_registered

if request_once "$(service_http 0)" native_1 'Return the single word OK.' 8; then
  pass "single request succeeded"
else
  fail "single request failed"
fi
if request_once "$(service_http 0)" native_2 'Return the number 2 only.' 8; then
  pass "second request succeeded"
else
  fail "second request failed"
fi

old_hash=$(role_hash DEFAULT)
stop_named engine0
if wait_role_count DEFAULT 0 90; then
  pass "DEFAULT lease disappeared after engine stop"
else
  fail "DEFAULT lease remained after engine stop"
fi
ready_after_stop=$(http_code "http://127.0.0.1:$(service_http 0)/readyz")
printf '%s\n' "$ready_after_stop" > "$SNAP/readyz_after_engine_stop.txt"
snapshot 01_engine_stopped "$(service_http 0)" "$(engine_http 0)"

start_engine engine0 DEFAULT "$CARD" 0 127.0.0.1
if wait_role_count DEFAULT 1 "$REGISTER_TIMEOUT"; then
  pass "DEFAULT engine re-registered"
else
  fail "DEFAULT engine did not re-register"
fi
new_hash=$(role_hash DEFAULT)
if [[ -n "$old_hash" && -n "$new_hash" && "$old_hash" != "$new_hash" ]]; then
  pass "restart created a different registry value/incarnation"
else
  fail "restart did not change the registry value/incarnation"
fi
if wait_code "http://127.0.0.1:$(service_http 0)/readyz" 200 "$READY_TIMEOUT" &&
   request_once "$(service_http 0)" native_after_restart 'Return OK after restart.' 8; then
  pass "request succeeded after engine restart"
else
  fail "request failed after engine restart"
fi

snapshot 01_final "$(service_http 0)" "$(engine_http 0)"
record_process_snapshot 01_final
finish
