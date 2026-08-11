#!/usr/bin/env bash
set -u -o pipefail

TEST_NAME=preflight
FAIL_COUNT=0
source "$(dirname "$0")/lib.sh"
TEST_START_EPOCH=$(date +%s)
TEST_START_ISO=$(date '+%F %T %z')

log "preflight run_dir=$RUN_DIR"
if ! check_common; then
  fail "common prerequisites are missing"
fi
if etcd_alive; then
  pass "etcd reachable at $ETCD_ADDR"
else
  fail "etcd is not reachable at $ETCD_ADDR"
fi

if command -v npu-smi >/dev/null 2>&1; then
  timeout "$NPU_SMI_TIMEOUT" npu-smi info > "$SNAP/preflight_npu-smi.txt" 2>&1
  if [[ $? -eq 0 ]]; then
    pass "npu-smi completed"
  else
    fail "npu-smi did not complete within ${NPU_SMI_TIMEOUT}s"
  fi
else
  fail "npu-smi is not installed"
fi

if [[ -d "$SERVICE_ROOT/build" ]]; then
  ctest --test-dir "$SERVICE_ROOT/build" -N > "$SNAP/ctest_list.txt" 2>&1 || true
  if grep -q 'Total Tests: 0' "$SNAP/ctest_list.txt"; then
    fail "current Service build exposes zero CTest tests; configure with -DBUILD_TESTING=ON"
  else
    pass "CTest list is non-empty"
  fi
else
  fail "Service build directory is missing"
fi

{
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'run_dir=%s\n' "$RUN_DIR"
  printf 'service_bin=%s\n' "$SERVICE_BIN"
  printf 'xllm_bin=%s\n' "$XLLM_BIN"
  printf 'model_path=%s\n' "$MODEL_PATH"
  printf 'etcd_addr=%s\n' "$ETCD_ADDR"
  printf 'etcd_namespace=%s\n' "$ETCD_NAMESPACE"
  printf 'npu_ids=%s\n' "${NPU_IDS:-}"
  printf 'date=%s\n' "$(date '+%F %T %z')"
} > "$SNAP/preflight_manifest.txt"
record_process_snapshot preflight
finish
