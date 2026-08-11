#!/usr/bin/env bash
set -u -o pipefail

TEST_NAME=cleanup
FAIL_COUNT=0
source "$(dirname "$0")/lib.sh"
TEST_START_EPOCH=$(date +%s)
TEST_START_ISO=$(date '+%F %T %z')

for pid_file in "$STATE"/*.pid; do
  [[ -f "$pid_file" ]] || continue
  name=$(basename "$pid_file" .pid)
  stop_named "$name"
done

if [[ "${DELETE_NAMESPACE:-1}" == 1 ]] && etcd_alive; then
  cleanup_namespace
  if [[ "$(namespace_lines | wc -l)" -eq 0 ]]; then
    pass "test namespace removed: $ETCD_NAMESPACE"
  else
    fail "test namespace still contains keys: $ETCD_NAMESPACE"
  fi
else
  skip "namespace deletion disabled or etcd unavailable"
fi

record_process_snapshot cleanup
finish
