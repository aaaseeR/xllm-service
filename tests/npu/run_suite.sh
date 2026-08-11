#!/usr/bin/env bash
set -u -o pipefail

TEST_NAME=runner
FAIL_COUNT=0
source "$(dirname "$0")/lib.sh"
TEST_START_EPOCH=$(date +%s)
TEST_START_ISO=$(date '+%F %T %z')

REQUIRED_CARDS=${REQUIRED_CARDS:-2}
WAIT_FOR_ETCD=${WAIT_FOR_ETCD:-1}
ETCD_WAIT_INTERVAL=${ETCD_WAIT_INTERVAL:-10}
STOP_ON_FAIL=${STOP_ON_FAIL:-1}
SKIP_CLEANUP=${SKIP_CLEANUP:-0}
STAGES=${STAGES:-"00_preflight 01_single_native 02_single_pd 04_v3_shadow"}
mkdir -p "$RUN_DIR/stages"
exec 3>&1
exec > >(tee -a "$RUN_DIR/runner.log") 2>&1

log "runner started run_dir=$RUN_DIR stages=$STAGES"

missing_stage=0
for stage in $STAGES; do
  script="$SUITE_DIR/$stage"
  [[ "$script" == *.sh ]] || script="${script}.sh"
  if [[ ! -x "$script" ]]; then
    printf 'FAIL\trunner\tmissing stage script %s\n' "$script" | tee -a "$RESULTS"
    missing_stage=1
  fi
done
if (( missing_stage != 0 )); then
  bash "$SUITE_DIR/report.sh" || true
  exit 1
fi

if [[ "$WAIT_FOR_ETCD" == 1 ]]; then
  while ! etcd_alive; do
    log "waiting for etcd at $ETCD_ADDR interval=${ETCD_WAIT_INTERVAL}s"
    sleep "$ETCD_WAIT_INTERVAL"
  done
fi

if [[ -z "${NPU_IDS:-}" ]]; then
  mapfile -t discovered_cards < <(wait_for_free_npus "$REQUIRED_CARDS")
  if (( ${#discovered_cards[@]} < REQUIRED_CARDS )); then
    printf 'FAIL\trunner\tNPU allocation returned fewer than required cards\n' >> "$RESULTS"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  else
    NPU_IDS=$(IFS=,; printf '%s' "${discovered_cards[*]}")
    export NPU_IDS
    log "allocated NPU_IDS=$NPU_IDS"
  fi
else
  export NPU_IDS
  log "using explicitly assigned NPU_IDS=$NPU_IDS"
fi

export RUN_ROOT RUN_ID RUN_DIR STATE LOGS SNAP REQUESTS RESULTS TEST_SUMMARY
export MODEL_PATH REQUEST_MODEL ETCD_ADDR ETCD_URL ETCD_NAMESPACE

runner_status=0
for stage in $STAGES; do
  script="$SUITE_DIR/$stage"
  [[ "$script" == *.sh ]] || script="${script}.sh"
  if [[ ! -x "$script" ]]; then
    printf 'FAIL\trunner\tmissing stage script %s\n' "$script" | tee -a "$RESULTS"
    runner_status=1
    [[ "$STOP_ON_FAIL" == 1 ]] && break
    continue
  fi
  log "starting stage=$stage"
  bash "$script" 2>&1 | tee "$RUN_DIR/stages/$(basename "$script").log"
  stage_rc=${PIPESTATUS[0]}
  log "finished stage=$stage exit=$stage_rc"
  if (( stage_rc != 0 )); then
    runner_status=1
    [[ "$STOP_ON_FAIL" == 1 ]] && break
  fi
done

if [[ "$SKIP_CLEANUP" != 1 ]]; then
  bash "$SUITE_DIR/cleanup.sh" 2>&1 | tee "$RUN_DIR/stages/cleanup.log"
  cleanup_rc=${PIPESTATUS[0]}
  (( cleanup_rc != 0 )) && runner_status=1
fi

bash "$SUITE_DIR/report.sh" || report_rc=$?
report_rc=${report_rc:-0}
(( report_rc != 0 )) && runner_status=1
log "runner finished exit=$runner_status report_rc=$report_rc"
exit "$runner_status"
