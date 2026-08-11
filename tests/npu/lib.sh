#!/usr/bin/env bash

# Shared helpers for black-box NPU experiments.  The helpers only touch the
# configured etcd namespace and PIDs recorded by this test run.

set -o pipefail

SUITE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SERVICE_ROOT=$(cd "$SUITE_DIR/../.." && pwd)
NEW_SERVICE_ROOT=$(cd "$SERVICE_ROOT/.." && pwd)
RUN_ROOT=${RUN_ROOT:-${NPU_TEST_ROOT:-$SUITE_DIR/run}}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}
RUN_DIR=${RUN_ROOT}/${RUN_ID}
STATE=${STATE:-$RUN_DIR/state}
LOGS=${LOGS:-$RUN_DIR/logs}
SNAP=${SNAP:-$RUN_DIR/snap}
REQUESTS=${REQUESTS:-$RUN_DIR/requests}
RESULTS=${RESULTS:-$RUN_DIR/results.tsv}
TEST_SUMMARY=${TEST_SUMMARY:-$RUN_DIR/test_summary.tsv}

SERVICE_BIN=${SERVICE_BIN:-$SERVICE_ROOT/build/xllm_service/xllm_master_serving}
XLLM_BIN=${XLLM_BIN:-$NEW_SERVICE_ROOT/xllm/build/xllm/core/server/xllm}
MODEL_PATH=${MODEL_PATH:-/export/home/models/Qwen3-0.6B}
REQUEST_MODEL=${REQUEST_MODEL:-$(basename "$MODEL_PATH")}
ETCD_ADDR=${ETCD_ADDR:-127.0.0.1:2379}
ETCD_URL=${ETCD_URL:-http://${ETCD_ADDR}}
ETCD_NAMESPACE=${ETCD_NAMESPACE:-xllm-npu-${USER:-unknown}-${RUN_ID}}
SERVICE_HTTP_BASE=${SERVICE_HTTP_BASE:-31000}
SERVICE_RPC_BASE=${SERVICE_RPC_BASE:-31100}
ENGINE_HTTP_BASE=${ENGINE_HTTP_BASE:-32000}
ENGINE_MASTER_BASE=${ENGINE_MASTER_BASE:-32100}
ENGINE_PD_BASE=${ENGINE_PD_BASE:-32200}
ENGINE_TRANSFER_BASE=${ENGINE_TRANSFER_BASE:-32300}
ENGINE_HCCL_BASE=${ENGINE_HCCL_BASE:-32400}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-180}
REGISTER_TIMEOUT=${REGISTER_TIMEOUT:-240}
READY_TIMEOUT=${READY_TIMEOUT:-90}
NPU_SMI_TIMEOUT=${NPU_SMI_TIMEOUT:-15}
NPU_FREE_HBM_MB=${NPU_FREE_HBM_MB:-1024}
WAIT_INTERVAL=${WAIT_INTERVAL:-30}
TELEMETRY_INTERVAL=${TELEMETRY_INTERVAL:-5}
XLLM_ENABLE_PREFIX_CACHE=${XLLM_ENABLE_PREFIX_CACHE:-false}
XLLM_ENABLE_CHUNKED_PREFILL=${XLLM_ENABLE_CHUNKED_PREFILL:-true}
XLLM_DISABLE_TTFT_PROFILING=${XLLM_DISABLE_TTFT_PROFILING:-false}
CLEAN_NAMESPACE=${CLEAN_NAMESPACE:-1}

mkdir -p "$STATE" "$LOGS" "$SNAP" "$REQUESTS"

if [[ ! -f "$TEST_SUMMARY" ]]; then
  printf 'test\tstatus\tfailures\tstarted_at\tfinished_at\tduration_s\n' > "$TEST_SUMMARY"
fi

NS_PREFIX=""
_ns=${ETCD_NAMESPACE#/}
_ns=${_ns%/}
if [[ -n "$_ns" ]]; then
  NS_PREFIX="/$_ns/"
fi

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
pass() { printf 'PASS\t%s\t%s\n' "${TEST_NAME:-unknown}" "$*" | tee -a "$RESULTS"; }
fail() { FAIL_COUNT=$((FAIL_COUNT + 1)); printf 'FAIL\t%s\t%s\n' "${TEST_NAME:-unknown}" "$*" | tee -a "$RESULTS"; }
skip() { printf 'SKIP\t%s\t%s\n' "${TEST_NAME:-unknown}" "$*" | tee -a "$RESULTS"; }
finish() {
  local rc=0 status end_iso duration
  if [[ ${FAIL_COUNT:-0} -ne 0 ]]; then
    log "result=FAIL failures=$FAIL_COUNT run_dir=$RUN_DIR"
    rc=1
    status=FAIL
  else
    log "result=PASS run_dir=$RUN_DIR"
    status=PASS
  fi
  end_iso=$(date '+%F %T %z')
  duration=$(( $(date +%s) - ${TEST_START_EPOCH:-$(date +%s)} ))
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${TEST_NAME:-unknown}" "$status" "${FAIL_COUNT:-0}" \
    "${TEST_START_ISO:-unknown}" "$end_iso" "$duration" >> "$TEST_SUMMARY"
  if [[ "${AUTO_REPORT:-1}" == 1 && -x "$SUITE_DIR/report.sh" ]]; then
    bash "$SUITE_DIR/report.sh" > "$RUN_DIR/report_${TEST_NAME:-unknown}.log" 2>&1 || true
  fi
  return "$rc"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing command: $1" >&2
    return 1
  }
}

check_common() {
  local missing=0
  for c in curl jq awk sed sha256sum ps timeout ctest; do
    need_cmd "$c" || missing=1
  done
  [[ -x "$SERVICE_BIN" ]] || { echo "missing service binary: $SERVICE_BIN" >&2; missing=1; }
  [[ -x "$XLLM_BIN" ]] || { echo "missing xllm binary: $XLLM_BIN" >&2; missing=1; }
  [[ -d "$MODEL_PATH" ]] || { echo "missing model path: $MODEL_PATH" >&2; missing=1; }
  [[ -n "${NPU_IDS:-}" ]] || { echo "NPU_IDS is empty; set e.g. NPU_IDS=0,1" >&2; missing=1; }
  return "$missing"
}

load_ascend_env() {
  set +u
  [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]] &&
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
  [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]] &&
    source /usr/local/Ascend/nnal/atb/set_env.sh
  set -u
}

etcd_alive() {
  curl -fsS --max-time 5 "$ETCD_URL/version" >/dev/null 2>&1
}

etcd_range_json() {
  curl -fsS --max-time 15 "$ETCD_URL/v3/kv/range" \
    -H 'Content-Type: application/json' -X POST \
    -d '{"key":"AA==","range_end":"AA==","limit":10000}'
}

etcd_namespace_json() {
  local prefix_b64
  prefix_b64=$(printf '%s' "$NS_PREFIX" | base64 -w0)
  etcd_range_json | jq --arg prefix "$prefix_b64" \
    '.kvs = [(.kvs[]? | select(.key | startswith($prefix)))]'
}

etcd_snapshot_text() {
  etcd_namespace_json | jq -r \
    '.kvs[]? | [(.key|@base64d),(.value|@base64d),(.mod_revision|tostring),(.lease|tostring)] | @tsv'
}

snapshot() {
  local tag=${1:-snapshot}
  local dir="$SNAP/$tag"
  mkdir -p "$dir"
  date '+%F %T %z' > "$dir/time.txt"
  etcd_namespace_json > "$dir/etcd.json" 2>"$dir/etcd.error" || true
  etcd_snapshot_text > "$dir/etcd.tsv" 2>"$dir/etcd_decode.error" || true
  {
    printf 'namespace=%s\n' "$ETCD_NAMESPACE"
    printf 'service_bin=%s\n' "$SERVICE_BIN"
    printf 'xllm_bin=%s\n' "$XLLM_BIN"
    printf 'model_path=%s\n' "$MODEL_PATH"
    printf 'host=%s\n' "$(hostname -f 2>/dev/null || uname -n)"
    printf 'kernel=%s\n' "$(uname -a)"
  } > "$dir/manifest.txt"
  ps -eo pid,ppid,pgid,stat,etime,args > "$dir/processes.txt" 2>&1 || true
  ss -ltnp > "$dir/listen.txt" 2>&1 || true
  if command -v npu-smi >/dev/null 2>&1; then
    timeout "$NPU_SMI_TIMEOUT" npu-smi info > "$dir/npu-smi.txt" 2>&1 || true
  fi
  for port in "$@"; do
    [[ "$port" == "$tag" ]] && continue
    curl -sS --max-time 10 "http://127.0.0.1:$port/livez" > "$dir/livez_$port.body" 2>"$dir/livez_$port.error" || true
    curl -sS --max-time 10 "http://127.0.0.1:$port/readyz" > "$dir/readyz_$port.body" 2>"$dir/readyz_$port.error" || true
    curl -sS --max-time 10 "http://127.0.0.1:$port/metrics" > "$dir/metrics_$port.txt" 2>"$dir/metrics_$port.error" || true
  done
  log "snapshot=$dir"
}

namespace_lines() {
  etcd_snapshot_text | awk -F '\t' -v p="$NS_PREFIX" 'index($1, p "XLLM") == 1'
}

logical_lines() {
  namespace_lines | awk -F '\t' -v p="$NS_PREFIX" \
    '{print substr($1, length(p) + 1) "\t" $2 "\t" $3 "\t" $4}'
}

role_count() {
  local role=$1
  namespace_lines | awk -F '\t' -v p="$NS_PREFIX" -v r="$role" \
    'index($1, p "XLLM:" r ":") == 1 {n++} END {print n+0}'
}

role_hash() {
  local role=$1
  namespace_lines | awk -F '\t' -v p="$NS_PREFIX" -v r="$role" \
    'index($1, p "XLLM:" r ":") == 1 {print}' | sort | sha256sum | awk '{print $1}'
}

key_count() {
  local prefix=$1
  logical_lines | awk -F '\t' -v p="$prefix" 'index($1, p) == 1 {n++} END {print n+0}'
}

cleanup_namespace() {
  local key encoded body
  while IFS=$'\t' read -r key _value _revision _lease; do
    [[ -n "$key" ]] || continue
    encoded=$(printf '%s' "$key" | base64 -w0)
    body=$(jq -nc --arg key "$encoded" '{key:$key}')
    curl -fsS --max-time 10 "$ETCD_URL/v3/kv/deleterange" \
      -H 'Content-Type: application/json' -X POST -d "$body" >/dev/null || true
  done < <(namespace_lines)
}

maybe_clean_namespace() {
  [[ "$CLEAN_NAMESPACE" == 1 ]] || return 0
  etcd_alive || return 1
  log "cleaning only etcd namespace=$ETCD_NAMESPACE"
  cleanup_namespace
}

http_code() {
  curl -sS -o /dev/null -w '%{http_code}' --max-time 10 "$1" 2>/dev/null || echo 000
}

wait_code() {
  local url=$1 expected=$2 timeout_s=${3:-60} end=$((SECONDS + timeout_s)) code
  while (( SECONDS < end )); do
    code=$(http_code "$url")
    [[ "$code" == "$expected" ]] && return 0
    sleep 1
  done
  return 1
}

wait_role_count() {
  local role=$1 expected=$2 timeout_s=${3:-$REGISTER_TIMEOUT} count
  local end=$((SECONDS + timeout_s))
  while (( SECONDS < end )); do
    if etcd_alive; then
      count=$(role_count "$role" 2>/dev/null || echo -1)
      [[ "$count" == "$expected" ]] && return 0
    fi
    sleep 2
  done
  return 1
}

wait_all_roles() {
  local timeout_s=${1:-$REGISTER_TIMEOUT}
  wait_role_count DEFAULT 0 1 2>/dev/null || true
  local end=$((SECONDS + timeout_s))
  while (( SECONDS < end )); do
    local p d def
    p=$(role_count PREFILL); d=$(role_count DECODE); def=$(role_count DEFAULT)
    if (( p + d + def > 0 )); then return 0; fi
    sleep 2
  done
  return 1
}

pid_alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }

start_bg() {
  local name=$1 logfile=$2
  shift 2
  mkdir -p "$(dirname "$logfile")"
  setsid "$@" > "$logfile" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "$pid" > "$STATE/$name.pid"
  printf '%q ' "$@" > "$STATE/$name.command"
  printf '\n' >> "$STATE/$name.command"
  log "started $name pid=$pid log=$logfile"
  echo "$pid"
}

stop_named() {
  local name=$1 pid_file="$STATE/$1.pid" pid
  [[ -f "$pid_file" ]] || return 0
  pid=$(cat "$pid_file" 2>/dev/null || true)
  pid_alive "$pid" || return 0
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  local end=$((SECONDS + 20))
  while pid_alive "$pid" && (( SECONDS < end )); do sleep 1; done
  if pid_alive "$pid"; then
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}

service_http() { echo $((SERVICE_HTTP_BASE + $1)); }
service_rpc() { echo $((SERVICE_RPC_BASE + $1)); }
engine_http() { echo $((ENGINE_HTTP_BASE + $1)); }
engine_master() { echo $((ENGINE_MASTER_BASE + $1)); }
engine_pd() { echo $((ENGINE_PD_BASE + $1)); }
engine_transfer() { echo $((ENGINE_TRANSFER_BASE + $1)); }
engine_hccl() { echo $((ENGINE_HCCL_BASE + $1 * 100)); }

start_service() {
  local index=$1 http rpc
  http=$(service_http "$index"); rpc=$(service_rpc "$index")
  local args=("$SERVICE_BIN"
    "--http_server_port=$http"
    "--rpc_server_port=$rpc"
    "--etcd_addr=$ETCD_ADDR"
    "--etcd_namespace=$ETCD_NAMESPACE"
    "--tokenizer_path=$MODEL_PATH")
  if [[ -n "${PLACEMENT_CONFIG_PATH:-}" ]]; then
    args+=("--placement_config_path=$PLACEMENT_CONFIG_PATH"
      "--placement_mode_override=${PLACEMENT_MODE_OVERRIDE:-1}")
  fi
  start_bg "service$index" "$LOGS/service_${index}.log" "${args[@]}" >/dev/null
}

start_engine() {
  local name=$1 role=$2 card=$3 index=${4:-0} host=${5:-127.0.0.1}
  local http master pd transfer hccl
  local nnodes=${XLLM_NNODES:-1}
  local node_rank=${XLLM_NODE_RANK:-0}
  local master_addr=${XLLM_MASTER_NODE_ADDR:-}
  http=$(engine_http "$index"); master=$(engine_master "$index")
  pd=$(engine_pd "$index"); transfer=$(engine_transfer "$index"); hccl=$(engine_hccl "$index")
  [[ -n "$master_addr" ]] || master_addr="$host:$master"
  local args=("$XLLM_BIN"
    "--model=$MODEL_PATH"
    "--devices=npu:$card"
    "--host=$host"
    "--port=$http"
    "--master_node_addr=$master_addr"
    "--nnodes=$nnodes"
    "--node_rank=$node_rank"
    "--max_memory_utilization=${XLLM_MAX_MEMORY_UTILIZATION:-0.5}"
    "--block_size=${XLLM_BLOCK_SIZE:-128}"
    "--communication_backend=hccl"
    "--enable_prefix_cache=$XLLM_ENABLE_PREFIX_CACHE"
    "--enable_chunked_prefill=$XLLM_ENABLE_CHUNKED_PREFILL"
    "--enable_schedule_overlap=${XLLM_ENABLE_SCHEDULE_OVERLAP:-false}"
    "--enable_service_routing=true"
    "--etcd_addr=$ETCD_ADDR"
    "--etcd_namespace=$ETCD_NAMESPACE")
  if [[ "$role" != DEFAULT ]]; then
    args+=("--enable_disagg_pd=true"
      "--instance_role=$role"
      "--disagg_pd_port=$pd"
      "--transfer_listen_port=$transfer")
  fi
  if [[ "$XLLM_DISABLE_TTFT_PROFILING" == true ]]; then
    args+=("--disable_ttft_profiling=true")
  fi
  local previous_hccl=${HCCL_IF_BASE_PORT-}
  export HCCL_IF_BASE_PORT="$hccl"
  start_bg "$name" "$LOGS/${name}.log" "${args[@]}" >/dev/null
  if [[ -n "$previous_hccl" ]]; then
    export HCCL_IF_BASE_PORT="$previous_hccl"
  else
    unset HCCL_IF_BASE_PORT
  fi
}

metric_value() {
  local port=$1 name=$2
  curl -fsS --max-time 10 "http://127.0.0.1:$port/metrics" 2>/dev/null |
    awk -v n="$name" '$1 == n {print $2; found=1} END {if (!found) print ""}' | head -1
}

metric_present() {
  local port=$1 name=$2
  curl -fsS --max-time 10 "http://127.0.0.1:$port/metrics" 2>/dev/null |
    awk -v n="$name" '$1 == n {found=1} END {exit(found ? 0 : 1)}'
}

request_once() {
  local port=$1 tag=$2 prompt=$3 max_tokens=${4:-16}
  local body="$REQUESTS/${tag}.request.json"
  local response="$REQUESTS/${tag}.response.json"
  local meta="$REQUESTS/${tag}.meta.tsv"
  jq -nc --arg model "$REQUEST_MODEL" --arg prompt "$prompt" \
    --argjson max_tokens "$max_tokens" \
    '{model:$model,messages:[{role:"user",content:$prompt}],max_tokens:$max_tokens,temperature:0.0,stream:false}' > "$body"
  curl -sS --max-time "$REQUEST_TIMEOUT" \
    -H 'Content-Type: application/json' \
    -o "$response" \
    -w '%{http_code}\t%{time_total}\n' \
    "http://127.0.0.1:$port/v1/chat/completions" \
    -d @"$body" > "$meta" 2>"$REQUESTS/${tag}.curl.error" || true
  awk -F '\t' '$1 == 200 {ok=1} END {exit(ok ? 0 : 1)}' "$meta" &&
    jq -e '(.choices|length)>0 and ((.choices[0].message.content // .choices[0].text // "") | type == "string")' \
      "$response" >/dev/null 2>&1
}

collect_request_metrics() {
  local port=$1 tag=$2
  curl -sS --max-time 10 "http://127.0.0.1:$port/metrics" \
    > "$SNAP/${tag}_metrics.txt" 2>"$SNAP/${tag}_metrics.error" || true
}

record_process_snapshot() {
  local tag=$1
  ps -eo pid,ppid,pgid,stat,etime,args > "$SNAP/${tag}_processes.txt" 2>&1 || true
  if command -v npu-smi >/dev/null 2>&1; then
    timeout "$NPU_SMI_TIMEOUT" npu-smi info > "$SNAP/${tag}_npu-smi.txt" 2>&1 || true
  fi
}

free_npu_ids() {
  if [[ -n "${NPU_IDS:-}" ]]; then
    tr ', ' '\n\n' <<< "$NPU_IDS" |
      awk '/^[0-9]+$/ && !seen[$1]++ {print $1}' | sort -n
    return 0
  fi
  if [[ -n "${NPU_DISCOVERY_CMD:-}" ]]; then
    bash -c "$NPU_DISCOVERY_CMD" |
      awk '/^[0-9]+$/ && !seen[$1]++ {print $1}' | sort -n
    return 0
  fi
  command -v npu-smi >/dev/null 2>&1 || return 1
  timeout "$NPU_SMI_TIMEOUT" npu-smi info 2>/dev/null |
    awk -v limit="$NPU_FREE_HBM_MB" '
      /0000:/ {
        physical=$3
        used=""
        for (i=1; i<=NF; ++i) {
          if ($i == "/" && $(i-1) ~ /^[0-9]+$/) used=$(i-1)
          if ($i ~ /^[0-9]+\/$/) {used=$i; sub("/", "", used)}
        }
        if (physical ~ /^[0-9]+$/ && used ~ /^[0-9]+$/ && used+0 < limit)
          print physical
      }
    ' | awk '/^[0-9]+$/ && !seen[$1]++ {print $1}' | sort -n
}

wait_for_free_npus() {
  local required=${1:-2} output count now wait_started=$SECONDS
  local wait_log="$RUN_DIR/resource_wait.tsv"
  if [[ ! -f "$wait_log" ]]; then
    printf 'timestamp\trequired\tcandidates\tstatus\n' > "$wait_log"
  fi
  while :; do
    if [[ "${RESOURCE_WAIT_TIMEOUT:-0}" != 0 ]] &&
       (( SECONDS - wait_started >= RESOURCE_WAIT_TIMEOUT )); then
      printf '%s\t%s\t\tTIMEOUT\n' "$(date '+%F %T %z')" "$required" >> "$wait_log"
      return 1
    fi
    output=$(free_npu_ids 2>"$RUN_DIR/resource_wait.error" || true)
    count=$(printf '%s\n' "$output" | awk 'NF {n++} END {print n+0}')
    now=$(date '+%F %T %z')
    if (( count >= required )); then
      printf '%s\t%s\t%s\tREADY\n' "$now" "$required" "$(tr '\n' ',' <<< "$output")" >> "$wait_log"
      printf '%s\n' "$output" | head -n "$required"
      return 0
    fi
    printf '%s\t%s\t%s\tWAITING\n' "$now" "$required" "$(tr '\n' ',' <<< "$output")" >> "$wait_log"
    log "waiting for NPU cards: have=$count required=$required interval=${WAIT_INTERVAL}s" >&2
    sleep "$WAIT_INTERVAL"
  done
}

start_telemetry_sampler() {
  local tag=$1
  shift
  local args=(env
    "RUN_ROOT=$RUN_ROOT"
    "RUN_ID=$RUN_ID"
    "STATE=$STATE"
    "LOGS=$LOGS"
    "SNAP=$SNAP"
    "REQUESTS=$REQUESTS"
    "RESULTS=$RESULTS"
    "TEST_SUMMARY=$TEST_SUMMARY"
    "ETCD_ADDR=$ETCD_ADDR"
    "ETCD_URL=$ETCD_URL"
    "ETCD_NAMESPACE=$ETCD_NAMESPACE"
    "NPU_SMI_TIMEOUT=$NPU_SMI_TIMEOUT"
    "TELEMETRY_INTERVAL=$TELEMETRY_INTERVAL"
    "$SUITE_DIR/telemetry_sampler.sh"
    "$tag")
  args+=("$@")
  start_bg "telemetry_$tag" "$LOGS/telemetry_${tag}.log" "${args[@]}" >/dev/null
}
