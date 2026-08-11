#!/usr/bin/env bash
set -u -o pipefail

TAG=${1:?telemetry tag is required}
shift
source "$(dirname "$0")/lib.sh"

TELEMETRY_DIR="$RUN_DIR/telemetry/$TAG"
METRICS_DIR="$TELEMETRY_DIR/metrics"
NPU_DIR="$TELEMETRY_DIR/npu"
ETCD_DIR="$TELEMETRY_DIR/etcd"
mkdir -p "$METRICS_DIR" "$NPU_DIR" "$ETCD_DIR"
printf 'timestamp\tport\tlivez\treadyz\tmetrics_file\n' > "$TELEMETRY_DIR/http.tsv"
printf 'timestamp\tetcd_snapshot\tnpu_snapshot\n' > "$TELEMETRY_DIR/samples.tsv"

trap 'exit 0' TERM INT HUP

while :; do
  stamp=$(date +%s%3N)
  npu_file="$NPU_DIR/${stamp}.txt"
  etcd_file="$ETCD_DIR/${stamp}.tsv"
  if command -v npu-smi >/dev/null 2>&1; then
    timeout "$NPU_SMI_TIMEOUT" npu-smi info > "$npu_file" 2>&1 || true
  else
    printf 'npu-smi unavailable\n' > "$npu_file"
  fi
  etcd_snapshot_text > "$etcd_file" 2>"$ETCD_DIR/${stamp}.error" || true
  for port in "$@"; do
    metrics_file="$METRICS_DIR/${stamp}_${port}.txt"
    curl -sS --max-time 10 "http://127.0.0.1:$port/metrics" > "$metrics_file" 2>"$metrics_file.error" || true
    livez=$(http_code "http://127.0.0.1:$port/livez")
    readyz=$(http_code "http://127.0.0.1:$port/readyz")
    printf '%s\t%s\t%s\t%s\t%s\n' "$stamp" "$port" "$livez" "$readyz" "$metrics_file" >> "$TELEMETRY_DIR/http.tsv"
  done
  printf '%s\t%s\t%s\n' "$stamp" "$etcd_file" "$npu_file" >> "$TELEMETRY_DIR/samples.tsv"
  sleep "$TELEMETRY_INTERVAL"
done
