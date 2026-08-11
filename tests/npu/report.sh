#!/usr/bin/env bash
set -u -o pipefail

source "$(dirname "$0")/lib.sh"

REPORT_MD=${REPORT_MD:-$RUN_DIR/functional_verification_report.md}
REPORT_JSON=${REPORT_JSON:-$RUN_DIR/functional_verification_report.json}
pass_count=$(awk -F '\t' '$1 == "PASS" {n++} END {print n+0}' "$RESULTS" 2>/dev/null || echo 0)
fail_count=$(awk -F '\t' '$1 == "FAIL" {n++} END {print n+0}' "$RESULTS" 2>/dev/null || echo 0)
skip_count=$(awk -F '\t' '$1 == "SKIP" {n++} END {print n+0}' "$RESULTS" 2>/dev/null || echo 0)
test_count=$(awk -F '\t' 'NR > 1 && NF >= 3 {n++} END {print n+0}' "$TEST_SUMMARY" 2>/dev/null || echo 0)
request_count=$(find "$REQUESTS" -type f -name '*.meta.tsv' 2>/dev/null | wc -l | awk '{print $1}')
telemetry_count=$(find "$RUN_DIR/telemetry" -type f -name '*.txt' 2>/dev/null | wc -l | awk '{print $1}')

if (( fail_count > 0 )); then
  verdict=FAIL
elif (( test_count == 0 )); then
  verdict=BLOCKED
elif (( skip_count > 0 )); then
  verdict=PARTIAL
else
  verdict=PASS
fi

{
  printf '# NPU Functional Verification Report\n\n'
  printf -- '- Verdict: **%s**\n' "$verdict"
  printf -- '- Run ID: `%s`\n' "$RUN_ID"
  printf -- '- Generated: `%s`\n' "$(date '+%F %T %z')"
  printf -- '- Host: `%s`\n' "$(hostname -f 2>/dev/null || uname -n)"
  printf -- '- etcd: `%s`, namespace: `%s`\n' "$ETCD_ADDR" "$ETCD_NAMESPACE"
  printf -- '- Model: `%s`\n' "$MODEL_PATH"
  printf -- '- NPU IDs: `%s`\n\n' "${NPU_IDS:-not assigned}"

  printf '## Summary\n\n'
  printf '| Item | Count |\n|---|---:|\n'
  printf '| PASS assertions | %s |\n' "$pass_count"
  printf '| FAIL assertions | %s |\n' "$fail_count"
  printf '| SKIP assertions | %s |\n' "$skip_count"
  printf '| Test stages | %s |\n' "$test_count"
  printf '| Request samples | %s |\n' "$request_count"
  printf '| Telemetry files | %s |\n\n' "$telemetry_count"

  printf '## Stage Results\n\n'
  if [[ -s "$TEST_SUMMARY" ]]; then
    printf '| Test | Status | Failures | Started | Finished | Duration(s) |\n'
    printf '|---|---|---:|---|---|---:|\n'
    tail -n +2 "$TEST_SUMMARY" | awk -F '\t' 'NF >= 6 {printf "| %s | %s | %s | %s | %s | %s |\n", $1,$2,$3,$4,$5,$6}'
  else
    printf 'No stage summary was generated.\n'
  fi
  printf '\n'

  printf '## Assertions\n\n'
  if [[ -s "$RESULTS" ]]; then
    printf '| Status | Test | Detail |\n|---|---|---|\n'
    awk -F '\t' 'NF >= 3 {detail=$3; for (i=4; i<=NF; ++i) detail=detail " " $i; gsub(/\|/, "\\|", detail); printf "| %s | %s | %s |\n", $1,$2,detail}' "$RESULTS"
  else
    printf 'No assertions were recorded.\n'
  fi
  printf '\n'

  printf '## Evidence\n\n'
  printf -- '- Results: `%s`\n' "$RESULTS"
  printf -- '- Stage summary: `%s`\n' "$TEST_SUMMARY"
  printf -- '- Requests: `%s`\n' "$REQUESTS"
  printf -- '- Snapshots: `%s`\n' "$SNAP"
  printf -- '- Periodic telemetry: `%s`\n' "$RUN_DIR/telemetry"
  printf -- '- Resource wait log: `%s`\n\n' "$RUN_DIR/resource_wait.tsv"

  printf '## Interpretation\n\n'
  case "$verdict" in
    PASS) printf 'All recorded stages and assertions passed. This is a functional gate result; performance and long-duration soak conclusions require their dedicated stages.\n' ;;
    PARTIAL) printf 'The run completed without failures but contains skipped assertions. It is not a complete verification gate.\n' ;;
    FAIL) printf 'At least one assertion failed. Use the stage log, request artifacts, metrics, etcd snapshots, and telemetry around the failure timestamp for diagnosis.\n' ;;
    BLOCKED) printf 'No test stage completed. The run is blocked by prerequisites, resource allocation, or an early runner failure.\n' ;;
  esac
} > "$REPORT_MD"

results_json='[]'
summary_json='[]'
if [[ -f "$RESULTS" ]]; then
  results_json=$(jq -R -s 'split("\n") | map(select(length > 0)) | map(split("\t") | {status:.[0], test:.[1], detail:(.[2:] | join(" "))})' "$RESULTS")
fi
if [[ -f "$TEST_SUMMARY" ]]; then
  summary_json=$(tail -n +2 "$TEST_SUMMARY" | jq -R -s 'split("\n") | map(select(length > 0)) | map(split("\t") | {test:.[0], status:.[1], failures:(.[2]|tonumber), started_at:.[3], finished_at:.[4], duration_s:(.[5]|tonumber)})')
fi
jq -n \
  --arg verdict "$verdict" \
  --arg run_id "$RUN_ID" \
  --arg generated_at "$(date '+%F %T %z')" \
  --arg host "$(hostname -f 2>/dev/null || uname -n)" \
  --arg etcd "$ETCD_ADDR" \
  --arg namespace "$ETCD_NAMESPACE" \
  --arg model "$MODEL_PATH" \
  --arg npu_ids "${NPU_IDS:-}" \
  --argjson pass "$pass_count" \
  --argjson fail "$fail_count" \
  --argjson skip "$skip_count" \
  --argjson requests "$request_count" \
  --argjson telemetry_files "$telemetry_count" \
  --argjson results "$results_json" \
  --argjson stages "$summary_json" \
  '{verdict:$verdict,run_id:$run_id,generated_at:$generated_at,host:$host,etcd:$etcd,namespace:$namespace,model:$model,npu_ids:$npu_ids,counts:{pass:$pass,fail:$fail,skip:$skip,requests:$requests,telemetry_files:$telemetry_files},stages:$stages,assertions:$results}' > "$REPORT_JSON"

printf 'report=%s\njson=%s\nverdict=%s\n' "$REPORT_MD" "$REPORT_JSON" "$verdict"
[[ "$verdict" == PASS ]]
