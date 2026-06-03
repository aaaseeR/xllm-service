#!/usr/bin/env bash
# =============================================================================
# 把一个 vLLM 实例注册进 xllm-service 集群。
#
# 用法: register_vllm.sh [addr]    addr 默认 127.0.0.1:18000
#
# 原理：xllm-service 通过 watch etcd 的 XLLM:DEFAULT: 前缀发现实例。这里直接写
# 一条 InstanceMetaInfo JSON（backend_type=vllm, type=0=DEFAULT），master 的
# watcher 会自动 register_instance → 建 HTTP channel → 实例上线。
#
# 注意 type 必须是 0(DEFAULT)：单实例无 decode 时调度器只放行 DEFAULT 类型。
# （后续会用 vLLM sidecar 自动做这件事，替代手动写 etcd。）
# =============================================================================
set -euo pipefail

ADDR="${1:-127.0.0.1:18000}"
ETCD="${ETCD:-127.0.0.1:2379}"
KEY="XLLM:DEFAULT:${ADDR}"

JSON=$(cat <<EOF
{"name":"${ADDR}","rpc_address":"${ADDR}","type":0,"backend_type":"vllm","incarnation_id":"vllm-1","register_ts_ms":1}
EOF
)

etcdctl --endpoints="$ETCD" put "$KEY" "$JSON" >/dev/null
echo "  ✓ 已注册: $KEY"
echo "    $JSON"
