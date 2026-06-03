#!/usr/bin/env bash
# 清理 demo：停 master + vLLM。保留 etcd（可能别的东西在用）。
set -uo pipefail
MASTER_PID=/tmp/xllm-master.pid
VLLM_PID=/export/home/zhangyi.932/logs/vllm.pid
ETCD=127.0.0.1:2379

[ -f "$MASTER_PID" ] && kill "$(cat $MASTER_PID)" 2>/dev/null && echo "  ✓ 停 master ($(cat $MASTER_PID))" || echo "  - master 未在跑"
[ -f "$VLLM_PID" ]   && kill "$(cat $VLLM_PID)"   2>/dev/null && echo "  ✓ 停 vLLM ($(cat $VLLM_PID))，释放卡3" || echo "  - vLLM 未在跑"
# 注销 etcd 里的实例 key（避免下次脏数据）
etcdctl --endpoints=$ETCD del --prefix XLLM:DEFAULT: >/dev/null 2>&1 && echo "  ✓ 清理 etcd 实例注册" || true
echo "  （etcd 进程保留未动）"
