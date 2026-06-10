#!/usr/bin/env bash
# =============================================================================
# xllm-service ↔ vLLM 联通 Demo —— 一键起环境
#
# 起：vLLM(后端) + xllm-service master(编排层) + 把 vLLM 注册进集群
# 复用已在跑的 etcd（:2379）。幂等：已起的组件会跳过。
#
# 共享 GPU 机器：vLLM 固定用卡 3 + gpu-mem 0.18，勿改大。
# =============================================================================
set -uo pipefail

ROOT=/export/home/zhangyi.932/xllm-service
VLLM_VENV=/export/home/zhangyi.932/vllm-venv
MODEL_DIR=/export/home/zhangyi.932/models/Qwen2.5-7B-Instruct
VLLM_LOG=/export/home/zhangyi.932/logs/vllm.log
VLLM_PID=/export/home/zhangyi.932/logs/vllm.pid
MASTER_LOG=/tmp/xllm-master.log
MASTER_PID=/tmp/xllm-master.pid
SIDECAR_LOG=/tmp/xllm-sidecar.log
SIDECAR_PID=/tmp/xllm-sidecar.pid

ETCD=127.0.0.1:2379
VLLM_PORT=18000
HTTP_PORT=9998
RPC_PORT=8889
MODEL_NAME=qwen2.5-7b
INSTANCE_ADDR=127.0.0.1:${VLLM_PORT}

say() { printf '\n\033[1;36m[demo] %s\033[0m\n' "$*"; }
ok()  { printf '\033[1;32m  ✓ %s\033[0m\n' "$*"; }
die() { printf '\033[1;31m  ✗ %s\033[0m\n' "$*"; exit 1; }

# ---- 0. etcd ----------------------------------------------------------------
say "0/4 检查 etcd ($ETCD)"
curl -s -m 3 http://$ETCD/version >/dev/null 2>&1 \
  && ok "etcd 在跑" \
  || die "etcd 不可达，请先启动 etcd（监听 $ETCD）"

# ---- 1. vLLM 后端 -----------------------------------------------------------
say "1/4 vLLM 后端 (:$VLLM_PORT, 卡3)"
if curl -s -m 3 http://127.0.0.1:$VLLM_PORT/v1/models >/dev/null 2>&1; then
  ok "vLLM 已在跑"
else
  TORCH_LIB=$("$VLLM_VENV"/bin/python -c "import torch,os;print(os.path.dirname(torch.__file__)+'/lib')")
  nohup env CUDA_VISIBLE_DEVICES=3 \
    LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}" \
    TMPDIR=/export/home/zhangyi.932/tmp \
    "$VLLM_VENV"/bin/vllm serve "$MODEL_DIR" \
      --host 0.0.0.0 --port $VLLM_PORT --served-model-name $MODEL_NAME \
      --tensor-parallel-size 1 --gpu-memory-utilization 0.18 \
      --max-model-len 8192 --enable-prefix-caching --trust-remote-code \
      > "$VLLM_LOG" 2>&1 &
  echo $! > "$VLLM_PID"
  printf '  起 vLLM (pid=%s) 等待就绪' "$(cat $VLLM_PID)"
  for i in $(seq 1 40); do
    curl -s -m 3 http://127.0.0.1:$VLLM_PORT/v1/models >/dev/null 2>&1 && break
    ps -p "$(cat $VLLM_PID)" >/dev/null 2>&1 || die "vLLM 进程退出，看 $VLLM_LOG"
    printf '.'; sleep 5
  done
  echo
  curl -s -m 3 http://127.0.0.1:$VLLM_PORT/v1/models >/dev/null 2>&1 \
    && ok "vLLM 就绪" || die "vLLM 就绪超时，看 $VLLM_LOG"
fi

# ---- 2. xllm-service master -------------------------------------------------
say "2/4 xllm-service master (http :$HTTP_PORT, rpc :$RPC_PORT, backend=vllm)"
if [ -f "$MASTER_PID" ] && ps -p "$(cat $MASTER_PID)" >/dev/null 2>&1; then
  ok "master 已在跑 (pid=$(cat $MASTER_PID))"
else
  nohup "$ROOT"/build/xllm_service/xllm_master_serving \
    --default_backend_type=vllm \
    --etcd_addr=$ETCD \
    --http_server_port=$HTTP_PORT \
    --rpc_server_port=$RPC_PORT \
    > "$MASTER_LOG" 2>&1 &
  echo $! > "$MASTER_PID"
  sleep 4
  ps -p "$(cat $MASTER_PID)" >/dev/null 2>&1 \
    && ok "master 起来了 (pid=$(cat $MASTER_PID))" \
    || die "master 启动失败，看 $MASTER_LOG"
fi

# ---- 3. sidecar 自动注册 vLLM 实例 ------------------------------------------
say "3/4 启动 sidecar 自动注册 vLLM 实例 ($INSTANCE_ADDR)"
if [ -f "$SIDECAR_PID" ] && ps -p "$(cat $SIDECAR_PID)" >/dev/null 2>&1; then
  ok "sidecar 已在跑 (pid=$(cat $SIDECAR_PID))"
else
  nohup env PYTHONPATH="$ROOT" python3 -m xllm_service.vllm_sidecar.sidecar \
    --etcd-endpoints "$ETCD" \
    --vllm-url "http://127.0.0.1:$VLLM_PORT" \
    --register-addr "$INSTANCE_ADDR" \
    > "$SIDECAR_LOG" 2>&1 &
  echo $! > "$SIDECAR_PID"
  ok "sidecar 起来了 (pid=$(cat $SIDECAR_PID))：自动注册 + 租约续租 + 健康门控"
fi
# 旧的手动注册方式仍可用作降级： bash demo/register_vllm.sh "$INSTANCE_ADDR"
sleep 2
# readiness gate：有实例后 master 才放开 HTTP 入口
for i in $(seq 1 10); do
  curl -s -m 3 http://127.0.0.1:$HTTP_PORT/v1/models >/dev/null 2>&1 && break
  sleep 1
done
curl -s -m 3 http://127.0.0.1:$HTTP_PORT/v1/models >/dev/null 2>&1 \
  && ok "xllm-service HTTP 入口就绪 (:$HTTP_PORT)" \
  || die "HTTP 入口未就绪，看 $MASTER_LOG"

# ---- 4. 完成 ----------------------------------------------------------------
say "4/4 环境就绪 ✅  下一步跑演示："
cat <<EOF
  bash $ROOT/demo/demo_show.sh        # 分步联通演示（录屏主体）
  bash $ROOT/demo/demo_compare.sh     # 直连 vs 经service 透明性对比
  bash $ROOT/demo/demo_down.sh        # 清理（停 master+vLLM，保留 etcd）

  统一入口:  http://127.0.0.1:$HTTP_PORT   (OpenAI 兼容)
  后端 vLLM: http://127.0.0.1:$VLLM_PORT
EOF
