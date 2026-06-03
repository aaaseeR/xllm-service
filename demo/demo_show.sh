#!/usr/bin/env bash
# =============================================================================
# xllm-service ↔ vLLM 联通演示（录屏主体）。每步暂停，按回车推进。
# 设 PAUSE=0 可关闭暂停一口气跑完。
# =============================================================================
set -uo pipefail
HTTP=127.0.0.1:9998        # xllm-service 统一入口
ETCD=127.0.0.1:2379
MASTER_LOG=/tmp/xllm-master.log
PAUSE="${PAUSE:-1}"

hr()    { printf '\033[1;34m──────────────────────────────────────────────────────────\033[0m\n'; }
title() { hr; printf '\033[1;36m▶ %s\033[0m\n' "$*"; hr; }
run()   { printf '\033[1;33m$ %s\033[0m\n' "$*"; eval "$*"; echo; }
pause() { [ "$PAUSE" = "1" ] && read -rp $'\033[2m  [回车继续]\033[0m' _ || true; }

clear
title "镜头1 ── 集群里有哪些后端实例？"
echo "xllm-service 通过 etcd 发现后端。看看注册了什么："
run "etcdctl --endpoints=$ETCD get --prefix XLLM:DEFAULT: | tail -1 | python3 -m json.tool"
echo "↑ 一个 backend_type=vllm 的实例。master 是怎么接纳它的："
run "grep -E 'Register a new|instances available' $MASTER_LOG | tail -2"
pause

title "镜头2 ── 统一入口看到的模型 (GET /v1/models)"
echo "客户端只跟 xllm-service($HTTP) 说话，不需要知道后端是 vLLM："
run "curl -s http://$HTTP/v1/models | python3 -m json.tool | head -8"
pause

title "镜头3 ── 非流式 chat (POST /v1/chat/completions)"
run "curl -s http://$HTTP/v1/chat/completions -H 'Content-Type: application/json' \\
  -d '{\"model\":\"qwen2.5-7b\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句话介绍你自己\"}],\"max_tokens\":64}' \\
  | python3 -c \"import sys,json;r=json.load(sys.stdin);print('回答:',r['choices'][0]['message']['content']);print('用量:',r['usage'])\""
pause

title "镜头4 ── 流式 chat (SSE，实时逐字)"
echo "OpenAI 风格的 SSE 透明转发，逐 token 下发，最后 data: [DONE]："
run "curl -sN http://$HTTP/v1/chat/completions -H 'Content-Type: application/json' \\
  -d '{\"model\":\"qwen2.5-7b\",\"messages\":[{\"role\":\"user\",\"content\":\"从1数到5\"}],\"max_tokens\":40,\"stream\":true}'"
hr
echo "演示完毕。下钻技术细节/透明性证明见：bash demo/demo_compare.sh"
