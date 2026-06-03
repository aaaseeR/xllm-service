#!/usr/bin/env bash
# =============================================================================
# 透明性证明：同样的请求(seed=42,temperature=0)，直连 vLLM vs 经 xllm-service，
# 输出应逐字一致 —— 证明 xllm-service 是“透明转发”，没有改写/破坏语义。
# 再用 vLLM 访问日志佐证请求确实是 service 转发进去的。
# =============================================================================
set -uo pipefail
VLLM=127.0.0.1:18000
HTTP=127.0.0.1:9998
VLLM_LOG=/export/home/zhangyi.932/logs/vllm.log

PAYLOAD='{"model":"qwen2.5-7b","messages":[{"role":"user","content":"用一句话介绍杭州这座城市"}],"max_tokens":60,"temperature":0,"seed":42}'
get_content() { python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])"; }

echo "请求体(两边完全相同): $PAYLOAD"; echo

DIRECT=$(curl -s -m 30 http://$VLLM/v1/chat/completions  -H 'Content-Type: application/json' -d "$PAYLOAD" | get_content)
VIA=$(   curl -s -m 30 http://$HTTP/v1/chat/completions  -H 'Content-Type: application/json' -d "$PAYLOAD" | get_content)

printf '  直连 vLLM      (:%s) : %s\n' "${VLLM##*:}" "$DIRECT"
printf '  经 xllm-service(:%s) : %s\n' "${HTTP##*:}" "$VIA"
echo
if [ "$DIRECT" = "$VIA" ]; then
  printf '\033[1;32m  ✓ 输出逐字一致 —— xllm-service 转发透明，未破坏语义\033[0m\n'
else
  printf '\033[1;31m  ✗ 输出不一致（检查 seed/温度/采样）\033[0m\n'
fi

echo; echo "vLLM 访问日志（佐证请求经 service 转发进来）："
grep "POST /v1/chat/completions" "$VLLM_LOG" | tail -2
