# Demo Runbook — xllm-service ↔ vLLM 联通

> **一句话成果**：xLLM 的服务编排层（xllm-service）已能把标准 OpenAI 请求透明转发到 vLLM 推理后端——
> 客户端零改造，统一入口接入异构推理引擎，单实例端到端跑通 chat / completions / models（含流式）。

录屏/截图素材。先 `bash demo/demo_up.sh` 起好环境，再按下面 4 个镜头录。全程在终端，约 3 分钟。

---

## 架构图（开场放这张，讲“为什么”）

```
        OpenAI 标准请求 (curl / openai SDK，零改造)
                       │
                       ▼
        ┌─────────────────────────────────┐
        │   xllm-service  (统一编排入口)    │   :9998  OpenAI 兼容
        │   · 服务发现 (etcd watch)         │
        │   · 调度 / 负载均衡               │
        │   · 按 backend_type 分流转发      │
        └───────────────┬─────────────────┘
            brpc(xLLM)   │   HTTP 透传(vLLM)   ← 本次接回的链路
                         ▼
                 ┌───────────────┐
                 │  vLLM backend │   :18000  Qwen2.5-7B
                 └───────────────┘
        发现来源: etcd  XLLM:DEFAULT:127.0.0.1:18000  {backend_type:"vllm"}
```

**讲价值（非技术也懂）**：xllm-service 像一个“推理网关”——上层应用只认一个 OpenAI 入口，
底层换成 vLLM / xLLM / 未来其它引擎都无感。本次打通了 vLLM 这条后端。

---

## 录制前准备（不入镜，或快进）

```bash
cd /export/home/zhangyi.932/xllm-service
bash demo/demo_up.sh      # 起 vLLM + master + 注册，全部就绪后打印入口地址
```
> 复用已在跑的 etcd(:2379)。vLLM 固定卡 3、gpu-mem 0.18（共享 GPU，勿调大）。

---

## 分镜脚本

录制主体一条命令：`bash demo/demo_show.sh`（每镜头按回车推进，节奏可控）。
下面给每个镜头的**真实预期画面**和**字幕/截图点**。

### 镜头 1 — 后端实例如何被发现（30s）
命令：`etcdctl get --prefix XLLM:DEFAULT:` + `grep master 日志`
预期画面：
```json
{ "name": "127.0.0.1:18000", "backend_type": "vllm", "type": 0, ... }
```
```
instance_mgr.cpp:1381] Register a new default instance : 127.0.0.1:18000
master.cpp:108] HTTP server started, instances available, endpoint: 0.0.0.0:9998
```
- **字幕**：“vLLM 实例注册进集群 → master 自动发现并建立转发通道 → 入口放行。”
- **截图点**：`backend_type:"vllm"` 这一行 + “instances available”这一行（说明有了健康后端才对外服务）。

### 镜头 2 — 统一入口的模型列表（20s）
命令：`curl http://127.0.0.1:9998/v1/models`
预期画面：`"id": "qwen2.5-7b" ... "owned_by": "vllm"`
- **字幕**：“客户端只跟 9998 说话，看到的就是后端 vLLM 的模型。”
- **截图点**：`id: qwen2.5-7b`。

### 镜头 3 — 非流式对话（30s）
命令：`curl .../v1/chat/completions`（见 demo_show.sh）
预期画面：
```
回答: 我叫Qwen，是来自阿里云的大规模语言模型，很高兴为你服务。
用量: {'prompt_tokens': 33, 'total_tokens': 52, ...}
```
- **字幕**：“一条标准 chat 请求，经 xllm-service 转发、由 vLLM 生成、原样带回（含 usage）。”

### 镜头 4 — 流式对话 SSE（30s）★ 视觉最佳
命令：`curl -N .../v1/chat/completions {stream:true}`
预期画面：逐条 `data: {...delta...}` 实时刷出，最后
```
data: [DONE]
```
- **字幕**：“OpenAI 风格 SSE 逐 token 透传，最后 `[DONE]` 收尾——流式完全打通。”
- **截图点**：滚动的 SSE chunk + 结尾 `data: [DONE]`（动图/录屏效果最好）。

---

## 高光镜头 — 透明性证明（技术下钻，1min）

命令：`bash demo/demo_compare.sh`
预期画面（**已实测逐字一致**）：
```
请求体(两边完全相同): {... "temperature":0, "seed":42}
  直连 vLLM      (:18000) : 杭州是一座历史悠久、文化底蕴深厚且现代繁华并存的江南水城。
  经 xllm-service(:9998) : 杭州是一座历史悠久、文化底蕴深厚且现代繁华并存的江南水城。
  ✓ 输出逐字一致 —— xllm-service 转发透明，未破坏语义
vLLM 访问日志: 127.0.0.1:xxxxx - "POST /v1/chat/completions HTTP/1.1" 200 OK
```
- **字幕**：“同样的输入（固定 seed/温度），直连 vLLM 和经 xllm-service 输出**逐字一致**——
  证明编排层是无损透明转发；下面 vLLM 日志也确认请求确实由 service 转发进来。”
- **截图点**：两行一致的输出 + 绿色 ✓。这是整个 demo 最有说服力的一张图。

---

## 收尾

```bash
bash demo/demo_down.sh    # 停 master + vLLM（释放卡3），清理 etcd 注册，保留 etcd 进程
```

---

## 备问（评审时可能被下钻）

- **转发怎么实现的？** `http_service/service.cpp` 的 `handle_vllm`：拿调度选中实例的 HTTP channel，
  把客户端**原始 JSON** 直接 HTTP POST 给 vLLM；流式复用 brpc 渐进读 `CustomProgressiveReader`，
  非流式复用 `handle_non_stream_response`。xLLM 后端走原 brpc 路径，按 `backend_type` 分流，老路径零回归。
- **为什么 vLLM 不用 tokenize/调度计算？** vLLM 自己做 tokenize 和 chat template，`schedule()` 对 vllm 集群
  跳过这些，只做实例选择（`--default_backend_type=vllm`）。
- **为什么实例 type=DEFAULT 不是 MIX？** 单实例无 decode 时调度器只放行 DEFAULT；这是本次实测确认的约束。
- **现在的注册是手动写 etcd，正式怎么做？** 下一步上 vLLM **sidecar**：自动注册 + 心跳 + 上报 metrics，
  替代手动 `register_vllm.sh`。本 demo 用手动注册聚焦“转发链路已通”这一成果。
- **本轮范围**：单 vLLM 实例端到端（M1+M2）。未做：KV 事件桥/CacheAwareRouting、Disaggregated PD、
  多 backend 混合集群（P1）。
