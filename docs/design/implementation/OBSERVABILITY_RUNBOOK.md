<!-- Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================-->

# xLLM Service V2 Debug、日志与性能分析手册

## 1. 使用原则

默认运行保留低基数 bvar、周期集群快照、WARNING/ERROR 和 xLLM 既有日志。需要定位
单请求或拆解阶段耗时时，在限定实例和时间窗口开启 `--v=1`；完成后关闭。逐请求事件
会增加 protobuf、ring 和日志量，不应在未做容量评估时全集群长期打开。

推荐启动参数：

```text
--v=1
--observability_event_capacity=65536
--observability_export_batch_size=1024
--observability_export_interval_ms=20
--observability_snapshot_interval_ms=5000
--observability_build_id=<immutable-artifact-id>
```

`build_id` 只能包含字母、数字及 `-_.:/@+`。ring 最大 1,048,576；batch 必须不大于
ring。配置非法时 Service 拒绝启动。若租户头由鉴权 Gateway 注入并且下游无法被客户
端绕过，才可开启 `--trusted_tenant_headers_enabled=true`；否则保持 false。

## 2. 日志族与关联

| 前缀 | 级别 | 作用 |
| --- | --- | --- |
| `xllm_service_request_event` | VLOG(1) | Service 逐请求稳定 JSON 事件 |
| `xllm_attempt_event` | xLLM VLOG(1) | D admission、commit、transfer、cancel/query、terminal attempt 状态 |
| `xllm_stage_event` | xLLM VLOG(1) | P/D allocation、first generation、KV restore 等阶段耗时 |
| `xllm_service_cluster_snapshot` | INFO | 每 Service 的周期容量、吞吐、KV 和健康快照 |
| `xllm_service_observability_loss` | WARNING | ring capacity/contention/invalid 造成事件不完整 |

从 Gateway 向下先按 `global_request_id` 或 `trace_id` 找 Service 入口，再以不可变
`request_uid` 贯穿 Service/P/D。一次首输出前 retry 使用递增 `attempt_seq`；同一请求内
并发 producer 可能让日志物理顺序与逻辑顺序不同，必须按
`owner_incarnation_id + request_uid + attempt_seq + event_seq` 排序。Engine 重启后用
Engine UID + incarnation 区分；禁止只按可复用的进程地址关联。

`REQUEST_TERMINAL` 表示终态已抢占成功，但 E2E/TPOT metric sample 可能取得更大的
event_seq，因此不能假设 terminal 是物理最后一行。若 event loss 任一 counter 增长，
该窗口只能做部分诊断，不能把“未看到事件”解释为“阶段没有发生”。
强制 shutdown 超时后，仍在执行的 producer 可能晚于 exporter 最终 drain 产生事件；
这种窗口同样视为 trace 不完整，必须结合 shutdown/terminal counter 判断。

## 3. 单请求定位顺序

1. 查 `INGRESS` 与 `PREFILL_QUEUE`：没有入口通常是 Gateway/HTTP 边界；拒绝则看
   `error_stage/reason`，并核对 readiness、deadline 和 flow capacity。
2. 查 `ROUTE`：分别核对 P/D Engine UID/incarnation、Provider/profile、model revision、
   execution mode 和 queue wait。缺 route 且重复回队通常是池饱和或硬过滤后无候选。
3. 查 `P_DISPATCH` 与 xLLM `d_admission`：D 临时容量不足、永久不兼容、attempt terminal
   必须用 disposition/reason 区分，不能统一当作超时重试。
4. 查 transfer/commit 与 `p_first_generation/d_kv_restore/d_first_generation`：定位 P
   计算、传输、D restore、首事件 handoff 的耗时和失败归属。
5. 查 `FIRST_TOKEN_FLUSH`、TTFT/ITL/TPOT：只有累计 generated-token count 增长才形成
   token 边界；缺 usage 时指标如实缺失，不用 chunk 数补造。
6. 查 output sequence outcome 与 terminal：gap/buffered/duplicate/closed、cancel、deadline、
   provider error、terminal proof 和 cleanup 必须与 attempt/incarnation 一致。

日志不包含请求正文。如果必须核对业务内容，应在受控上游审计系统用
`global_request_id` 关联，禁止临时把 prompt/output/token IDs 加回普通 Service 日志。
仓库既有 `--enable_request_trace` 属于独立的 legacy 内容 trace，会把请求/响应正文写入
本地 `trace/trace.json`；它不是本机制的一部分，默认必须关闭，也不得用于常规生产
排障或性能分析。若受控调试确需启用，部署侧必须另行提供访问控制、保留期和安全清理。

## 4. 集群性能解读

以相同 `snapshot_window_ms` 对每个 Service snapshot 求和或求分位，并始终按
`build_id + service_incarnation_id` 分组，避免发布或重启导致 counter delta 混合。

| 现象 | 首查字段 | 解释与动作 |
| --- | --- | --- |
| queue 持续增长、terminal delta 不增 | readiness、saturation、Engine state/link、dispatch | 控制面失明、整池饱和或路由无候选；先恢复状态/容量，禁止扩大盲探测 |
| queue wait 高但 P/D stage 平稳 | queued requests/tokens/bytes、tenant/priority、dispatch delta | Service 流控或容量不足；核对模型池预算、公平性和 dispatch rate 下界 |
| TTFT 高、queue wait 低 | P allocation、first generation、KV restore/transfer | Engine Prefill、D admission 或 KV 传输瓶颈；按阶段而不是只看 E2E 优化 |
| TPOT 高、TTFT 正常 | D state/headroom、decode stage、output gap | Decode batch/算力/输出传输瓶颈；核对 mode 和 D 负载 |
| E2E 高且 TTFT/TPOT 均正常 | queue wait、terminal/cleanup、客户端 write | 排队、尾部终态证明或下游反压；核对 measurement boundary |
| predicted 高、actual 低 | overpredicted、decode overpredicted、admission conflict | KV shadow 陈旧、residence/survival 校准偏乐观；关闭 ENFORCED gate 回 SHADOW |
| predicted transfer 高、skipped 低 | namespace/model/profile、P/D actual hit | hash/isolation 不一致或 D 未驻留；按 request namespace/Engine stream 分开检查 |
| output gap/duplicate 增长 | attempt、incarnation、event sequence、first-event recovery | 重试竞态、发送方越权或网络乱序；不要绕过 fencing/sequence gate |
| event loss 增长 | ring events、capacity/contention、export interval | 降低 VLOG 范围、增大有界 ring/batch或缩短 interval；该窗口 trace 不完整 |

窗口吞吐近似：`sum(successful_terminals_delta) / window_seconds`；token throughput 为
`sum(delivered_tokens_delta) / window_seconds`。失败率分母使用成功与失败 terminal delta
之和。首个 snapshot 的 delta 固定为 0，不能纳入速率。跨模型/tenant/Engine 分析只从
逐请求 VLOG 离线聚合，不把高基数身份变成 bvar label。

## 5. KV 与多层 shadow

HBM predicted/actual 是 V2 K1 路由校准依据；HOST/SSD/Store 字段都以 `_ub` 结尾，
表示 bounded shortlist 内的可观测命中 token 上界。它们不表示数据可读、带宽可用、
对象已提交或 Store 数据路径已交付，也不进入 V2 route/admission。任何优化建议必须先
在 V2.5 实现真实 tier producer、数据搬运、成本与故障契约，再开独立门禁。

默认不信任 tenant header 时，每请求 namespace 由 request UID 派生，CAR 回到 load-only，
但同一次 P→D 的 hash domain 仍一致。可信头模式只允许同 tenant 复用；跨 tenant、模型、
Provider cache semantics 或未来 dynamic adapter identity 均生成不同 namespace。

## 6. 告警与发布门

- 立即告警：readiness 不接受请求、state hard-stale、Engine/Link 数突降、event loss、
  invalid event、duplicate terminal、admission conflict、output gap、失败 terminal 激增。
- 灰度 KV ENFORCED 前：在同 model revision + Provider profile + base/request namespace
  窗口核对 actual 缺失率、overprediction、fallback、conflict、skipped bytes；从小 bucket
  开始，异常立即关闭 gate。
- 发布对比：按 build ID 对齐相同 workload 的 throughput、queue wait、TTFT、TPOT、E2E、
  failure reason、mode mix、KV actual/predicted 和 event loss。没有相同 workload/窗口时，
  只记录观察，不下因果结论。
- CPU 只能验证公式、边界和事件完整性机制；NPU 上必须重新测量日志开销、真实阶段、
  HBM/Link 行为、阈值与长时稳定性，才能把状态升级为 `VERIFIED`。
