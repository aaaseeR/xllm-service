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

# V2-B10 首版代码交付状态

更新时间：2026-08-09
状态：`CPU_VERIFIED / NPU_AND_CLUSTER_PENDING`

## 结论

首个版本按设计直接交付 V2，没有独立 V1。B0-B10 的仓库内生产代码、公共协议、
CPU 可达链路、simulated HBM、支持矩阵和运维文档已闭环；代码可以进入 NPU 与真实
集群验证。`CPU_VERIFIED` 只证明控制面、协议、状态机、host 资源和模拟 HBM 不变量，
不证明 CANN、真实 HBM/DMA/Link、设备吞吐或生产 SLO。

## 支持矩阵

| Provider | 执行模式 | V2 代码状态 | 尚缺外部证据 |
| --- | --- | --- | --- |
| xLLM Native | `REMOTE_PD/LAYERWISE_PUSH/P_FIRST` | CPU_VERIFIED；P/D 原子 admission、commit、deadline、attempt、fencing、首事件恢复、输出定序和清理闭环 | NPU P/D、真实 Link、CANN/HBM、故障注入和 soak |
| xLLM Native | `LOCAL_PREFILL_DECODE/NONE/D_ONLY` | CPU_VERIFIED；稳定 bucket、token cap、能力门禁和 mode-specific commit | NPU 本地模式吞吐/显存/尾延迟 |
| xLLM Native | `PREFILL_ONLY/NONE/P_ONLY` | CPU_VERIFIED；输出上限、单 P 计划与无 execution holder 语义闭环 | NPU 输出一致性和资源回收 |
| vLLM-Ascend | `AGGREGATED/NONE/SINGLE` | CPU_VERIFIED；严格 Agent、唯一 ingress、attempt/deadline/fencing、cancel/query loopback | vLLM Ascend NPU、同命部署、原始端口隔离和 soak |
| 任意跨 Provider P/D | 任意 split | 稳定拒绝；selector、compatibility 和负向测试覆盖 | V2 明确不支持，不是待开开关 |
| HOST/SSD/Store 数据路径 | 分层 KV | 未交付；仅发布 `_ub` shadow credit | 属于 V2.5，不能按 V2 能力宣传 |

## B10 完成范围

- ModelPool 隔离：Provider route、RR/SLO/CAR、最终 incarnation binding 和有界队列
  都按精确 model revision 过滤。Descriptor-less contract-v0 仅保留显式
  BEST_EFFORT 兼容，不获得 V2 多模型声明。
- tenant/KV 隔离：租户/flow/model 键有 256 字节上限并拒绝 NUL/CR/LF。默认忽略不可信
  客户端 tenant/flow 头，以 request UID 派生 KV namespace 并关闭跨请求 credit；
  只有鉴权 Gateway 部署显式开启可信头开关后，才允许同 tenant 复用。
- K2 组合：公平队列先决定下一个请求，KV/load planner 只在出队后、公共硬过滤后的
  候选中排序，不反向改变 priority/tenant 账本。HOST/SSD/Store 只查询有界 shortlist，
  只形成命中 token 上界，不进入 V2 cost、route 或 admission。
- 执行模式：Native 三种模式走统一 capability Resolver 和 mode-specific commit；
  vLLM 只开放严格 AGGREGATED。跨 Provider、缺 capability、profile/model/KV/Link
  不兼容均 fail closed。
- 可观测：常开低基数 bvar 指标与周期集群快照；`--v=1` 开启有界、无请求正文的
  逐请求 JSON 事件；xLLM Engine 使用同风格 attempt/stage VLOG。详见
  [观测与性能分析手册](./OBSERVABILITY_RUNBOOK.md)。

## 硬件边界

多硬件差异由底层 Engine/Provider 暴露。Service 只感知不可变 Provider/profile、
execution capability、KV/Connector compatibility、topology、统一 admission 和
Engine/Link 状态；不调用 CANN/CUDA，不保存 device pointer、真实 HBM 地址、stream、
event、allocator 或 kernel。simulated HBM 是 Engine/backend 契约的 CPU 测试实现，
不是 Service 生产依赖，也不能代替 NPU 验证。新增 NPU/CUDA/MLU/DCU backend 必须复用
同一 Contract/Resolver，并按 profile 提交 conformance 证据，禁止在 Service 增加
芯片特例分支。

## 需求与验证追踪

| 能力 | CPU/Torch CPU | simulated HBM | NPU/集群 |
| --- | --- | --- | --- |
| 多模型、Provider/profile、tenant 隔离 | route/model 负向、ModelPool blocking、namespace golden、账本键边界 PASS | namespace 隔离、owner/incarnation PASS | 待真实多租户 Gateway 与多模型池 |
| 公平流控和过载 | 容量、deadline、priority、tenant RR、EDF、starvation、并发账本 PASS | N/A | 待生产饱和曲线和长时公平性 |
| K0-K2 HBM KV-aware | event/snapshot/planner/gate/actual 对账 PASS | 容量、地址、checksum、OOM、碎片、并发、故障回收 PASS | 待真实 HBM、Link 与校准门 |
| 双 Provider 与执行模式 | Native/vLLM conformance、loopback、稳定拒绝 PASS | Native transfer/hold PASS | 待两种真实 runtime |
| 日志与性能分析 | schema、presence、无正文 JSON、ring/drop、clock/metric validity PASS | KV predicted/actual 对账 PASS | 待采样开销、告警阈值与集群聚合 |

最终验证命令与精确计数记录在本提交的验证日志和
[B6-B10 总状态](./B6_B10_STATUS.md)。全量门包括外部 xLLM override、Service pinned
gitlink、三个生产二进制动态链接、vLLM sidecar pytest、压力重复和 sanitizer 切片。
当前基线为 xLLM `446bae12` 的八目标 118/118（simulated HBM 12/12、Provider 9/9、
RequestEvent 14/14），Service pinned/override 380/380，vLLM sidecar 60/60。
xLLM attempt/simulated-HBM/RequestEvent 三个目标各重复 100 轮；Service 的 recorder、
hash/namespace、KV planner/metrics、Provider route 和 flow-control 57 项各重复 100 轮，
共 5700 次，无失败。同一组 57 项还在 GCC 13 `-fsanitize=address,undefined` 下通过，
`detect_leaks=1` 且 ASan/UBSan 均为首错退出，未发现内存、泄漏或未定义行为报告。
沙箱镜像未携带 ARM64 Clang ASan compiler-rt，因此 sanitizer 证据使用 GCC 13；这不
影响生产 Clang C++20 `-Werror` build/link 门，后者仍独立通过。

## 三轮深度 review 记录

1. 资源/并发轮：修复低层 tier 对全候选查询、响应 chunk 冒充 token、正常终态重复计数、
   recorder 容量上限与默认 VLOG 开销；补齐不可信 tenant/KV namespace 隔离和有界键。
2. 协议/兼容轮：确认所有 proto 仅 additive 并固定字段号；修复多模型 hard filter、最终
   incarnation 重新校验和 contract-v0 BEST_EFFORT 边界，K2 shadow 不进入 V2 planner。
3. 测试/文档/隐私轮：刷新陈旧测试计数与 PARTIAL 状态；移除 INFO 级完整 JSON 输出，
   legacy 内容 trace 启用时显式告警；补齐 5700 次压力门、生产二进制和运行手册。

三轮 review 后没有遗留的仓库内 correctness blocker；剩余项均是本文件明确列出的
NPU、真实集群、平台和生产标定证据。

## 回滚、发布与剩余门

- 关闭 `kv_route_enforced_gate_open` 或把 bucket 设为 0，立即回到 SHADOW；切换
  load-balance policy 可回到 load-only。低层 tier 无需回滚，因为从不参与 V2 决策。
- 关闭 `--trusted_tenant_headers_enabled` 会忽略租户头、关闭跨请求 KV credit；发布
  系统必须同时确保 Gateway 去除客户端自报头，不能只依赖文档约定。
- `--v=1` 只用于需要逐请求诊断的实例/窗口；关闭后不构造逐请求 protobuf，常开
  低基数指标和集群快照仍保留。
- 协议变更均为 additive；Service gitlink 必须指向已推送的同版 xLLM commit。
- 达到 `VERIFIED` 仍必须完成 NPU/CANN、真实 HBM/Link/etcd、多 Service/P/D、Agent
  同命/端口隔离、故障矩阵、容量拐点、长时 soak、观测开销和 SLO 阈值验收。
