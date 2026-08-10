# xLLM Service V2 现状、业界对标与演进蓝图汇报

更新时间：2026-08-10
汇报口径：首个交付版本直接为 V2；当前 `CPU_VERIFIED / NPU_AND_CLUSTER_PENDING`

## 1. 执行摘要

历史线上数据表明，问题不是一个固定的 Decode 内核瓶颈，而是流量跨过容量拐点后，
瓶颈在路由、D Admission、Prefill、KV 和 Decode 之间迁移：热点 P/D 的 TTFT/TPOT
同步恶化，部分请求在 Gateway 超时后仍占用设备，旧日志又无法用统一 request/attempt
身份解释全过程。

V2 的解法不是增加一个更复杂的负载均衡公式，而是建立完整控制闭环：

- Service 做有界公平准入、Provider/执行模式选择、P/D 软选点和 attempt 协调；
- Engine/Agent 做最终原子准入、真实 HBM/KV、deadline、fencing 与资源释放；
- Registry 只保存低频身份/能力，高频 Engine/Link/KV 状态进入可丢失软视图；
- 全链路以同一 request/attempt/incarnation 身份解释错误、容量和性能；
- CPU 与 simulated HBM 先证明链路和不变量，最终由 NPU/真实集群证明性能与硬件行为。

当前 B0–B10 已形成完整仓库实现并通过 CPU 门，具备进入 NPU/集群验证的条件。总体
组件拓扑见[总体架构图](./01_XLLM_SERVICE_ARCHITECTURE_DESIGN.md#4-总体架构组件拓扑与平面边界)，
当前功能与远端代码索引见[V2 当前能力](./12_XLLM_SERVICE_V2_CURRENT_CAPABILITIES.md)。

## 2. 数据分析：为什么必须建设推理控制闭环

以下是 V2 开发前的历史线上证据，用于解释设计优先级，不是 V2 上线后的收益证明。
完整口径、窗口与限制见
[GLM-5.2 线上瓶颈分析](./10_XLLM_SERVICE_GLM52_ONLINE_BOTTLENECK_ANALYSIS.md)。

| 事实 | 数据 | 对 V2 的约束 |
| --- | ---: | --- |
| Gateway 原始样本 | 2,273,519 请求，连续 4.736 天 | 不能用单实例或短压测替代线上分布 |
| Engine 合并去重 | 858,274 个成功请求；744,644 个完整 P/D | 必须统一 P/D request/attempt 身份 |
| 非网关流控事件窗 | 成功率 72.4%，503 占 22.2% | 过载要在 Service/Engine 准入闭环内稳定拒绝 |
| P→D 条件性长尾 | 事件窗 p95 18.96s；稳态样本 p95 3.7ms | D 不能过早锁死；准入需要实时 credit 与有限重选 |
| 相同 TP2/DP16 高低流量对照 | 高流量 TTFT p95 8.83–11.22s，低流量 0.832–0.844s | 必须识别非线性容量拐点，不只看请求数 |
| Decode 衰减 | 高流量 TPOT p50 40.2–44.6ms，低流量 20.5–21.7ms | 选点成本必须包含 Decode headroom/SLO residual |
| D KV 可用池 | 21.10–25.28GB；1,709 个硬 Prompt 容量失败 | Service 只做预判，Engine allocator 必须原子准入 |
| 超时后无效执行 | 8,626 请求超过 300s，最长 4,891s | deadline 必须下沉到 Engine 并驱动 cancel/release |
| 计时质量 | 8.34% 多 token TPOT=0；3,361 条负 ITL | 指标必须定义 clock domain 和 validity，不能盲调权重 |
| Prefix Cache | 11/12 个 P 开启，但 750,740 成功请求命中全为 0 | KV-aware 上线前必须打通位置、实际命中与收益对账 |

核心判断：容量、可用性和成本问题来自同一个缺失闭环。只优化算子或提高平均设备利用率，
不能解决 503、Admission 长尾、热点倾斜和 deadline 后无效计算。

## 3. 业界现状与我们的选择

主流开源方案正在收敛到“请求决策、执行资源、异步状态/KV 分离”的架构，但各自侧重不同：

| 方案 | 官方架构重点 | 可借鉴点 | xLLM Service 的取舍 |
| --- | --- | --- | --- |
| [llm-d](https://llm-d.ai/docs/dev/architecture) | Router、InferencePool、Model Server；EPP 实现 filter/score/pick 与 flow control | 独立路由决策、KV/负载感知、标准化池接口 | 保留 Provider Contract，并把跨 P/D attempt/commit/fencing 纳入 Service 正确性职责 |
| [NVIDIA Dynamo](https://docs.nvidia.com/dynamo/dev/knowledge-base/overview) | Request、Control、Storage & Events 平面；Frontend/Router/Worker/KV/Planner 组合 | 平面拆分、事件驱动、分布式 KV 与规划器边界 | V2 不让共享 Store 成为正确性依赖；先完成 Engine 原子准入和本地 HBM 闭环 |
| [SGLang Model Gateway](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/sgl_model_gateway.md) | Gateway 统一 worker 生命周期、路由、重试、熔断和观测，并支持 PD | 高性能接入、worker 管理、故障与路由一体化 | Gateway 保持鉴权/API 边界；Service 专注推理级公平、Provider/mode 与 P/D 协调 |
| [AIBrix](https://github.com/vllm-project/aibrix) | Gateway、routing、autoscaling、runtime、分布式 KV 与异构基础设施 | 云原生控制面、自动扩缩、异构资源与 KV 扩展 | V2 先交付请求控制；Placement/Autoscale 在 V3 慢环，避免进入请求关键路径 |

共同趋势不是“统一所有 Runtime 内部实现”，而是用稳定契约描述能力、状态和错误，让
Router 只选择已证实可执行的组合。我们的差异化重点是：动态 xLLM Native P/D 的逐层
PUSH、D 原子 admission、结果不明 hold、GenerationCommit、旧 incarnation fencing，
以及 xLLM Native 与 vLLM-Ascend 在同一 Provider 框架下按能力非对称接入。

## 4. 我们当前已经完成什么

| 层 | 当前 V2 能力 | 远端实现入口 |
| --- | --- | --- |
| 接入与信任 | OpenAI/Anthropic、CanonicalRequest、UUIDv7、可信 Gateway identity、HMAC KV session | [HTTP Service](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/http_service)、[Trust Policy](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/http_service/request_trust_policy.cpp) |
| 公平与过载 | 多维硬预算、priority、tenant→flow 轮转、FCFS/EDF、deadline、drain | [Flow Control](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/scheduler/flow_control_queue.cpp) |
| Provider 与模式 | Descriptor/capability/profile；Native 三模式；vLLM AGGREGATED；跨 Provider split 拒绝 | [Provider](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider)、[xLLM contract](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm/tree/service_dev/xllm/core/runtime/native_provider_contract.cpp) |
| 状态与路由 | Registry/State/KV lane、freshness/readiness/link、load/KV/SLO planner、SHADOW/ENFORCED gate | [Registry/Planner](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider)、[Scheduler](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/scheduler/scheduler.cpp) |
| 执行安全 | attempt、D admission、hold、Query/Cancel、commit、deadline、incarnation fence、release | [Service hold/control](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider)、[Engine attempt](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm/tree/service_dev/xllm/core/distributed_runtime/attempt_lifecycle_table.cpp) |
| KV 资源 | canonical block hash、HBM/HOST 事件、simulated HBM、真实 BlockManager RAII 适配 | [Service KV view](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider/kv_shadow_index.cpp)、[Engine KV resource](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm/tree/service_dev/xllm/core/framework/kv_cache) |
| 输出与容错 | 跨 sender 定序、重复/迟到 fencing、gap recovery、首输出前有限重试、SSE 终态 | [Request state machines](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/request) |
| Debug/性能 | 统一事件 schema、常开 bvar、VLOG JSON、cluster snapshot、drop/cleanup/KV pressure | [Observability](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/observability)、[shared proto](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm/tree/service_dev/xllm/proto/observability.proto) |
| 工程门禁 | 双仓 pin、Service CTest、三个 serving binary、vLLM pytest、xLLM CPU contract | [Coding CI](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/.coding-ci.yml) |

完成的含义是仓库实现和 CPU 证据完成，不代表真实 NPU 与集群门已经完成。

## 5. P/D 选择策略：从硬正确到 SLO goodput

### 5.1 当前 V2 决策链

```text
Request normalize / trust / deadline
  -> 有界 admission 与 tenant→flow 公平出队
  -> Provider/model/profile/mode/incarnation/lease/link/state 硬过滤
  -> 有限候选计划枚举
  -> Prefill queue + token cost + Decode headroom + HBM Prefix - transfer cost
  -> SLO feasibility / uncertainty / stable tie-break
  -> D Engine 原子 admission
  -> attempt holder 收敛与 GenerationCommit
  -> 首输出提交；之后永久关闭自动重试
```

三个原则：

1. **公平优先于收益。** KV 命中和低负载只影响已经公平出队请求的候选排序，不能让高
   cache-hit 租户越过 priority/tenant 账本。
2. **硬事实优先于预测。** lease、incarnation、capability、LinkState、KV layout、永久
   不可行和 D allocator 拒绝不可被软分数覆盖。
3. **延迟绑定与有限重选。** Service 尽量晚绑定 D；在首输出前、剩余 deadline 和 attempt
   预算内可换 D，输出提交后不进行可能产生双输出的自动重试。

实现入口：
[flow queue](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/scheduler/flow_control_queue.cpp)、
[route selector](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider/provider_route_selector.cpp)、
[KV planner](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider/kv_route_planner.cpp)、
[execution plan](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider/execution_plan_builder.cpp)。

### 5.2 从 SHADOW 到 ENFORCED

- SHADOW 计算计划和预测，不改变现有实际路由，用于校准 candidate coverage、预测/实际
  Prefix、route churn、fallback 和 SLO residual。
- ENFORCED 启动必须同时满足 CAR policy、完整 cost 参数、非零 `bytes_per_token`、有限
  bucket 和不可变 build-id；缺任何门禁拒绝启动。
- 灰度按稳定 request bucket 开放，出现 TTFT/TPOT、error、fallback、cleanup 或公平性
  回退时立即把 bucket 降为 0；不通过临时修改硬过滤“保流量”。
- NPU 阶段用固定 model/profile 阶梯 QPS 重新标定 cost，不把 CPU 参数带入生产。

## 6. 容错与错误体系

### 6.1 分层错误模型

| 层 | 典型错误 | 决策与恢复 |
| --- | --- | --- |
| Client/Gateway | 非法请求、身份断言不可信、deadline 不合法 | 永久拒绝；稳定公开 code + request_uid |
| Service flow | queue/token/bytes/tenant/model 预算耗尽 | 可排队则有界排队，否则稳定拒绝；账本精确回滚 |
| Route/observation | 无兼容 Provider、state stale、Registry blind、link not ready | 不降级到未知能力；退出新准入/READY 或有限 grace |
| Engine admission | permanent infeasible、temporary no credit、conflict | permanent 终止；temporary 仅在候选/deadline/attempt 预算内换点 |
| RPC/attempt | submit/cancel 结果不明 | 持有资源 hold；Query/Cancel/fence/TTL 证明终态前不创建替代执行 |
| Output | 旧 incarnation、重复、乱序、gap、sender 错误 | fencing、有限重排/查询；无法闭合则单一终态 |
| Resource | release deferred、cleanup timeout、DMA/lease uncertain | 不假定“RPC 失败=资源释放”；保留 quarantine 并告警 |
| Shutdown | 新请求与在飞请求竞争 | 先 NOT_READY、停止 admission、有界 drain、显式 unresolved 终态 |

### 6.2 稳定错误纪律

- 代码枚举参与控制决策，free-form message 只用于有界诊断；
- client message 不含原始 header、凭据、prompt、内部堆栈或无限长度文本；
- 每个拒绝至少带稳定类别与 `request_uid`，运维事件再带 stage/reason/attempt/incarnation；
- 永久、临时、结果不明三类不可混淆；“结果不明”不能当作“没有执行”；
- Provider、KV replica、flow-control、physical resource 保持各自强类型，在 adapter/ingress
  边界统一翻译，不构造万能 Status 类型。

## 7. Debug 与性能分析策略

### 7.1 全盘跟踪

统一关联链：

```text
global_request_id/trace_id
 -> request_uid
 -> attempt_seq
 -> provider/model/profile/mode/domain
 -> engine_uid/incarnation/dp-rank/link/prefix
```

事件按阶段覆盖 REQUEST、FLOW_CONTROL、ROUTE、D_ADMISSION、PREFILL、TRANSFER、
FIRST_OUTPUT、DECODE、RESOURCE_RELEASE 和 REQUEST_END；每阶段都有 STARTED/终态和稳定
reason。Gateway、Service、Agent、P/D 使用同一身份，不再靠 IP、主机名或时间猜关联。

### 7.2 常态观测与问题窗口

- **常态：** 低基数 bvar + 周期 cluster snapshot，持续看请求率、goodput、queue、
  admission、route/fallback、KV pressure、state freshness、cleanup、event drop；
- **问题窗口：** 对定向副本开启 `--v=1`，导出无 prompt/输出正文的逐请求 JSON；
- **性能分解：** `E2E = ingress + queue + route + P admission/prefill + transfer +
  D admission/decode + output`，每段同时报告有效样本数、clock domain 和 p50/p95/p99；
- **集群诊断顺序：** 先确认数据完整性和 stale/drop，再看公平/queue，再看 route 分布，
  再看 D admission/KV，最后才归因到 Prefill/Decode kernel；
- **优化闭环：** 预测值与 actual outcome 对账，按 model/provider/profile/mode 隔离 residual，
  避免一种硬件或模式的参数污染另一种。

详细字段、查询和故障手册见
[Observability Runbook](./implementation/OBSERVABILITY_RUNBOOK.md)。

## 8. 未来蓝图与版本规划

| 阶段 | 目标 | 关键验收 | 不做/边界 |
| --- | --- | --- | --- |
| V2 当前收口 | 完成仓库实现、CPU/Torch CPU、simulated HBM、双 Provider 控制链 | 391/391 Service、三个 serving binary、双仓 contract 门；Coding CI 入库 | 不声称 NPU/生产 VERIFIED |
| V2 NPU/集群发布门 | 在真实 xLLM/vLLM-Ascend、CANN/HBM/Link、etcd、多 Service/P/D 上证明正确性与 SLO | 故障矩阵、阶梯 QPS、capacity knee、TTFT/TPOT/goodput、24h+ soak、观测开销 | 这是 V2 发布验证，不另起 V1 |
| V2.5 KV Memory Layer | DRAM/SSD/Mooncake Store，D 写穿、D→P/Store restore、跨请求 Prefix | 对象 commit/GC、namespace/TTL、带宽成本、Store HA/OpLog、收益门 | 不保存 Decode checkpoint；不成为 V2 基础正确性依赖 |
| V3 Placement/Autoscale | 模型、P/D role、profile、副本 desired state 慢环 | load/warmup/drain、容量预测、cache-loss 成本、回滚 | 不在请求路径动态加载模型；可与 V2.5 并行 |
| V4 Domain Overflow | 跨 domain 整请求溢出 | domain health、配额、成本、数据合规和 SLO | 不迁移在飞 Decode |
| V5 有限跨域 P/D | 网络/拓扑/异构感知 P/D 与分层 KV 联合决策 | 兼容矩阵、传输收益、failure domain、fencing 与合规 | 未验证组合一律关闭；不追求任意硬件笛卡尔积 |

多硬件支持的长期边界保持不变：Service 感知 Provider、profile、capability、统一资源摘要
和稳定错误；CANN/CUDA、device pointer、stream/event、allocator、真实 HBM 与 Connector
由底层 Engine/Provider 实现。新增硬件先通过 Provider conformance 和真实设备证据，不能
在 Service 中增加芯片特例分支。

## 9. 下一阶段建议

1. 在 Coding 平台启用本仓 `.coding-ci.yml`，把完整 V2 CPU gate 设为 `service_dev/main`
   受保护分支 required check；以首次远端构建验证镜像、网络、submodule 和缓存。
2. 建立最小 NPU 单域环境：1P1D + 1 aggregated vLLM-Ascend + etcd + 2 Service，先跑
   correctness/fault matrix，再跑阶梯负载。
3. 使用固定 model/profile 分别校准 REMOTE_PD、LOCAL、PREFILL_ONLY 和 AGGREGATED，
   所有结论按 Provider/mode 隔离，不混合分位数。
4. 以历史数据的主要失效模式为回放集：D credit 耗尽、永久 KV 不可行、P/D 热点、
   state stale、Agent SIGKILL、Service drain、deadline 后继续执行、output gap。
5. 只有在 actual Prefix、route、admission、cleanup 和 SLO residual 完成对账后，才逐级
   打开 KV ENFORCED bucket；任何阶段不以“提高设备利用率”替代 goodput/SLO 门。
