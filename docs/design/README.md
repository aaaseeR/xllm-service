# xLLM Service 设计文档导航

本目录是 xLLM Service 设计的唯一入口。xLLM Service 本身就是推理控制面，不是“上层 LLM Service”之外的另一个控制面，也不是无状态 HTTP Proxy。

**当前现状：** 已有多副本接入、全量 Engine Registry 和 xLLM 逐请求动态 P/D，也有 vLLM HTTP 中继；但两者仍由全局 `backend_type` 分支拼接，vLLM-Ascend 还不是可参加统一调度的 Provider。选点、准入、deadline、故障隔离、资源回收和阶段观测也没有形成生产闭环。

**V1 目标：** 先在单域、单模型下可靠地把请求跑完，并建立统一 Engine Provider SPI。xLLM Native 交付严格动态 P/D；vLLM-Ascend 先以受 Provider Agent 约束的聚合模式进入同一 Registry、State Stream、调度和故障框架，能力不足的执行模式 fail closed。Engine 负责原子准入、本地超时停止和资源回收，统一事件说明请求为什么被选择、时间花在哪里以及失败后何时释放资源。

**最终形态：** 同一控制面通过 Provider Adapter 管理多 Runtime、多硬件、多超节点、多模型、P/D 角色和分层 KV；为每个请求选择已验证的 Provider 与执行模式，再决定在哪里 Prefill/Decode、复用或加载哪份 KV、何时排队或拒绝。目标是在 SLO、公平和成本约束下最大化有效请求量，而不是只追求设备利用率。

三种形态的请求路径、故障语义和明确边界见 [总体架构 §2](./01_XLLM_SERVICE_ARCHITECTURE_DESIGN.md)。最终目标沿三条正交轴共同推进：

- **请求调度轴：** V1 硬准入与快速拒绝 → V2 策略感知的有界流控 → V2 优先级与租户公平 → 持续演进 SLO goodput 与成本联合优化。
- **执行拓扑轴：** V1 单域单模型、多 Provider（xLLM 动态 P/D + vLLM-Ascend 聚合）→ V2 多模型与逐请求执行模式 → V3 模型与角色自动放置 → V4 跨域整请求溢出 → V5 收益可证明的有限跨域 P/D。
- **KV 内存轴：** V1 Engine 本地 HBM → V2 精确 Prefix 位置索引 → V2.5 DRAM、SSD 与共享 Store → V2.5 跨请求恢复、复用与放置。

箭头只表示同一轴内部的能力成熟顺序，不表示轴间全序依赖。三条轴共用一个不变边界：**Service 做软选择和协调，Engine/Store 持有资源与数据真相。**

## 阶段摘要

| 阶段 | 目标结果 | 明确不做 |
| --- | --- | --- |
| V1 | 单域单模型的生产控制面：公共 Provider SPI；xLLM Native 严格动态 P/D；vLLM-Ascend 严格聚合模式；全链路事件、健康、准入/TTL、本地 deadline 和首 token 前重试 | 未验证的 Provider 能力、跨 Provider P/D、持久请求状态、Service 侧策略队列、Decode 迁移 |
| V1.x | M1/M2 预测与在线校准，提高选择质量 | 改变 V1 资源协议 |
| V2 | 多模型、精确 HBM KV-aware、策略感知有界队列、优先级和租户公平；能力允许时逐请求选择本地或远程 Prefill | 共享 Store 成为正确性依赖 |
| V2.5 | 集群 KV 内存层：D 生成 KV 写穿共享层、D→P/Store 恢复、跨请求和 agentic Prefix 复用 | Decode checkpoint、以 session manifest 代替完整历史 |
| V3 | 模型、P/D 角色、profile 和副本的 Placement/Autoscale 慢环 | 请求关键路径加载模型或切换角色 |
| V4 | 跨 domain 整请求溢出 | 跨域在飞请求迁移 |
| V5 | 网络/拓扑/异构感知的有限跨域 P/D 与分层 KV 联合决策 | 未验证兼容组合和收益为负的远程执行 |

阶段编号表示能力成熟度和可独立验收结果，不表示三条轴只能串行开发。V2.5 与 V3 可以并行开发、独立上线、互不阻塞：V3 在没有共享 KV 层时显式计入缩容 cache loss，V2.5 可在对象提交后降低该损失，但不是 Placement/Autoscale 的硬前置。V2.5 是主路线交付，不再把跨请求/跨轮 KV 作为编号外的可选优化。

## 状态与责任边界

| 信息或资源 | xLLM Service 是否保存 | 权威位置 |
| --- | --- | --- |
| 当前请求、调度计划、重试和输出排序 | 仅接入副本内存，完成后释放 | 当前 Service `RequestContext` |
| Engine Provider、版本、模型、profile、能力、lifecycle、lease | 本地缓存 | etcd Registry；Provider Descriptor 对 incarnation 不可变 |
| queue、KV headroom、credit、吞吐和延迟 | 本地软缓存 | Engine，经 State Stream 发布 |
| Prefix block 到 Engine/tier 的位置 | V2 起本地有界软索引 | Engine 真实 cache/allocator |
| KV tensor、reservation、Decode 和 transfer handle | 不保存 | Engine HBM/DRAM/SSD |
| 共享层 KV 对象和可选 session manifest | 不作为 Service 本地状态 | Mooncake Store + 强一致元数据服务 |
| 对话文本、业务幂等、工具状态和用户记忆 | 不持久化 | Client/Gateway/上层应用 |
| 指标、trace 和审计事件 | 导出，不作为请求正确性状态 | 观测/计量系统 |
| 模型/角色/副本 desired state | V3 由独立慢环模块维护 | Placement Controller/部署系统 |

## 阅读顺序

| 顺序 | 文档 | 用途 |
| --- | --- | --- |
| 1 | [总体架构与阶段演进](./01_XLLM_SERVICE_ARCHITECTURE_DESIGN.md) | 最终目标、系统边界、状态归属、请求流程和 V1–V5 分工 |
| 2 | [V1 实现规格](./02_XLLM_SERVICE_V1_IMPLEMENTATION_SPEC.md) | V1 开发接口、状态、容错、开发顺序和上线门禁 |
| 3 | [多引擎 Provider 与 Adapter](./11_XLLM_SERVICE_MULTI_ENGINE_PROVIDER_DESIGN.md) | xLLM Native/vLLM-Ascend 的公共 SPI、能力门禁、执行计划、状态和兼容矩阵 |
| 4 | [Engine 性能建模](./03_XLLM_INFERENCE_ENGINE_MODELING_DESIGN.md) | M1/M2 使用的预测、校准、Scheduler 重放和配置搜索 |
| 5 | [Engine 能力基线](./04_XLLM_INFERENCE_ENGINE_CAPABILITY_GUIDE.md) | 指定 xLLM commit 的能力快照；不是当前架构协议 |
| 6 | [集群 KV 内存层与跨请求复用](./05_XLLM_PD_STORE_SESSION_DESIGN.md) | V2.5 的共享 KV、D 写穿、D→P/Store 恢复和 session 边界 |
| 7 | [设计决策记录](./06_XLLM_SERVICE_DECISION_LOG.md) | 已定取舍、代价和重新引入条件 |
| 8 | [设计评审日志](./07_XLLM_SERVICE_REVIEW_LOG.md) | 逐轮评审记录、问题清单和代码事实基线 |
| 9 | [集群级 KV-aware Router](./08_XLLM_SERVICE_CLUSTER_KV_AWARE_ROUTER_DESIGN.md) | V2 精确 KV 位置索引、P/D Prefix 与负载联合选择 |
| 10 | [V2 有界流控与执行模式](./09_XLLM_SERVICE_V2_FLOW_CONTROL_AND_EXECUTION_MODES_DESIGN.md) | V2 Service 队列、本地 Prefill、故障预算和 drain 语义 |
| 11 | [GLM-5.2 线上瓶颈分析](./10_XLLM_SERVICE_GLM52_ONLINE_BOTTLENECK_ANALYSIS.md) | 网关与 Engine 多日证据、P/D 阶段、KV 硬容量、重试/级联失败、负载拐点与优化约束 |
| 评估 | [外部架构评估](./opus5_xllm_review.md) | 对照业界现状与未来方向；结论需吸收到权威文档后才生效 |
| 背景 | [推理系统优化技术全景](./90_VLLM_INFERENCE_SYSTEM_OPTIMIZATION_GUIDE.md) | 业界方案和底层优化参考，不作为实现约束 |

## 文档效力

发生冲突时，以 01 的最终架构边界和 02 的当前阶段实现规格为准。03 是建模专项设计，04 是其标注 commit 的 xLLM 能力快照，05 是 V2.5 KV 内存层专项设计，08 是 V2 集群 KV 路由专项设计，09 是 V2 有界流控/执行模式专项设计，11 是多引擎 Provider 专项设计，06 记录取舍，07 记录评审进展，10 提供线上证据、问题优先级和后续优化约束，外部评估和 90 只提供证据与建议。10/11 的结论需同步为 01/02/03/08/09 的正式接口或门禁后，才构成对应阶段的实现要求；评估结论没有同步到权威设计前，不构成实现要求。

V1 的首个生产版本必须完成 02 中 G-2、G-1、G0–G4/M0，包括 Provider SPI、xLLM 协议底座、内置 State Stream、统一 `IsSchedulable`、Engine 原子准入和能力门禁；它不是现有单对路径或 vLLM relay 的重新包装。M1/M2 在相同公共状态和 Provider Contract 上继续迭代，不阻塞 M0 上线。

06 和 07 是常设文档：06 记录已定决策及其重新引入条件，07 逐轮追加评审结论并跟踪问题状态。修改 01/02 关闭问题后，必须同步更新 07 的状态与关闭依据。
