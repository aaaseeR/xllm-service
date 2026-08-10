# xLLM Service 设计文档导航

本目录是 xLLM Service 设计的唯一入口。xLLM Service 本身就是推理控制面，不是“上层 LLM Service”之外的另一个控制面，也不是无状态 HTTP Proxy。

**当前实现状态：** V2-B0 至 B10 的双仓代码与 CPU/simulated HBM 门已完成：xLLM Native 与 vLLM-Ascend 通过统一 Provider Contract、Registry、State/KV Stream、逐请求计划、有界流控和故障框架运行；多模型、HBM KV-aware、执行模式与端到端观测已有 CPU 证据。真实 NPU、CANN/Link、真实多 Service/P/D/etcd 集群、长时 soak 和线上阈值校准仍待验证，因此当前状态是 `CPU_VERIFIED / NPU_AND_CLUSTER_PENDING`，不是生产硬件 `VERIFIED`。

**首个交付版本：V2。** 不设置独立 V1 产品版本。原 V1 规格中的 Provider SPI、State Stream、原子准入、deadline、fencing、资源回收和观测闭环全部并入 V2 基础能力；首发还必须同时完成 V2 的多模型、精确 HBM KV-aware、有界流控、优先级/租户公平和逐请求执行模式。xLLM Native 与 vLLM-Ascend 进入同一 Registry、State Stream、调度和故障框架，能力不足的执行模式 fail closed。

**最终形态：** 同一控制面通过 Provider Adapter 管理多 Runtime、多硬件、多超节点、多模型、P/D 角色和分层 KV；为每个请求选择已验证的 Provider 与执行模式，再决定在哪里 Prefill/Decode、复用或加载哪份 KV、何时排队或拒绝。目标是在 SLO、公平和成本约束下最大化有效请求量，而不是只追求设备利用率。

总体组件拓扑、请求/状态/执行平面、多硬件边界及保留的简化请求流程见 [总体架构 §4](./01_XLLM_SERVICE_ARCHITECTURE_DESIGN.md)；三种形态的请求路径、故障语义和明确边界见同文 §2。最终目标沿三条正交轴共同推进：

- **请求调度轴：** V2-B0 硬准入与快速拒绝 → V2 策略感知的有界流控、优先级与租户公平 → 持续演进 SLO goodput 与成本联合优化。
- **执行拓扑轴：** V2-B0 单域单模型、多 Provider（xLLM 动态 P/D + vLLM-Ascend 聚合）→ V2 多模型与逐请求执行模式 → V3 模型与角色自动放置 → V4 跨域整请求溢出 → V5 收益可证明的有限跨域 P/D。
- **KV 内存轴：** V2-B0 Engine 本地 HBM → V2 精确 Prefix 位置索引 → V2.5 DRAM、SSD 与共享 Store → V2.5 跨请求恢复、复用与放置。

箭头只表示同一轴内部的能力成熟顺序，不表示轴间全序依赖。三条轴共用一个不变边界：**Service 做软选择和协调，Engine/Store 持有资源与数据真相。**

## 阶段摘要

| 阶段 | 目标结果 | 明确不做 |
| --- | --- | --- |
| V2-B0（内部开发门，不独立发布） | 公共 Provider SPI；xLLM Native 严格动态 P/D；vLLM-Ascend 严格聚合模式；全链路事件、健康、准入/TTL、本地 deadline 和首 token 前重试 | 不能作为 V1 或精简版 V2 对外交付 |
| V2（首个交付版本） | 在 V2-B0 全部能力之上完成多模型、精确 HBM KV-aware、策略感知有界队列、优先级和租户公平；能力允许时逐请求选择本地或远程 Prefill；M1/M2 可在公共协议上继续校准 | 共享 Store 成为正确性依赖、跨 Provider P/D、持久请求状态、Decode 迁移 |
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
| 0 | [V2 代码开发与交付规范](./00_XLLM_SERVICE_V2_DEVELOPMENT_STANDARD.md) | 首版直接 V2、xLLM 代码风格、CPU/Torch CPU 测试、开发文档和完成度门禁 |
| 1 | [总体架构与阶段演进](./01_XLLM_SERVICE_ARCHITECTURE_DESIGN.md) | 总体组件拓扑、请求/状态/执行平面、多硬件边界、状态归属、请求流程和 V2–V5 分工 |
| 2 | [V2 基础协议规格（原 V1 能力集）](./02_XLLM_SERVICE_V1_IMPLEMENTATION_SPEC.md) | V2 必须包含的基础接口、状态、容错、开发顺序和门禁；不独立发布 |
| 3 | [多引擎 Provider 与 Adapter](./11_XLLM_SERVICE_MULTI_ENGINE_PROVIDER_DESIGN.md) | xLLM Native/vLLM-Ascend 的公共 SPI、能力门禁、执行计划、状态和兼容矩阵 |
| 4 | [Engine 性能建模](./03_XLLM_INFERENCE_ENGINE_MODELING_DESIGN.md) | M1/M2 使用的预测、校准、Scheduler 重放和配置搜索 |
| 5 | [Engine 能力基线](./04_XLLM_INFERENCE_ENGINE_CAPABILITY_GUIDE.md) | 指定 xLLM commit 的能力快照；不是当前架构协议 |
| 6 | [集群 KV 内存层与跨请求复用](./05_XLLM_PD_STORE_SESSION_DESIGN.md) | V2.5 的共享 KV、D 写穿、D→P/Store 恢复和 session 边界 |
| 7 | [设计决策记录](./06_XLLM_SERVICE_DECISION_LOG.md) | 已定取舍、代价和重新引入条件 |
| 8 | [设计评审日志](./07_XLLM_SERVICE_REVIEW_LOG.md) | 逐轮评审记录、问题清单和代码事实基线 |
| 9 | [集群级 KV-aware Router](./08_XLLM_SERVICE_CLUSTER_KV_AWARE_ROUTER_DESIGN.md) | V2 精确 KV 位置索引、P/D Prefix 与负载联合选择 |
| 10 | [V2 有界流控与执行模式](./09_XLLM_SERVICE_V2_FLOW_CONTROL_AND_EXECUTION_MODES_DESIGN.md) | V2 Service 队列、本地 Prefill、故障预算和 drain 语义 |
| 11 | [GLM-5.2 线上瓶颈分析](./10_XLLM_SERVICE_GLM52_ONLINE_BOTTLENECK_ANALYSIS.md) | 网关与 Engine 多日证据、P/D 阶段、KV 硬容量、重试/级联失败、负载拐点与优化约束 |
| 12 | [V2 当前能力与远端代码索引](./12_XLLM_SERVICE_V2_CURRENT_CAPABILITIES.md) | 简洁但完整地说明已经具备的能力、远端 `service_dev` 代码/测试位置、证据边界和未完成门禁 |
| 13 | [V2 现状、业界对标与演进蓝图汇报](./13_XLLM_SERVICE_V2_STATUS_AND_ROADMAP_REPORT.md) | 数据现状、业界架构、当前实现、P/D 选择、容错错误、Debug/性能分析及 V2–V5 规划 |
| 14 | [V3 Placement 与 Autoscale](./14_XLLM_SERVICE_V3_PLACEMENT_AUTOSCALE_DESIGN.md) | V3 慢环、P/D/A 独立扩缩、leader fencing、desired state、生命周期 actuator、CPU 与线上门禁 |
| 状态 | [V2 功能开发状态](./implementation/README.md) | 每项功能的支持矩阵、需求到测试追踪、CPU/NPU 验证和剩余缺口 |
| 评估 | [外部架构评估](./opus5_xllm_review.md) | 对照业界现状与未来方向；结论需吸收到权威文档后才生效 |
| 背景 | [推理系统优化技术全景](./90_VLLM_INFERENCE_SYSTEM_OPTIMIZATION_GUIDE.md) | 业界方案和底层优化参考，不作为实现约束 |

## 文档效力

发生冲突时，版本、代码、测试和完成度门禁以 00 为准，系统边界以 01 为准，V2 基础协议以 02 为准，V2 路由、流控和执行模式分别以 08/09 为准，多 Provider 以 11 为准，V3 Placement/Autoscale 以 14 为准。03 是建模专项设计，04 是其标注 commit 的 xLLM 能力快照，05 是 V2.5 KV 内存层专项设计，06 记录取舍，07 记录评审进展，10 提供线上证据、问题优先级和后续优化约束，12 是实现事实与远端代码索引，13 是基于权威设计和实现事实生成的汇报视图，外部评估和 90 只提供证据与建议。评估结论没有同步到对应权威设计前，不构成实现要求。

首个生产版本直接交付 V2：必须完成 02 中 G-2、G-1、G0–G4/M0 的全部基础能力，并同时通过 08、09、11 的 V2 门禁。只完成 Provider SPI、协议底座、State Stream、`IsSchedulable` 或 Engine 原子准入，均只能标记为 V2 内部开发进度，不能作为 V1 或 V2 对外交付。M1/M2 在相同公共状态和 Provider Contract 上迭代，但不能削弱 V2 首发的正确性和功能范围。

06 和 07 是常设文档：06 记录已定决策及其重新引入条件，07 逐轮追加评审结论并跟踪问题状态。修改 01/02 关闭问题后，必须同步更新 07 的状态与关闭依据。
