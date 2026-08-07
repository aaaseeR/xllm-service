# xLLM Service GLM-5.2 线上瓶颈分析与优化约束

## 1. 文档信息

- 状态：线上证据分析；作为 01/02/03/05/08/09 后续设计与优化的输入
- 日期：2026-08-06
- 模型：GLM-5.2
- 网关数据：`online_logs/glm52_export.csv`
- Engine summary 数据：`online_logs/xllm_engine`
- Engine 完整实例日志：`online_logs/engine.log.tar.gz`
- 业务聚合辅助数据：`glm52_7days.csv`
- xLLM 源码核验基线：`8164a701bab7c3fc923db0ae077287c91fd0709b`，仅用于核验通用计时与旧路径；产生日志的实际 build 未包含在数据包中，不能用该 commit 的行号证明线上分支
- 分析目标：识别当前 xLLM Service 的容量、路由、Admission、Prefill、Decode 和可观测性瓶颈，不套用其他模型或历史 10s/15s SLO

本文是线上证据文档，不替代 01/02 的架构与实现规格。已确认约束已吸收到 01/02/03/05/08/09，映射见 07 §27/§29；没有阶段证据支持的推测不得直接写入调度策略。

## 2. 有效数据与分析口径

### 2.1 原始数据规模与覆盖天数

| 数据集 | 时间范围（Asia/Shanghai） | 覆盖时长 | 原始记录 | 用途 |
| --- | --- | ---: | ---: | --- |
| 网关原始日志 | 2026-07-31 23:55 至 2026-08-05 17:34 | 连续 4.736 天，跨 6 个自然日 | 2,273,519 | HTTP 结果、E2E 完成耗时、token、业务来源 |
| Engine summary 导出 | 2026-08-04 08:44 至 22:54 | 连续 14.17 小时，1 个自然日 | 229,346 行 | 全服务面阶段重建和网关关联 |
| Engine 完整实例日志 | 2026-07-28 17:58 至 2026-08-06 16:56 | 连续 8.957 天，跨 10 个自然日 | 24 个文件、1,538,161 行 | P/D 身份、配置、KV 容量、重试和失败原因 |
| 业务日聚合 | 2026-07-29 至 2026-08-05 | 8 个自然日 | 1,310 个 API Key/日聚合行、2,378,821 次调用 | 流量背景；不参与请求级阶段统计 |

“拿到了多少天数据”必须按覆盖完整度解释：**网关有连续 4.736 天原始请求，触及 6 个自然日（7 月 31 日只有最后约 5 分钟）；完整 Engine 包跨 10 个自然日、连续约 8.957 天，但 8 月 5 日前主要只有 4 对低流量 P/D 实例，不能称为全量集群 10 天。** 大部分高流量实例日志只覆盖 8 月 5–6 日。

网关三项时间同时存在的 1,515,726 条记录中，`elapsed_time = response_time + inner_time` 零反例；其余 757,793 条缺少至少一项，不能参与恒等式检验。这里的 `response_time` 是后端总耗时，不是用户可见 TTFT；网关日志本身没有首 token 时间戳。

### 2.2 Engine 去重与有效请求

同一 `request_id` 通常分别在 P 产生首 token、D 完成请求时输出 summary，不能把日志行数当请求数：

| 口径 | Engine summary 导出 | 完整实例日志 |
| --- | ---: | ---: |
| 成功 summary 行 | 229,346 | 1,496,209 |
| 去重成功请求 | 114,316 | 750,740 |
| P/D 两阶段完整请求 | 113,433 | 744,644 |
| 仅有一侧成功日志 | 883 | 6,096 |
| 错误 summary 行/唯一请求 | 0 | 2,169 / 2,169 |
| 其中仅有错误、无成功 summary | 0 | 1,729 |

两批日志有 6,782 个重复 `request_id`。合并去重后共得到 **858,274 个成功请求**，若把仅失败请求也计入，共 860,003 个唯一请求 ID。完整日志表按文件角色映射判断同一 ID 是否同时出现 P 首 token 与 D 完成 summary，因此得到 744,644 个完整请求和 6,096 个单侧请求；只按成功 summary 条数判断会得到 744,909/5,831，不能混用。完整日志中 744,349 个请求有 2 条成功 summary，560 个有 3 条；所有请求级指标均先按 ID、角色和事件去重。

### 2.3 网关与 Engine 关联质量

Engine 的 `x-request-id` 和 `x-request-time` 仍全部为空，Engine `request_id` 也不是网关 `request_id`。本轮继续使用 prompt token、output token、完成时间差不超过 1 秒且候选一对一的高置信关联：

| Engine 来源 | 高置信关联 | 四段区间可重建 | 关联时间范围 | 完成时间差 p50/p95/p99 |
| --- | ---: | ---: | --- | --- |
| summary 导出 | 87,932 | 87,919 | 8 月 4 日 08:44–22:54 | 3ms / 20ms / 124ms |
| 完整实例日志 | 43,022 | 43,022 | 8 月 1 日 00:00–8 月 5 日 17:35 | 3ms / 26ms / 158ms |

完整实例日志与网关的关联跨 5 个自然日，但前几天只代表被导出的低流量实例组。summary/完整日志中分别有 188/165 条记录在 1 秒窗内出现多个同形状候选，本轮取时间差最小且未使用的候选；它们只用于总体统计，不作为单请求因果证据。上述关联质量足以用于总体和实例组分析，不能替代生产 Trace ID。

### 2.4 专项可比数据集

| 子集 | 有效请求数/时长 | 用途 |
| --- | --- | --- |
| 六个生产来源 P 实例共同在线窗口 | 11.77 小时 | 比较相同工作负载下的路由份额、Admission 和 TTFT |
| 输入 1K–4K、输出 129–512 | 23,634 | 观察相似请求形状下并发重叠与 Decode TPOT |
| 6–7 对高流量完整 P/D 实例 | 约 69.5 万完成请求、6–26 小时/组 | 其中只选 4 对相同 TP2/DP16 布局进入 §5.3 的可比 TTFT/TPOT 表 |
| 4 对连续低流量完整 P/D 实例 | 49,228 完成请求、约 212–215 小时/组 | 作为约 0.016 RPS 的低负载对照 |

### 2.5 数据限制

- summary 导出中的 ID 前缀只能识别来源 P；完整实例日志可以按文件识别实际 P/D，但两者都没有 Gateway 共用的 global request ID。
- 完整日志明确记录了部分 Admission 重试和失败原因，但仍没有每个请求的完整 attempt、实时 KV used/reserved/headroom、running sequence 和 KV transfer 分段。
- 完整日志的 10 个自然日不是完整集群 10 天：旧实例连续、低流量，新实例主要从 8 月 5 日启动；跨实例总体分位数不能当成同一部署的容量曲线。
- 不同 P/D 机器有毫秒级时钟误差；约 -1ms 的阶段差按时钟噪声处理，秒级尾部不能由该误差解释。
- summary 成功请求存在 survivor bias；尤其 D `total_latency > 300s` 的 8,626 个请求不会进入成功网关关联，导致关联子集的 Decode p95 从全量 584.62s 截短到 115.52s。集群成功率仍以网关为准，Engine 错误日志用于定位机制而非反推全量失败率。

## 3. 核心问题与核心结论

### 3.1 核心问题

以下九个问题分别对应 §4–§12，先回答“现网哪里有问题”；后续章节再给出数据口径、证据、原因和约束。

1. **P 派发与 D Admission 存在条件性长尾。** 8 月 4 日事件窗口 P→D p95=18.96s，而完整日志稳态关联样本只有 3.7ms；瓶颈会随负载与实例组切换，详见 §4。
2. **路由不均会把热点实例推过非线性容量拐点。** 相同 TP2/DP16 profile 下，高流量组请求率约为低流量组的 83–96 倍，TTFT p95 和 TPOT 同时显著恶化，详见 §5。
3. **KV、batch budget 和 deadline 缺口已经转化为可用性问题。** D 最终 KV 池只有 21.10–25.28GB；日志同时出现 1,709 个硬容量失败、短请求批量失败和 8,626 个越过 300s 的继续执行请求，详见 §6。
4. **Admission 成功不代表首 token 路径健康。** 完整日志中 P→D p95 只有 3.7ms，但 D 创建→首 token p95=9.43s，长尾已转移到 Admission 后的 Prefill 或调度阶段，详见 §7。
5. **Decode 会在并发重叠下明显衰减。** 高流量组 TPOT p50 从低流量组的 20.5–21.7ms 升至 40.2–44.6ms，不能用单请求 Decode 性能外推在线容量，详见 §8。
6. **当前 TPOT 计时语义不可信。** summary 窗口 8.34% 的多 token 请求记录 TPOT=0，完整日志又出现 3,361 条负 inter-token latency，详见 §9。
7. **Prefix Cache 配置、容量和效果没有闭环。** 11/12 个 P 已启用 Prefix Cache，但 750,740 个去重成功请求命中仍全为 0；P 每 rank 只有 448–542 个 block，64K Prompt 已接近或超过可保留池，详见 §10。
8. **混合工作负载和输出截断污染全局指标。** 两批数据的 output p95 分别为 17,956 和 1,127，summary 窗口又有 21.7% 请求以长度上限结束，不能用一个全局分位数比较实例或版本，详见 §11。
9. **现有可观测性不足以支撑闭环调度。** Gateway、Service、P、D 没有统一 Trace 和 attempt 事件，91% 左右的临时 Admission 拒绝缺少 D 侧记录，也没有实时 KV headroom、并发和 transfer 分段，详见 §12。

### 3.2 核心结论

1. **主瓶颈不是固定的 Decode 内核，而是负载跨过不同资源拐点后，瓶颈在路由、Admission、Prefill 和 Decode 之间切换。** 8 月 4 日事件以 P 派发和 D Admission 长尾为主；完整日志高流量稳态下 P→D 已恢复毫秒级，但 Prefill/首 token 和 Decode 同时恶化。因此不能用单一“设备利用率高”或“KV 传输慢”解释全部 TTFT。
2. **当前最主要的系统性问题是资源控制闭环落后于真实容量。** Service 依赖滞后状态并过早绑定 D，D 的实际 KV 池远小于物理显存，临时不足使用固定周期重试，batch budget 失败会牵连其他请求，业务 deadline 又没有在 Engine 本地终止。这些机制会把一次容量不足放大成排队、批量失败和无效 Decode。
3. **容量问题已经同时影响性能、可用性和成本。** summary 同窗非流控成功率只有 72.4%，22.2% 为快速 503；8,626 个请求在网关 300s 超时后仍继续执行，最长 4,891s。仅优化算子或提高设备利用率，无法收回这些失败与无效计算。
4. **优化顺序必须是先建立可信证据和资源安全，再优化选点与内核。** O0 先修统一 Trace、阶段计时和结构化 Admission 事件；O1 修永久可行性预检、原子 reservation、单请求隔离和本地 deadline；O2 再用实时容量与 SLO residual 修复路由倾斜；O3 最后基于实测 surface 优化 Prefill、Decode 和 Prefix Cache。跳过 O0/O1 直接调整 Router 权重，无法证明收益，也可能继续放大错误状态。
5. **对 xLLM Service 设计的直接约束是：Service 做实时软选择，Engine 做最终原子准入和释放。** V1 需要 State Stream、P-D pair 就绪过滤、延迟绑定 D、结构化 Admission 终态和 Engine 本地 deadline；KV-aware Router 只有在 Prefix 位置、容量、生存上界和收益指标闭环后才能进入主评分。

## 4. 问题一：P 派发与 D Admission 的条件性长尾

### 4.1 阶段重建方法

对同一请求：

```text
P Request 创建时间 = P 首 token 日志时间 - TTFT
D Request 创建时间 = D 完成日志时间 - D total_latency
```

再与网关开始时间关联，可得到五个边界之间的四段区间：

| 阶段 | p50 | p90 | p95 | 可解释范围 |
| --- | ---: | ---: | ---: | --- |
| 网关收到 → P 创建 | 13ms | 250ms | 338ms | Gateway、Service 转发、P 建请求 |
| P 创建 → D 创建 | 0.2ms | 9.29s | 18.97s | P 派发队列、D Allocation/Admission、重试 |
| D 创建 → P 首 token 日志 | 426ms | 7.14s | 9.98s | Prefill 排队/计算及首 token 前准备 |
| P 首 token日志 → D 完成日志 | 5.72s | 87.06s | 115.52s | Decode；强烈受输出长度影响 |

`网关耗时 - D total_latency` 不能笼统称为“Service 时间”。它实际近似 `网关收到 → D Request 创建`，其中绝大部分尾部来自 P 派发和 D Admission。

该表只覆盖成功关联请求，Decode 尾部被网关 300 秒 deadline 系统性截尾：113,433 个完整 P/D 对的 Decode 区间 p95 为 584.62s，而高置信关联子集只有 115.52s。阶段 budget 必须同时报告全量 Engine 与成功关联视图。

### 4.2 完整日志说明该瓶颈是条件性的

新增完整日志提供了第二个时间切片：

| 关联样本 | 有效请求 | P→D p50/p95 | D 创建→首 token p50/p95 | TTFT p50/p95 |
| --- | ---: | ---: | ---: | ---: |
| 8 月 4 日 summary 导出 | 87,919 | 0.2ms / 18.96s | 426ms / 9.97s | 616ms / 23.87s |
| 8 月 1–5 日完整实例日志 | 43,022 | 0.4ms / 3.7ms | 448ms / 9.43s | 449ms / 9.46s |

这两个结果并不矛盾。8 月 4 日窗口发生了大范围 P dispatch/Admission 排队事件；完整日志覆盖的多数请求中 D Allocation 很快，长尾转移到 Allocation 成功后的 Prefill/调度。**因此 P→D 的秒级尾部不是固定成本，而是过载、路由倾斜或 D credit 不足时出现的容量相变。** 调度器必须同时约束两个阶段，不能用某一天的主瓶颈替代完整状态机。

### 4.3 日志事实与代码证据边界

产生日志的实际 build 未包含在数据包中。声明的核验基线 `8164a701` 不含 `Decode cannot fit request prompt at any load`、`Decode try_allocate failed temporarily` 等线上日志分支，其 `disagg_pd_service_impl.cpp:146-168` 也只有裸 404，不能作为线上二进制的行号证据。因此本节只保留下列可直接复现的运行事实：

- P 侧记录 2,414 条 `Decode AddNew non-200`，涉及 325 个请求；固定重试间隔 1,000ms，最多 32 次；
- 每请求重试次数 p50/p90/p95 为 4/21/30.8；从原始 attempt 起算的重试窗口为 4.03s/21.28s/31.20s；
- 1,447 个可定位到 D 的永久 KV 拒绝与 325 个重试请求交集为 0，说明线上 Engine 已经区分永久不可行与临时不足；
- 325 个重试请求涉及的 6 个目标 D 全部存在于归档且时间窗口完整，但 D 侧只记录 28 个临时失败，即约 91% 的临时 Admission 拒绝没有 D 侧事件；
- P 使用 `selected_instance=ip:port`，D 日志使用不同主机名，缺少 canonical incarnation，无法可靠对齐同一实例。

旧代码基线可用于提出“P 队列、同步 AddNew、拒绝后重排队”这一待验证机制，但不能证明线上 build 完全相同。8 月 4 日的短 Prompt P→D 长尾也不能全部归结为 325 个长 Prompt 重试；必须用结构化 dispatch/Admission attempt 事件拆分串行 RPC、排队、重试与 credit 影响。

### 4.4 227929 与 929141 的对比

| 指标 | 227929 | 929141 | 比值/差异 |
| --- | ---: | ---: | ---: |
| 网关→P 创建 p50 | 13ms | 12ms | 基本一致 |
| 网关→P 创建 p95 | 354ms | 344ms | 基本一致 |
| P→D 创建 p90 | 18.75s | 3.53s | 5.32 倍 |
| P→D 创建 p95 | 31.26s | 12.49s | 2.50 倍 |
| D 创建→首 token p50 | 908ms | 365ms | 2.49 倍 |
| D 创建→首 token p95 | 10.09s | 10.23s | 基本一致 |
| TTFT p50 | 2.29s | 0.44s | 5.24 倍 |
| TTFT p95 | 36.21s | 18.86s | 1.92 倍 |
| 重建 TPOT p50 | 28.79ms | 26.33ms | 1.09 倍 |
| 重建 TPOT p95 | 41.80ms | 39.49ms | 1.06 倍 |

已证实：两组成功请求的主要差异位于 P 派发/D Admission，Decode 不是主因。

summary 窗口仍无法知道 227929 最终落到哪个 D、每次 attempt 和当时 KV headroom；新增完整日志证明相同代码路径确实会发生 `try_allocate` 拒绝和秒级重试，但不能把另一时间窗的直接事件一一归因给 227929。

### 4.5 后续优化约束

- P→D 派发必须支持异步和批量，禁止单请求同步 RPC 成为串行瓶颈；
- D Admission 必须把现有永久/临时判断作为结构化结果返回，并携带可重试性、建议退避和实时 credit；每次 attempt 都必须产生终态事件；
- 重试必须有 attempt 上限、deadline、指数退避和换 D 能力；
- D reservation 必须有 TTL、幂等键和释放事件，避免重试造成资源泄漏；
- Service/Router 必须看到 P dispatch backlog、D alloc credit、KV headroom 和近期 rejection rate，不能只看粗粒度请求数；实例统一使用 Registry `incarnation_id`，不能靠 IP/主机名关联。

## 5. 问题二：路由不均把实例推过容量拐点

### 5.1 共同时间窗比较

六个相似生产来源 P 实例在 2026-08-04 08:54 至 20:40 共同在线，工作负载中位数均约为输入 1.77K、输出 230–256，可直接比较：

| 来源 P 实例哈希 | 请求数 | 平均请求率 | TTFT p50 | TTFT p95 | P→D p90 | P→D p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 227929960477534879 | 15,439 | 0.364 RPS | 3.03s | 50.02s | 23.91s | 45.90s |
| 5726147014706130127 | 13,050 | 0.308 RPS | 0.83s | 21.55s | 4.99s | 14.22s |
| 3851868113984591809 | 11,884 | 0.281 RPS | 0.74s | 22.23s | 8.93s | 17.51s |
| 9291410008051589590 | 8,854 | 0.209 RPS | 0.36s | 17.06s | 11ms | 12.05s |
| 6861700987355121635 | 8,238 | 0.194 RPS | 0.35s | 18.22s | 1.39s | 12.68s |
| 8110073178014793122 | 7,537 | 0.178 RPS | 0.42s | 23.32s | 11.53s | 20.83s |

请求率不是集群 QPS，而是每个来源 P 组的成功请求到达率。请求持续时间较长，低 RPS 仍可形成较高在途并发。

### 5.2 原因判断

227929 的请求率是 929141 的 1.74 倍、811007 的 2.05 倍。随着请求率升高，P→D p90 从毫秒/秒级跃迁到 23.91s，表现为典型 Admission 拐点，而不是线性变慢。

Service 自身在请求路径上的执行耗时并未增加，但 Service 的选择结果造成 P/D 负载倾斜。因此应区分：

```text
Service 处理慢       —— 数据不支持
Service 路由选择不佳 —— 数据强烈支持
```

### 5.3 完整日志中的低负载/高负载对照

完整日志还给出了同为 TP2/DP16 的跨实例对照：

| 指标 | 4 对低流量实例 | 4 对高流量实例 | 差异 |
| --- | ---: | ---: | ---: |
| 完成请求率 | 0.0156–0.0167 RPS | 1.325–1.540 RPS | 约 79–99 倍 |
| Prompt p50 | 19.9K–20.3K | 1.76K–1.78K | 高流量组反而短约 11 倍 |
| Output p50 | 53–56 | 107–149 | 高流量组约 2–2.8 倍 |
| P→D p95 | 2.5–2.7ms | 1.7–3.0ms | 都很快 |
| TTFT p95 | 0.832–0.844s | 8.83–11.22s | 恶化约 10.5–13.5 倍 |
| 重建 TPOT p50 | 20.5–21.7ms | 40.2–44.6ms | 恶化约 1.85–2.18 倍 |
| 重建 TPOT p95 | 34.0–35.8ms | 53.1–61.1ms | 恶化约 1.48–1.80 倍 |

这是当前最强的负载拐点证据：高流量组输入明显更短，P→D 仍是毫秒级，但 TTFT 和 TPOT 同时恶化。说明在该批实例上，问题主要发生在 Admission 成功后的 Prefill/continuous scheduler 和高并发 Decode，而不是网关转发或单纯长输入。

该表仍不是严格实验：输出长度、启动时间和其他流量形状不同。它可作为建模约束和回放假设，不能直接拟合硬件极限；最终需要固定 profile 的阶梯 QPS 实验验证。

### 5.4 后续优化约束

- 路由负载必须使用 token、在途完成时间、KV/credit 和预测 SLO，而不是请求个数；
- 对相同 profile 实例持续计算 token load、admission rejection 和 TTFT residual 的 max/min 与变异系数；
- Router 必须识别实例已接近拐点，在硬拒绝前主动降权；
- D 不能在请求早期永久锁定，首 token 前允许受控重选；
- 路由评估必须在共同在线窗口、相同请求形状和相同 profile 下比较。

## 6. 问题三：容量不足已经转化为可用性问题

### 6.1 全时间范围

网关 2,273,519 个请求中：

| 结果 | 数量 | 占全部请求 |
| --- | ---: | ---: |
| HTTP 200 | 1,017,088 | 44.7% |
| HTTP 为空 | 758,386 | 33.4% |
| 其中明确网关流控 | 749,470 | 33.0% |
| HTTP 503 | 473,841 | 20.8% |
| HTTP 500 | 16,051 | 0.71% |
| HTTP 504 | 6,565 | 0.29% |
| HTTP 404 | 1,301 | 0.057% |
| HTTP 502 | 287 | 0.013% |

排除明确的网关流控后，1,524,049 个请求成功率为 66.7%，503 占 31.1%，但流控并非常态：97.8% 集中在 8 月 1–2 日，这两天流控率为 41.6%/48.7%，8 月 3–5 日仅为 2.0%/3.2%/2.5%。聚合比例主要描述前两天的容量事件，不能外推为长期稳态。另有 8,916 个 HTTP 为空但没有 `flow_limit_result=1` 的请求，需补充网关失败原因分类，不能自动归为流控。

### 6.2 Engine 同时间窗

| 指标 | 数值 |
| --- | ---: |
| 网关请求 | 135,986 |
| 流控拒绝 | 6,239 |
| 非流控请求 | 129,747 |
| HTTP 200 | 93,975，非流控成功率 72.4% |
| HTTP 503 | 28,758，占非流控请求 22.2% |
| HTTP 500 | 2,768，占 2.13% |
| HTTP 504 | 2,737，占 2.11% |

以下错误耗时来自全量 4.736 天，不是上表的 Engine 同窗：

| 错误码 | p50 | p95 | 判断 |
| --- | ---: | ---: | --- |
| 503 | 7ms | 38ms | 快速拒绝，更像无可用容量/实例而非执行失败 |
| 500 | 65.65s | 165.31s | 执行或等待很久后失败 |
| 504 | 300.008s | 300.048s | 固定 300 秒上游超时 |

Engine 同窗内 503 p95 为 126ms，500 p50/p95 为 77.3s/200.1s；成功请求 E2E p50/p95 为 8.99s/131.2s，而全量仅 5.28s/33.8s，说明 §4/§5 观察窗口本身处于显著劣化状态。

2026-08-04 14:20–15:55 出现明显事故带：按非流控请求为分母，多个 5 分钟窗口错误率为 62.2%–80.3%；若把入口流控也计作错误才是 64%–86%。同期 Engine TTFT p95 最高达到 683s。时间桶不能证明每个错误都来自同一 D，但足以确认 Admission/排队长尾与可用性恶化在同一容量事件中共现。

### 6.3 完整日志直接暴露的资源失败

完整实例日志中的错误 summary 可按原因精确分类：

| Engine 失败原因 | 唯一请求 | 数据解释 |
| --- | ---: | --- |
| `Request prompt exceeds decode KV cache capacity` | 1,709 | 单请求永远无法放入目标 D |
| `No enough budget to schedule single sequence` | 425 | batch/scheduler budget 失败 |
| `Decode AddNew failed after max retries` | 16 | 32 次内仍未完成 D Admission |
| `No enough resource to schedule a single sequence` | 15 | 单序列资源不足 |
| `Failed to add new requests to decode instance` | 4 | P/D AddNew 失败 |

P/D 启动配置进一步解释了第一类失败及 Prefix 生存空间：

| 角色/布局 | KV cache/worker | KV block | 单 rank token 槽位 |
| --- | ---: | ---: | ---: |
| P（启用 Prefix Cache 的实例） | 5.94–7.20GB | 448–542 | 57,344–69,376 |
| D TP2 / DP16 | 21.10–22.36GB | 1,591–1,686 | 203,648–215,808 |
| D TP4 / DP8 | 25.04–25.28GB | 1,888–1,906 | 241,664–243,968 |

每个 Die 总显存约 61.27GB，但启动日志中的 available memory 只有 27.22–31.67GB，最终可分给 KV 的又只有 21.10–25.28GB。1,709 个硬拒绝 Prompt 的最小值为 203,529、p50 为 243,972，和上述单 rank KV 边界直接对齐。Admission 必须按实际 layout、目标 DP rank 和 block headroom 判断，不能按“单 Die 61GB”估算。

第二类错误不是简单的超长 Prompt：425 个 `No enough budget` 中 423 个落在同毫秒簇，只对应 25 个秒级事件，最大单簇 35 个请求；348 个 Prompt 小于 16K、397 个小于 64K，最短仅 71 token。这已经证明 batch budget 失败存在级联，而不是“可能由长 Prompt 导致”。需要修复 scheduler 的单请求隔离、错误文案和失败后预算恢复。

这些 Engine 错误来自部分实例，不能直接与 473,841 个网关 503 一一对应；它们的价值是证明具体失败机制，而非代表全量失败占比。

### 6.4 后续优化约束

- Goodput 必须同时统计成功率和 SLO，不能只统计成功请求延迟；
- 明确区分入口流控、Service 无候选、P dispatch 超时、D Admission 拒绝、执行失败和网关超时；
- 复用现网 Engine 已有的永久/临时判断并结构化返回；Service 在下发前按真实 D layout 做永久可行性预判，使永久不可容纳的 Prompt 不进入 D，临时 headroom 不足才允许换 D/退避；
- scheduler 遇到单个超预算请求时只失败该请求，恢复 batch budget 后继续处理其他已准入请求，禁止同毫秒批量清空；
- 在 Service 层提供有界等待和可预测的 overload response，避免同时出现快速 503 与 300 秒 504；
- 自动扩容、降级和 shed load 必须使用 admission rejection、queue age 和 SLO residual，而不是设备利用率单指标。

### 6.5 Deadline 后继续执行造成容量泄漏

113,433 个完整 P/D 对中，8,626 个（7.60%）D `total_latency` 超过网关 300 秒硬超时；这些请求的 p99 为 1,028 秒，最长 4,891 秒。它们既造成成功关联样本对 Decode 尾部的系统性截短，也说明客户端已经放弃后 Engine 仍在占用 KV、slot 和 HBM 带宽。

Service Cancel 只能作为快速路径，不能是唯一终止条件。每一跳必须传递剩余 duration，并由接收方转成本地 monotonic deadline；P 在排队/Prefill 边界、D 在 Decode 调度前主动检查，到期以 `DEADLINE_EXCEEDED` 正常终态停止并释放资源。故障 cleanup/fence 仍只处理 outcome 不明和异步内存安全，不能替代正常 deadline。

## 7. 问题四：Admission 成功后仍存在 Prefill 长尾

整体 D 创建到 P 首 token 生成的 p50=426ms、p95=9.98s。进一步只保留 `P 创建→D 创建 <=10ms` 的 72,813 个请求，TTFT 仍为：

| 指标 | 数值 |
| --- | ---: |
| TTFT p50 | 342ms |
| TTFT p95 | 8.62s |
| D 创建→首 token p50 | 342ms |
| D 创建→首 token p95 | 8.62s |

这说明 Admission 并不是唯一问题。D Allocation 成功后，P 执行队列、Prefill batch/chunk、长 Prompt 计算和首 token 前的数据准备仍会制造尾部。

完整实例日志再次验证了这一点，而且样本更大：744,644 个 P/D 阶段完整请求的 P→D p95 只有 2.4ms，D 创建→首 token p95 却达到 11.66s；与网关高置信关联的 43,022 个请求对应值为 3.7ms 和 9.43s。六个高流量实例组中，P→D p95 都只有 1.7–3.0ms，但 D 创建→首 token p95 为 8.82–16.21s。

启动配置显示 P 侧全部启用 chunked prefill、`max_tokens_per_batch=32768`；D 侧全部关闭 chunked prefill、`max_tokens_per_batch=512`。现有 summary 仍无法把 `D 创建→P 首 token` 拆成 P queue、各 Prefill chunk、首次 KV push 和首 token 生成，因此这里只能确认阶段范围。D 侧 512-token budget 与批量 `No enough budget` 失败也应作为独立 scheduler 缺陷验证，不能和正常 P Prefill 计算混为一类。

长 Prompt 占比不可忽略：

| Prompt 长度 | 请求数 | 占 Engine 唯一请求 | TTFT p50 | TTFT p95 |
| --- | ---: | ---: | ---: | ---: |
| 64K–128K | 10,803 | 9.45% | 3.58s | 27.45s |
| ≥128K | 4,902 | 4.29% | 6.51s | 31.25s |
| 合计 ≥64K | 15,705 | 13.74% | — | — |

当前 summary 不能继续拆分 Prefill queue、chunk 数、forward、KV transfer 和首 token 返回，因此只能确认阶段，不能确定该阶段内部的第一根因。

后续必须记录：`prefill_queue_enter/start/end`、每个 chunk token 数、batch token、forward 时间、KV transfer bytes/start/end、首 token generated/sent/acked。调度上应按 Prompt token 预算和 SLO 分级，防止长 Prompt 形成 head-of-line blocking。

## 8. 问题五：Decode 在并发重叠下衰减

固定输入 1K–4K、输出 129–512 后，用同一来源 P 在途请求数作为并发代理：

| 同来源重叠请求 | 请求数 | 重建 TPOT p50 | 重建 TPOT p95 | 完成耗时 p50 |
| --- | ---: | ---: | ---: | ---: |
| ≤4 | 12,147 | 24.11ms | 31.64ms | 6.08s |
| 5–8 | 3,742 | 27.23ms | 35.12ms | 7.76s |
| 9–16 | 2,625 | 30.31ms | 39.65ms | 8.70s |
| 17–32 | 4,255 | 32.99ms | 43.01ms | 8.82s |
| 33–64 | 865 | 36.36ms | 48.51ms | 9.88s |

从低重叠到高重叠，TPOT p50/p95 分别增加约 51%/53%。在请求形状受控后仍有该趋势，说明 Decode batching、KV 读取带宽或运行时调度存在容量衰减。

完整日志提供了独立佐证：同为 TP2/DP16 的高流量实例即使 Prompt p50 只有约 1.76K，其 TPOT p50 仍为 40.2–44.6ms；低流量实例 Prompt p50 约 20K，TPOT p50 反而只有 20.5–21.7ms。高并发对 Decode 的影响足以压过单序列 KV 长度差异。不过高低流量组的 Output、上线时间和业务混合不同，该结果是建模约束，不是严格的 hardware/runtime surface。

但来源 P 重叠不等于某一 D 的真实并发；在补齐 D 实例 ID、running sequence、KV bytes/step 和有效 HBM 带宽前，该结果只能作为强相关证据，不能用于拟合最终 Decode surface。

后续模型必须以真实 D 快照校准 `TPOT=f(active_sequences, kv_bytes_per_step, batch_shape, memory_bw, runtime_mode)`，并在 Admission 中使用 TPOT headroom 约束。

## 9. 问题六：当前 TPOT 指标存在计时语义错误

### 9.1 数据现象

- 9,758 个请求出现 `D total_latency < P TTFT`；
- 112,045 个输出大于 1 的请求中，9,348 个记录 `TPOT=0`，占 8.34%；
- 这些请求并非没有 Decode，而是日志公式的 guard 未通过。

### 9.2 代码原因

`request.cpp:81-94` 使用：

```text
generation_latency = total_latency - ttft
tpot = generation_latency / (generated_tokens - 1)
```

P 上的 TTFT 从 P Request 创建开始，D 上的 `total_latency` 从 D Request 创建开始。D 可以在 P 创建数秒后才 Admission 成功，因此两个时长不能直接相减：

```text
logged generation latency
  = (D finish - D create) - (P first token - P create)
  = real decode latency - (D create - P create)
```

Admission 等待越长，记录的 TPOT 越小；等待大于真实 Decode 时间时直接记录 0。这会系统性掩盖热实例的 Decode 成本。

### 9.3 重建结果与约束

用 `D 完成日志时间 - P 首 token 日志时间` 重建：

| 指标 | 重建 TPOT | 仅保留正值的原日志 TPOT |
| --- | ---: | ---: |
| p50 | 29.12ms | 28.90ms |
| p95 | 39.97ms | 39.70ms |

总体中位数接近不代表公式正确：原日志丢掉了最受 Admission 影响的 8.34% 请求，实例和尾部比较会产生选择偏差。

修复要求：所有阶段使用统一 monotonic span 或显式传递 `p_created_at`、`first_token_generated_at`、`d_created_at`、`decode_finished_at`；TPOT 必须由首 token 到最终 token 的同一计时域计算。修复前禁止用当前 `avg tpot` 直接训练 M1/M2。

### 9.4 完整日志补充证据

完整日志中 744,241 个多 token 成功请求有 1,404 个 TPOT=0，占 0.19%，低于 summary 窗口的 8.34%，说明问题强烈依赖版本/实例/Admission 状态，并非恒定比例。同时日志出现 3,361 条 `inter_token_latency_milliseconds is negative`，负值 p50=-3.10s、最小=-40.37s。该指标直接丢弃负值，会继续造成尾部分布的选择偏差。

因此不能因为完整日志总体的原始/重建 TPOT p50（43.1ms/43.27ms）接近，就认为计时正确。验收必须同时检查负值、零值、丢弃计数和同一 monotonic clock 的原始时间戳。

## 10. 问题七：Prefix Cache 状态不可用

summary 导出的 229,346 行以及完整日志的 750,740 个去重成功请求，`num_prefix_cache_tokens` 全部为 0。完整启动配置显示 11/12 个 P 启用 `enable_prefix_cache=1`，1 个 P 和全部 12 个 D 关闭。P 每 rank 只有 448–542 个 block、57,344–69,376 token；64K Prompt 约需 512 个 block，在较小 P 上无法完整保留，在最大 P 上也只剩约 30 个 block。由此可以排除“所有 P 都没启用”，并把“淘汰导致有效容量接近 0”提升为当前数据支持的主要解释，但仍不能判断 summary 是否漏报命中。

1. P 已启用但实际没有命中；
2. P 命中但没有传播到 summary；
3. P 侧 KV 容量和淘汰策略导致有效容量接近 0；
4. 多轮/Agent 请求经过模板化后无法复用前缀。

summary 窗口 13.74% 的请求输入超过 64K；完整日志中也有 27,422 个 ≥64K、7,592 个 ≥128K 的成功请求。若确实没有命中，Prefill 和 KV transfer 浪费会很大；若只是指标问题，Router 和建模同样无法利用真实命中。

需要同时输出 `prefix_lookup_tokens/hit_tokens/source/tier`、P/D 各自命中、实际 Prefill token、跳过传输 bytes、cache eviction 原因和 P 可用 block；用固定共享前缀压测验证指标闭环，并先以 Prefix 大小/驻留时间相对 P 可用 block 计算收益上界。生存容量不足时应优先调整 P KV 配额或验证 V2.5 分层 Store，不能只优化 Router 分数。

## 11. 问题八：工作负载混合和输出截断会污染总体指标

summary 导出包含 18 个 request ID 前缀，至少可识别三类不同工作负载：

| 工作负载族 | 典型 Prompt p50 | Output p50 | 特征 |
| --- | ---: | ---: | --- |
| 线上短输入/中等输出 | 约 1.77K | 约 256 | TTFT Admission 长尾明显 |
| 长 Decode 负载 | 约 10K | 约 3.4K–3.9K | TPOT 约 33–35ms，总耗时很长 |
| 长输入/短输出 | 约 19K–20K | 约 51–56 | Prefill 占主导，TPOT 较低 |

summary 导出的 output p50=256、p95=17,956、p99=32,000；完整实例日志的 output p50=99、p95=1,127、p99=1,748。两批 Engine 数据来自不同时间窗和实例覆盖，分布差异本身就证明全局分位数不能跨 pool/profile 直接比较。完整日志的 Prompt p50/p95/p99 为 1.78K/47.0K/131.6K，仍是明显的混合工作负载。

此外：

- summary 导出中 24,757 个请求以 `finish_reason=length` 结束，占 21.7%；
- `max_tokens=256` 的 33,183 个请求中，47.4% 达到长度上限；
- `max_tokens=32000` 的 26,472 个请求中，15.5% 达到长度上限。

完整日志另有 46,345 个 `finish_reason=length`，占去重成功请求 6.17%。两者不能合并成一个截断率，必须按 runtime profile 和业务类型分别解释。

这些上限可能是业务有意配置，但会同时影响结果完整性、Decode 负载和容量评估。后续报表必须按 `pool/model_revision/runtime_profile/prompt_bin/output_bin/max_tokens/finish_reason/business_class` 分组，禁止仅报告全局平均。

## 12. 问题九：现有可观测性不足以支撑闭环控制

### 12.1 已确认缺口

- `x-request-id`、`x-request-time` 全部为空；
- 没有 gateway/service/P/D 共用 Trace ID；
- summary 中 request ID 只能识别来源 P；完整日志只能靠文件名和个别 `selected_instance` 字符串恢复 D，事件本身没有统一 D ID；
- 完整日志记录了失败重试次数和部分 reason，但成功路径没有结构化 Admission attempt，仍没有 P dispatch queue age；
- 325 个临时重试请求只留下 28 个 D 侧拒绝记录；每次 Admission attempt 缺少稳定终态事件；
- 没有 D KV used/reserved/headroom、running sequences 和真实并发；
- 没有 KV transfer bytes、开始/结束/失败；
- 同一请求输出 2–3 条 summary，但没有明确 `role` 和 `event_type`；
- Error summary 与成功 summary 不在同一结构化事件流中，`No enough budget` 等原因只能解析自由文本；
- 记录了 KV 启动容量，却没有请求到达时的 per-rank free block，无法解释同一长度为何有时成功、有时进入 32 次重试。

### 12.2 V1 最小事件模型

| 事件 | 必需字段 |
| --- | --- |
| `gateway_received` | global_request_id、tenant、model、prompt_tokens、deadline |
| `service_routed` | P、D、候选集、选择分数、snapshot version、prediction |
| `p_request_created` | P instance、queue class、request attempt |
| `d_admission_attempt` | canonical D incarnation、attempt、requested KV、credit、result、reason；每次 attempt 恰好一个终态 |
| `d_reservation_created` | reservation_id、KV blocks/bytes、TTL、dp_rank |
| `prefill_started/finished` | queue age、batch token、chunk token、forward latency |
| `kv_transfer_started/finished` | src/dst、bytes、mode、duration、status |
| `first_token_generated/acked` | P/D/Service 时间戳、TTFT 各分量 |
| `decode_step_sample` | active sequences、KV bytes、batch shape、step latency |
| `request_finished/failed` | finish reason、error stage、TPOT、E2E、resource release |

所有事件必须使用同一个 global request ID 和 attempt ID。资源真相仍由 Engine 持有，Service 只消费事件和快照，不把观测系统变成正确性依赖。

### 12.3 Deadline 终态

Gateway 只给出业务 deadline，Service 在每次下发时计算剩余 duration；P/D 接收后转换为各自本地 monotonic deadline。P 在排队和 Prefill 边界检查，D 在每次 Decode 调度前检查；到期进入 `DEADLINE_EXCEEDED` 正常终态，停止后续 step 并释放 KV/credit/slot。该路径不能依赖 Service Cancel 必达，也不能使用跨机绝对时间。

## 13. 问题之间的因果链与结论边界

当前证据支持四条不同的因果链，调度不能用单一“设备利用率高”概括：

1. **8 月 4 日路由/Admission 事件：** Service 路由份额倾斜 → 热 P dispatch backlog 增长 → 同步 D Allocation 遇到 KV/credit 不足 → 非 200 后在原队列按秒重试 → P→D 进入秒级或十秒级长尾 → TTFT 恶化，并与快速 503、固定 300s 超时在同一事故窗口共现。
2. **完整日志高流量稳态：** 请求率从约 0.016 RPS 升到 1.33–1.54 RPS → P→D 仍保持毫秒级 → Prefill/首 token 前调度 p95 升至 8.82–16.21s → Decode TPOT p50 从约 21ms 升至 40–45ms → TTFT 与 TPOT 同时越过负载拐点。
3. **KV 硬边界与调度级联：** Prompt 接近或超过单 rank 20.4 万–24.4 万 token 槽位 → 永久不可容纳直接失败，临时不足进入最多 32 次、约 32s 的重试窗口；另一条 batch budget 异常路径会在同毫秒簇内让多个正常短请求一起失败。
4. **Deadline 容量泄漏：** 网关在 300s 终止请求，但 D 仍继续 Decode；8,626 个请求越过该边界，最长执行到 4,891s。被遗弃请求持续占用 KV、slot 和 HBM 带宽，又把健康请求推向 Decode 拐点。

当前证据不支持以下过度结论：

- 不能把所有 TTFT 长尾归因于 KV transfer；
- 不能把全部网关 503 归因于 1,709 个 Engine KV 硬拒绝；两者没有全局 Trace 闭环；
- 不能把“跨 10 个自然日”写成“完整集群连续采集 10 天”；
- 不能用单 Die 61GB 推导可用 KV；应使用实际 layout 的 block 数和 per-rank headroom；
- 不能把来源 P 哈希当成最终 D 实例；
- 不能把网关 `response_time` 当成 TTFT；
- 不能把全局 Engine 分位数当成某个 profile 的容量；
- 不能根据设备利用率判断 Admission、HBM 带宽或 KV headroom；
- 不能用当前原始 `avg tpot` 训练调度模型。

## 14. 解决方案、优先级与验收指标

九个问题与解决方案的对应关系如下。O0–O3 是依赖顺序，不表示后续项必须等前一项全部结束才开始实验；任何生产上线都必须先满足其依赖门禁。

| 核心问题 | 主要解决方案 | 优先级 |
| --- | --- | --- |
| P 派发与 D Admission 长尾（§4） | 异步/批量派发、永久可行性预检、有界换 D/退避、原子 reservation | O1 |
| 路由不均与容量拐点（§5） | State Stream 实时容量、token load/SLO residual 评分、热点快速重选 | O2 |
| KV/batch/deadline 可用性（§6） | per-rank block 预检、单请求 budget 隔离、Engine 本地 deadline 和释放 | O1 |
| Admission 后 Prefill 长尾（§7） | 拆分 queue/forward/transfer，按 Prompt 分桶优化 chunked prefill 与 token budget | O0 + O3 |
| Decode 并发衰减（§8） | 重建 effective bandwidth surface，按 TPOT headroom 做 Decode Admission guard | O3 |
| TPOT 计时错误（§9） | 统一 monotonic span，修复零值/负值并记录被丢弃样本 | O0 |
| Prefix Cache 不闭环（§10） | 补 Prefix/eviction/block 事件，先算生存上界，再决定扩 P KV、Router 或 Store | O0 + O3 |
| 工作负载混合与输出截断（§11） | 按 pool/profile/prompt/output/finish reason 分组报表 | O0 |
| 可观测性不足（§12） | 统一 Trace、attempt、incarnation、reservation 和阶段事件 | O0 |

### 14.1 O0：先修复可观测性和指标正确性

- global request/attempt/P/D/reservation ID 覆盖率达到 100%；
- 日志携带实际 build/schema/profile 标识；代码证据与 build 不匹配时只保留日志事实；
- P/D 阶段日志可一对一闭环，禁止依赖 token+时间猜测关联；
- `TPOT=0 && generated_tokens>1` 的非错误请求归零；
- negative inter-token latency 归零，任何被 guard 丢弃的样本单独计数并报警；
- 每次 Admission attempt 恰好一个结构化终态；Admission、queue、KV 和 transfer 事件缺失率可监控；
- 报表按 workload/pool/profile 分组。

### 14.2 O1：修复 P dispatch 与 D Admission

- 单请求同步派发改为异步/批量；
- D 拒绝后支持有界退避、换 D 和 deadline；
- Service 下发前做永久可行性判断：`required_blocks > max_blocks_per_sequence/rank` 立即失败，不进入 D；Engine 复用现有永久/临时分流并结构化返回，临时 headroom 不足才允许换 D/退避；
- Admission reservation 具备幂等、TTL、commit/release；
- 修复 D scheduler 的 batch budget 隔离，单请求超预算不得批量失败其他短请求；
- Service 逐跳下发剩余 deadline，P/D 转成本地 monotonic deadline；D 到期后停止新 Decode step并释放资源，不依赖远端 Cancel 必达；
- `deadline_exceeded_to_stop_ms`、`deadline_exceeded_to_release_ms` 和过期后继续生成 token 数进入 V1 门禁；
- 分别考核 P dispatch queue age、D allocation RPC、retry count 和总 Admission wait；
- 在安全负载内 P→D p95 必须保持在 profile 定义的 Admission budget 内，达到拐点前主动降权或拒绝。
- 同时给 `D 创建→首 token` 单独设 budget；P→D 健康不能掩盖 Prefill/调度长尾。

### 14.3 O2：修复 Router 负载倾斜

- 使用 prompt/decode token、在途完成时间、KV headroom、credit 和 SLO residual；
- 相同 profile 共同时间窗内比较实例 token load 和 rejection rate；
- 对热点、状态陈旧和 D 首次拒绝提供快速重选；
- 路由目标以 SLO goodput 和失败率为主，不以设备利用率或请求数为主。

### 14.4 O3：优化 Prefill、Decode 和 Prefix Cache

- 按长短 Prompt 分类验证 chunked prefill 和 token budget；
- 使用真实 D 并发重建 Decode effective bandwidth surface；
- 在预测 TPOT headroom 不足时拒绝新的 Decode/KV/本地 Prefill 干扰；
- 以 P 可用 block、Prefix 大小和驻留时间先计算命中生存上界；再验证“P 已启用但 75 万请求命中仍为 0”的指标闭环，决定扩 P KV、KV-aware Router 或 V2.5 Store 的优先级；
- 每项优化同时报告 TTFT、TPOT、E2E、成功率和 SLO goodput。

## 15. 下一轮验证计划

1. 选择相同 runtime profile 的 P/D 池，补齐真实 P/D/attempt/KV 事件；
2. 固定请求形状，分别做阶梯 arrival rate 和阶梯并发，定位 Admission 与 TPOT 拐点；
3. 固定 P，轮换 D；固定 D，轮换 P，拆分 P dispatch、D credit 和硬件差异；
4. 对 D Admission 注入可控拒绝，验证退避、换 D、TTL 和幂等释放；
5. 按 TP2/DP16 与 TP4/DP8 分别重放 200K、204K、216K、240K、244K Prompt，验证 block 边界、永久拒绝和临时 headroom 不足的状态码；
6. 在一批 1K–16K 请求中注入一个超长/超预算请求，验证是否复现同毫秒批量 `No enough budget`，并以“只失败单请求”为门禁；
7. 对 1K、16K、64K、128K Prompt 分别重放，拆出 queue、forward、transfer；
8. 用共享前缀和无共享前缀 A/B 验证 Prefix Cache 指标、TTFT 和 transfer bytes；
9. 将 Service Cancel 丢弃，验证 P/D 本地 deadline 仍能停止请求，并测量过期后 token 与资源释放时延；
10. 在 448–542 block 的现网 P 配额和更大配额下重放相同 Prefix trace，验证收益上界与淘汰主因；
11. 将实测 surface 回灌 03 的 M1/M2 和 Scheduler replay，再用线上残差校准；
12. 只有在请求级 Trace 闭环后，才比较新旧 Router 的 SLO goodput 和成本。

本文后续更新必须保留原始样本数、时间窗口、过滤条件、关联覆盖率和代码基线，避免把一次线上事件的相关性误写成跨版本、跨模型的固定规律。
