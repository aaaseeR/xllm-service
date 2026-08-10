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

# V3 Placement/Autoscale 线上验证与反馈手册

## 1. 目的和停止线

本手册把 V3 从 `CPU_VERIFIED / NPU_AND_CLUSTER_PENDING` 推进到真实集群证据。任何阶段
出现 Router SLO 回退、重复副作用、旧 leader 写入、无证明强杀、operation 无界增长或
指标/日志无法解释时，立即把 reloadable gflag `placement_mode_override=1` 切回 SHADOW，
保留现场并停止进入下一阶段。

SHADOW 不生成新 desired/operation，但已有未知 operation 仍会 Query/收敛。不得手工删除
UNKNOWN command/status，不得把 committed DRAINING 强改为 READY，不得绕过 incarnation
fencing。需要完全停 loop 时才用 `0`；常规回滚使用 `1`。

## 2. 上线前置

必须逐项记录版本、负责人和证据链接：

- xllm-service、xllm、Provider Agent、Engine、部署 gateway 的不可变 commit/build id；
- 独立测试 pool 的 provider/model revision/role/profile digest、NPU 型号、拓扑和设备数；
- profile 固定负载数据：load/warmup p50/p95/p99、P token/s、D token/s、A request/s、
  TTFT/TPOT residual、HBM/KV 容量；
- etcd 健康、配额、compact/backup、master key 与 Placement 三类 key 的 dashboard；
- deployment gateway 按 operation id 幂等，CREATE 返回新 uid/incarnation，TERMINATE 返回
  精确终止证明；幂等记录保留时间不短于 Service `terminal_retention_ms`；
- xLLM Native/vLLM-Ascend agent 能先关闭 admission，再返回 P queue、transfer、D
  reservation/sequence、output、cleanup 的有界 pending proof；
- Registry descriptor、fresh EngineState、lease 删除与进程/设备状态可以交叉核验；
- 使用严格样例配置，替换全部 `REPLACE_WITH_*`，启动参数包含
  `--placement_config_path=<absolute-json-path>`，初始 override 固定为 `1`。

配置门：`forecast_horizon_ms >= load_warmup_p99_ms`，protected devices 不超过全局预算，
`executor.max_records >= reconcile.max_operations_per_pool`，所有内部 token 非空且不含空格/
控制字符，`executor.terminal_retention_ms >=
reconcile.terminal_visibility_grace_ms`。配置解析失败必须阻止进程启动，不能降级为隐式
ENFORCED。

## 3. 必收证据

每个测试窗口导出以下时间序列和原始 VLOG；pool/model/engine 明细只在日志中关联，不添加
无界 metrics label。

| 类别 | 证据 |
| --- | --- |
| 控制环 | `xllm_service_v3_placement_leader/mode/pools/operations`、cycle latency、cycles outcome |
| 输入与决策 | observations outcome、recommendation action、forecast/actual、queue/reject、TTFT/TPOT、KV ratio、cache loss、budget reason |
| 副作用 | operations ADDED/DRIVEN/COMPACTED、operation id/action/status/attempt、gateway request/result |
| 生命周期 | descriptor/EngineState incarnation、LOADING/WARMING/READY/DRAINING/ABSENT、全部 pending proof |
| 请求面 | goodput、reject、queue wait、TTFT、TPOT、E2E、首 token 前 retry、terminal reason |
| 设备面 | NPU 利用率、HBM/KV used/free、模型加载、通信、进程/容器、设备释放时间 |
| 存储面 | desired/command/status 数量与字节、CAS/revision conflict、leader epoch、etcd latency/error |

每个阶段至少保留：配置、开始/结束时间、基线/实验流量、预期、实际、关键截图/查询、异常
operation 全记录、结论和下一步。线上阈值必须由本次真实数据校准，不从 CPU 默认值外推。

## 4. 分阶段执行

### O0：静态与 SHADOW 基线

1. 只部署一个测试 pool，override=`1`，至少覆盖一个完整峰谷周期；
2. 对比 recommendation 与人工容量、实际 READY、P/D 工作量和 SLO；
3. 验证 follower 不写 Placement，leader epoch/切换可解释，Router 指标与未启用时等价；
4. 校准 forecast headroom、hold、stabilization、cooldown、budget 和 cache-loss。

通过：无副作用；recommendation 无持续振荡/系统性欠配；cycle p99 明显低于 interval；日志能
从 observation 解释到 recommendation。否则留在 SHADOW。

### O1：create-only

1. override=`2`，desired 先只允许增加一个副本；
2. 验证 desired CAS → command/status → gateway CREATE → 新 incarnation → LOADING →
   WARMING → Registry/fresh EngineState READY；
3. 期间注入相同 operation 重试和一次响应丢失，确认 execute 一次、后续只 Query；
4. 逐步加入 P、D、Aggregated 和第二模型，但每次只改变一个变量。

通过：无重复实例；READY 证明不依赖 gateway 自报；load/warmup p99 在 profile 界内；新流量
只到 READY；旧/错误 incarnation 被拒绝。失败立即 SHADOW，并保留 command/status。

### O2：单 pool 单副本 scale-down

1. 选 cache value 最低且无活跃 reservation/transfer 的 fresh READY victim；
2. override=`3`，限制 `max_scale_down_step=1`、`max_drain_per_cycle=1`；
3. 验证 admission closed 后新请求不进入 victim，全部 P/D/output/cleanup pending 收敛为零；
4. commit 后验证不能 CANCEL；需求反弹时必须先 CREATE replacement，容量恢复后才
   TERMINATE；
5. 验证旧 lease/进程/真实 HBM 和设备资源释放，新进程必须使用新 incarnation。

通过：无在飞请求丢失/重复/乱序；无证明不 terminate；HBM/设备释放可观测；cache-loss 与
warmup 代价符合 planner 输入。任何 pending 无法归零时保持 DRAINING 并 SHADOW。

### O3：P/D 与流量阶梯

按低→中→拐点前→拐点后→回落执行，每级稳定至少三个 observation window：

- 短 prompt/长 output，验证 D 独立扩容；
- 长 prompt/短 output，验证 P 独立扩容；
- 长 prompt/长 output，验证 P/D 分别计算且不固定比例；
- KV pressure、queue/reject、TTFT、TPOT 单独及组合触发；
- 突增/突降、OOD、样本不足和全局 device budget overcommit。

通过：扩容先于持续 SLO 违约，缩容满足稳定/经济门；预算优先级稳定；无正反馈振荡；V2
Router goodput、错误和公平性满足预定门限。

### O4：故障矩阵

至少执行：leader kill、follower kill、etcd stall/timeout、旧 leader 网络恢复、gateway
timeout/5xx/响应丢失、Engine kill during load、Engine kill during drain、Agent restart、
Registry stale/lease delay、Provider conflict/fenced、operation snapshot 达容量门。

逐项验证：旧 leader 无写入；新 leader先恢复再动作；UNKNOWN 只 Query；corrupt/oversized/
revision conflict fail closed；Router unaffected；同 incarnation 无并发冲突动作；FAILED/
FENCED 保留且告警；成功终态按 retention 有界回收。

### O5：多模型 24h+ soak

至少覆盖一个自然峰谷和一次滚动发布。记录 operation ledger 高水位、GC 速率、etcd 字节、
cycle p99、模式、desired/actual 偏差、load/drain 时延、HBM/KV、goodput/SLO、公平性和成本。

通过：无 operation/内存/线程/连接增长；无 desired/actual 长期漂移；无跨模型预算饥饿；无
频繁扩缩；无 Router 回归。完成后再逐 pool 放大，不能直接全量。

## 5. 反馈与缺陷分级

| 级别 | 条件 | 动作 |
| --- | --- | --- |
| S0 | 请求正确性、无证明 terminate、旧 leader 写、重复副作用、设备泄漏 | 立即 SHADOW/隔离 pool，冻结证据，停止全量 |
| S1 | SLO/goodput 显著回退、持续欠配、operation/etcd 无界、无法收敛 | SHADOW，回滚阈值/版本，24h 内根因 |
| S2 | 抖动、预测偏差、成本收益不足、日志/指标缺口 | 保持当前小流量或 SHADOW，校准后重测 |

每个缺陷必须包含 build id、mode、leader tuple、pool key、observation/desired generation、
operation id、engine uid/incarnation、原始状态/日志、请求影响和可复现步骤。修复先回 CPU/
loopback 增加确定性回归，再回到失败的线上阶段，不跳级。

## 6. 升级为 VERIFIED 的签字项

- O0-O5 全通过，S0/S1 清零；
- xLLM Native 和实际启用的 vLLM-Ascend profile 均完成 conformance；
- NPU load/warmup、真实 HBM/cache-loss、drain 和设备释放有原始证据；
- leader/etcd/gateway/Engine 故障矩阵通过；
- 生产阶梯与 24h+ soak 通过，阈值和 dashboard 已固化；
- 能力状态文档补充集群、commit、时间、结果和证据链接。

任一项缺失时保持 `NPU_AND_CLUSTER_PENDING`。
