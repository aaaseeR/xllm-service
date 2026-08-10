# xLLM Service V2 当前能力与代码索引

更新时间：2026-08-10
状态：`CPU_AND_OFFLINE_CLUSTER_VERIFIED / NPU_AND_ONLINE_PENDING`

## 1. 一页结论

首个交付版本直接是 V2，没有独立 V1。当前代码已经形成可编译、可在 CPU/Torch CPU
上验证的完整控制链：统一 API 与请求身份、Provider Contract、Registry/State/KV
软视图、有界公平流控、P/D 选择、三种 xLLM Native 执行模式、vLLM-Ascend 严格
Agent、attempt/commit/fencing、deadline/cancel、输出定序、KV-aware SHADOW/ENFORCED
门禁以及全链路观测。

这不是 NPU 生产验收结论。CPU 结果只证明协议、状态机、并发账本、错误和资源不变量；
simulated HBM 以及包装真实 `BlockManager` free-list 的 CPU 契约测试，不证明 CANN、真实
HBM/DMA/Link、设备性能或生产 SLO。独立 etcd、双 Service、2P+2D 的离线多进程 E2E 已
证明进程内实现的租约恢复、self-fencing、Leader 切换和并发链路，但不冒充生产网络、
部署系统或 NPU 集群结论。

本文所有代码位置均指向远端 `service_dev` 分支：
[xllm-service](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev) 与
[xLLM](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm/tree/service_dev)。

## 2. 当前能力矩阵

| 能力 | 当前支持 | 生产代码 | 永久验证 | 状态/限制 |
| --- | --- | --- | --- | --- |
| API 接入与规范化 | OpenAI Completion/Chat、Anthropic Messages；统一 `CanonicalRequest`、UUIDv7 `request_uid` | [HTTP Service](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/http_service/service.cpp)、[Canonical Builder](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider/canonical_request_builder.cpp) | [HTTP/API tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/xllm_service/http_service) | CPU_VERIFIED；真实 Gateway/客户端矩阵待集群验证 |
| 身份与 KV 隔离 | tenant/flow/priority 只接受可信网关声明；标准 SDK `user` 仅与网关注入的已认证 client-id 派生；否则使用有 TTL、可轮转 HMAC token | [Trust Policy](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/http_service/request_trust_policy.cpp)、[Options/flags](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/common/options.h) | [Trust tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/xllm_service/http_service/request_trust_policy_test.cpp) | 默认 fail-safe；可信 header 开关默认关闭，网关不可旁路 |
| Provider Contract | 不按产品名猜能力；Descriptor、profile、capability、mode、incarnation 做硬兼容 | [Service Contract](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider/provider_contract.cpp)、[统一 proto](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm/tree/service_dev/xllm/proto/provider.proto) | [Service contract tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/xllm_service/provider/provider_contract_test.cpp)、[wire tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm/tree/service_dev/tests/proto/provider_protocol_test.cpp) | CPU_VERIFIED；跨 Provider P/D 稳定拒绝 |
| Registry 与观测状态 | etcd 保存低频身份/lease；高频 EngineState、LinkState、KV events 走独立有界通道；陈旧状态 fail closed | [Engine Registry](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider/engine_registry.cpp)、[State Outbox](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider/state_stream_outbox.cpp)、[KV State](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider/kv_state_outbox.cpp) | [Registry/state tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/xllm_service/provider)、[offline E2E](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/e2e/online_cluster_stress.py) | CPU/OFFLINE_CLUSTER_VERIFIED；生产 etcd 集群和网络分区待线上验证 |
| 有界流控与公平 | crash/memory/queue/token/tenant/model 多重硬预算；priority band、tenant→flow 两级轮转、FCFS/EDF、starvation bound | [Flow Queue](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/scheduler/flow_control_queue.cpp) | [Flow tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/xllm_service/scheduler/flow_control_queue_test.cpp) | CPU_VERIFIED；含 late-flow 公平与 flow 状态回收回归；生产饱和曲线待测 |
| P/D 候选与选点 | model/provider/profile/mode/incarnation/link/state 先硬过滤；公平出队后再做 load/KV/SLO 排序和稳定 tie-break | [Route Selector](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider/provider_route_selector.cpp)、[Instance Manager](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/scheduler/managers/instance_mgr.cpp)、[KV Planner](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider/kv_route_planner.cpp) | [Route/planner tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/xllm_service/provider) | CPU_VERIFIED；ENFORCED 需完整校准参数和灰度 bucket |
| HBM KV-aware K0–K2 | canonical chained block hash；本地有界 KVIndex；HBM 精确位置与新鲜度；HOST 只作上界；SSD/STORE 在 V2 删除并 reserved | [KV Index](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider/kv_shadow_index.cpp)、[KV metrics](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider/kv_route_metrics.cpp)、[xLLM hash](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm/tree/service_dev/xllm/core/framework/prefix_cache/canonical_block_hash.cpp) | [Service KV tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/xllm_service/provider)、[xLLM hash tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm/tree/service_dev/tests/core/framework/prefix_cache/canonical_block_hash_test.cpp) | CPU_VERIFIED；真实 HBM 命中收益与多卡一致性待 NPU |
| xLLM Native 执行 | `REMOTE_PD/LAYERWISE_PUSH/P_FIRST`、`LOCAL_PREFILL_DECODE`、`PREFILL_ONLY`；后两者默认关闭并经 capability/bucket/token cap 门禁 | [Mode Selector](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider/native_execution_mode_selector.cpp)、[Plan Builder](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider/execution_plan_builder.cpp)、[Engine validator](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm/tree/service_dev/xllm/core/framework/request/native_execution_plan_validator.cpp) | [Service mode tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/xllm_service/provider/native_execution_mode_selector_test.cpp)、[Engine plan tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm/tree/service_dev/tests/core/framework/request/native_execution_plan_validator_test.cpp) | CPU_VERIFIED；NPU 吞吐、显存和尾延迟门未完成 |
| attempt、准入与资源收敛 | D 原子 admission；`request_uid+attempt_seq+incarnation` 幂等；Submit 结果不明保留 hold；Query/Cancel/fence/TTL 后才释放或替换 | [Execution Hold](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider/execution_hold.cpp)、[Attempt client](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider/attempt_control_client.cpp)、[Engine attempt table](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm/tree/service_dev/xllm/core/distributed_runtime/attempt_lifecycle_table.cpp) | [Service resource tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/xllm_service/provider)、[Engine attempt tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm/tree/service_dev/tests/core/distributed_runtime/attempt_lifecycle_table_test.cpp) | CPU_VERIFIED；真实进程/DMA fencing 证据待 NPU/集群 |
| KV 资源契约 | simulated HBM 覆盖容量、地址、内容/checksum、OOM/碎片、transfer pin、fencing、ABA 与回收；生产适配器直接持有真实 BlockManager `Block` RAII | [BlockManager backend](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm/tree/service_dev/xllm/core/framework/block/block_manager_kv_resource_backend.cpp)、[KV resource contract](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm/tree/service_dev/xllm/core/framework/kv_cache/kv_block_resource.h) | [simulated HBM + production-adapter tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm/tree/service_dev/tests/core/framework/kv_cache/simulated_hbm_test.cpp) | CPU_VERIFIED 契约；pool accessor 尚无运行时消费者，NPU 接入必须兑现或删除；不冒充真实 HBM |
| 输出、deadline 与重试 | 跨 P/D sender sequence、重复/迟到 fencing、有界 gap recovery；客户端断连和 deadline cancel；只在首输出前有限重试 | [Output Sequencer](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/request/output_event_sequencer.cpp)、[Deadline queue](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/request/request_deadline_queue.cpp)、[Retry budget](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/request/first_output_retry_budget.cpp) | [Request tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/xllm_service/request) | CPU_VERIFIED；首输出后绝不自动重试 |
| vLLM-Ascend Provider | V2 只开放严格 `AGGREGATED`；Agent 是唯一 ingress，带 token 鉴权、attempt/deadline/query/cancel/fencing 与健康门 | [Agent](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/vllm_sidecar/agent.py)、[Attempt ledger](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/vllm_sidecar/attempts.py) | [Agent pytest](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/vllm_sidecar/tests) | CPU_VERIFIED 控制链；真实 Ascend、同命部署和原始端口隔离待验证 |
| Debug 与性能分析 | 五级关联键、稳定事件族、常开低基数 bvar、`VLOG(1)` 有界逐请求 JSON、周期 cluster snapshot、ring drop 自监控 | [Event Recorder](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/observability/request_event_recorder.cpp)、[Scheduler events/snapshot](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/scheduler/scheduler.cpp)、[共享事件 proto](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm/tree/service_dev/xllm/proto/observability.proto) | [Observability tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/xllm_service/observability)、[wire tests](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm/tree/service_dev/tests/proto/request_event_protocol_test.cpp) | CPU_VERIFIED；集群阈值、采样开销和告警需生产校准 |
| 离线模拟线上硬门 | 生产 Service 二进制 + etcd + 2P/2D + 并发 HTTP；Engine loss、短/长 etcd outage、物理 self-fence、新 incarnation、Leader SIGKILL、Torch HBM 清零 | [E2E harness](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/e2e/online_cluster_stress.py)、[Native Engine](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/e2e/mock_native_engine.cpp)、[gate README](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/tests/e2e/README.md) | smoke/stress JSON report 与独立进程日志 | CPU/OFFLINE_CLUSTER_VERIFIED；smoke 为 CI 硬门、stress 为版本交付硬门 |
| 仓库门禁 | pin/proto 校验、Service 全量 CTest、三个 serving 二进制、vLLM pytest、xLLM CPU contract | [Coding CI](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/.coding-ci.yml)、[GitHub mirror CI](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/.github/workflows/v2_cpu.yml) | [pin guard](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/scripts/verify_v2_pin.sh) | 仓库定义完成；Coding 首次远端执行和受保护分支 required check 仍需平台确认 |

## 3. P/D 选择策略：当前真实顺序

选择不是“从全部实例中取最低负载”这一条公式，而是以下有序决策：

1. **流控先行：** 请求先取得 Service queue、tenant/model/token/bytes 账本；priority、
   tenant→flow 公平性决定下一个可出队请求。KV 或负载分数不能插队。
2. **能力硬过滤：** 按 model revision、Provider、profile、执行模式、KV layout、
   incarnation、Registry lease、EngineState freshness 和 `LinkState=READY` 删除不合法候选。
3. **模式选择：** xLLM Native 的 REMOTE/LOCAL/PREFILL_ONLY 由 capability 与显式发布门
   决定；vLLM-Ascend V2 固定 AGGREGATED；跨 Provider split 直接拒绝。
4. **候选枚举：** 只生成完整、有限、可复核的计划；截断会显式记录，不把未枚举候选
   当成已评估。
5. **软评分：** 在合法集合内组合 Prefill queue/token cost、Decode running/headroom、
   HBM Prefix 收益、transfer cost、SLO residual 与不确定性；HOST 不进入 V2 cost。
6. **稳定选择：** near-equal 区间使用稳定 tie-break，防止状态微小抖动造成 route churn。
7. **Engine 最终准入：** Service 选择只是建议；D allocator/credit 的原子 admission 才是
   最终资源真相。拒绝可在 deadline/attempt 预算内换有限候选。
8. **提交屏障：** P/D holder 收敛并完成 GenerationCommit 后才提交输出；首输出一旦对外，
   retry window 永久关闭。

主实现入口是
[Scheduler](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/scheduler/scheduler.cpp)、
[ProviderRouteSelector](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider/provider_route_selector.cpp)、
[KVRoutePlanner](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/xllm_service/provider/kv_route_planner.cpp) 和
[xLLM Disaggregated Scheduler](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm/tree/service_dev/xllm/core/scheduler/disagg_pd_scheduler.cpp)。

## 4. 容错与错误语义

| 失败面 | 当前处理 | 对客户端/运维的证据 |
| --- | --- | --- |
| 输入、SLO、deadline、trust 非法 | fail closed，不进入调度 | 稳定错误类别 + `request_uid`；不返回原始内部 message |
| Service queue/公平预算耗尽 | 精确回滚账本，拒绝或在 deadline 内排队 | `FLOW_CONTROL_REJECTED status=...` + queue snapshot；FLOW_CONTROL 事件 |
| 无合法 P/D/Provider plan | 不降级到不兼容候选 | ROUTE REJECTED，区分 stale/capacity/mode/permanent infeasible |
| D 原子 admission 拒绝 | 按永久/临时分类；只在有界候选和 deadline 内继续 | D_ADMISSION 终态含 disposition、reason、attempt 与累计 RPC 时间 |
| Submit/Cancel RPC 结果不明 | 资源 hold 保持，Query/Cancel/fence/TTL 证明终态前不替换 | attempt key 与 RESOURCE_RELEASE STARTED/terminal |
| Registry/State 陈旧 | 区分 membership 与 observation；停止新选点，不伪造 DELETE | readiness reason、state/registry blind、stale 指标与 snapshot |
| Engine incarnation 变化 | 旧 output、event、reservation 全部 fencing | sender/coordinator incarnation + attempt 关联 |
| 输出乱序、重复或 gap | 有界重排；重复丢弃；gap 查询/超时后稳定终止 | OUTPUT 事件、gap/reorder/drop 指标 |
| deadline 或客户端断连 | cancel in-flight，释放 Service 账本；Engine 本地 deadline 兜底 | CANCEL/REQUEST_END/RESOURCE_RELEASE |
| Service shutdown | 先停止准入并退出 READY，有界 drain；未收敛资源显式告警 | readiness、shutdown unresolved hold 指标与终态事件 |

状态类型不强行合并：Provider、KV replica、Service flow-control 和 Engine physical resource
各自保留强类型结果，只在唯一 adapter/ingress 边界翻译为稳定公开错误与事件。统一的是
`Code + Result + bounded message + request_uid` 的纪律，不是一个丢失领域语义的万能枚举。

## 5. Debug 与集群性能解读

全链路关联键为：

```text
global_request_id / trace_id
  -> request_uid
    -> attempt_seq
      -> provider + model_revision + profile_digest + execution_mode
        -> engine_uid + incarnation_id + dp/rank + link/prefix identity
```

常态使用低基数指标和周期 `xllm_service_cluster_snapshot` 判断容量、陈旧状态、queue、
KV pressure、route/fallback、admission 和 cleanup；问题窗口对定向实例开启 `--v=1`，按
同一关联键还原 `REQUEST/FLOW_CONTROL/ROUTE/D_ADMISSION/PREFILL/TRANSFER/FIRST_OUTPUT/
DECODE/RESOURCE_RELEASE/REQUEST_END`。事件 ring 丢失可见但不能反压推理；普通日志不写
完整 prompt、输出、token、Authorization、API key 或 HMAC secret。

详细操作步骤见
[观测运行手册](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/docs/design/implementation/OBSERVABILITY_RUNBOOK.md)。

## 6. 验证证据与未完成项

本次本地 Linux CPU 门为 Service 511/511；三个 ARM64 Debug serving ELF 均完成编译和
动态链接检查。xLLM 公共 contract、attempt、registration、Provider wire、RequestEvent
和 simulated-HBM 测试继续由独立 CPU 门覆盖。精确命令由仓库
[Coding CI](http://xingyun.jd.com/codingRoot/xLLM_AI/xllm-service/tree/service_dev/.coding-ci.yml)
固化。

达到生产 `VERIFIED` 仍缺：

- NPU/CANN 的三种 Native mode 与 vLLM-Ascend AGGREGATED 实测；
- 真实 HBM 分配、碎片、D2H/H2D、逐层 Link、DMA 隔离和故障回收；
- 生产多 Service/P/D、etcd quorum/网络分区、Agent 同命和端口隔离故障矩阵；离线
  多进程、watch/lease、自隔离/重建和切主已通过 hard gate；
- 固定 profile 的阶梯 QPS、KV/credit 拐点、TTFT/TPOT/goodput、观测开销和长时 soak；
- Coding 流水线首次远端运行、受保护分支 required check 与告警平台接线。
