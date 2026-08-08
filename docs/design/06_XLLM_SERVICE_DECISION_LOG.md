# xLLM Service 设计决策记录

开发与版本要求以[V2 代码开发与交付规范](./00_XLLM_SERVICE_V2_DEVELOPMENT_STANDARD.md)为准，系统边界以[总体架构](./01_XLLM_SERVICE_ARCHITECTURE_DESIGN.md)为准；02 是 V2 基础协议，08/09/11 分别补齐 V2 路由、流控/执行模式和多 Provider 门禁。

## D1：按请求调度、执行拓扑和 KV 内存层三轴演进

先在现有逐请求单 P/D 选择之上交付单域单模型的候选多选、延迟 D 绑定和有界重试，再并行演进三条轴：策略感知流控与公平性、从单域到跨域的执行拓扑、从本地 HBM 到共享 Store 的 KV 内存层。每个阶段都迭代同一选择/成本模型，不为后续能力预建请求事务。V2.5 的跨请求 KV 是主路线，不再作为编号外扩展。

## D2：路由快环与放置慢环分离

Service request router 只选择 READY Engine。V3 placement controller 使用标准 leader election 管理 load/warmup/drain、角色和副本目标，不进入请求关键路径。

## D3：Service 对请求只持有进程内状态

请求不写 Coordination Store。Service 非计划崩溃时，由该副本协调的在飞请求明确失败，客户端连接收到错误或断开后可按 API 契约重试；计划事件使用 LB 摘流和 drain。重新引入跨副本接管的条件是其可量化业务损失持续超过预先固定阈值。

## D4：唯一性是 Service/D 本地不变量

Service 只转发当前 attempt 的输出；D 按 `request_uid + attempt_seq` 本地去重并保留短期 tombstone。不使用远端 Commit CAS。内部重复计算按浪费 device time 管理，不升级为分布式事务。

## D5：reservation 生命周期由 D 本地决定

P 在 scheduler admission 时按真实 Prefill/传输上界请求 reservation TTL，D 校验本地 min/max 策略、返回实际 TTL，并以本地 monotonic clock 判断。可靠逐层 PUSH 使用短 transfer-start TTL 和长 reservation TTL，其他模式只用长 TTL；V1 远程路径中同一请求最多一个未证明 reservation，V2 跨模式扩展见 D32。没有 Renew、ticket、Capability Issuer 或跨机 deadline。过期后先确保传输终止，再释放资源。

## D6：不做 Decode 状态迁移

首 token 前允许当前 Service 换 P/D 重试；首 token 后的 D 或 Service 故障中断请求。客户端可显式重试，不透明迁移 Decode 采样状态。

## D7：Store 不替代正常 P→D 直传

Mooncake Store 用于 V2.5 的 Prefix/跨请求 KV 内存层。基础 handoff 继续直接传输；Store 故障不得影响普通请求。D 侧生成 KV 写穿共享层是命名交付项，但不包含 Decode checkpoint，也不允许 session manifest 代替客户端完整历史或上层业务状态。

## D8：Service 保留在输出路径，但不持久化结果流

普通 P/D 请求必须获得 D 的幂等 FirstGeneration ACK 后才向客户端暴露首 token；`PREFILL_ONLY` 不创建 D reservation。ACK 是一次 P→D 点对点 handoff 确认，不写共享 Store。P 的首 token 和 D 的后续 token 通过 Generations 直达发起 Service，并使用进程内 `output_event_seq` 做有界重排。V1 改为有界异步并在订阅方失效时终止对应请求；不建设 Journal、durable cursor、D replay 或 parser checkpoint。

## D9：软选择、硬准入

Registry/State Stream、调度规则和性能预测器只决定候选顺序。P/D Engine 本地 allocator 是容量唯一事实；并发冲突通过稳定 reason、负缓存和有界重选解决。现有单对路径与动态池共用同一 allocator、状态和资源账本，只允许候选生成和绑定时机不同。

## D10：规则算法永久保留

性能建模从 V1 开始逐步扩展到全集群；每个性能预测器版本必须有 timeout、OOD 检测和 M0 fallback。性能预测器不能进入 Engine 原子准入临界区，也不承担正确性。

## D11：V1 没有 Engine manager

所有 Service 可以向任意满足请求模型、角色、profile、链路和协议能力要求的 READY Engine 下发请求。Engine drain/换版/角色切换由 V3 placement leader 管理，不能以 per-Engine manager lease 形式进入 V1。

## D12：只为 DMA 不确定性保留 quarantine

逐 task cancel 后必须轮询到终态；无法证明设备工作停止时 quarantine/retire buffer generation。quarantine 不用于 Commit、owner 或其他逻辑故障。

## D13：V1 交付内置 State Stream

首个生产版本停止使用 etcd 扇出高频负载。Engine heartbeat 继续进入 master，由 xllm-service 内置 State Stream 向所有副本发布版本化事件和周期性全量快照；etcd 只保留 Registry/lease。State Stream 是代码模块，不是独立部署服务。

## D14：V1 动态池不执行 MIX 角色翻转

现有 `flip_prefill_to_decode/flip_decode_to_prefill` 只改各 Service 本地视图，无法保证多副本一致。V1 动态池仅接收固定 PREFILL/DECODE 角色；角色变化留给 V3 placement，先 drain，再以新 incarnation 注册。

## D15：State Stream 按 master incarnation fencing

master lease value 和 StateBatch 都携带唯一 incarnation。订阅方只接受 Registry 当前 incarnation；旧 master 在 keepalive 丢失时停止发送，迟到 batch 被 fencing 丢弃。正常观测下所有路由策略与 fallback 共用同一 `IsSchedulable`；单个 Engine heartbeat 或 state 超 hard TTL 时停止向其分配新请求，即使 Registry lease 仍存活。

## D16：观测失明不改变 Engine 成员身份

Registry lease/incarnation 是成员身份依据，State Stream freshness 是路由依据。Registry 正常而 State Stream 失明时，先在 `state_blind_grace` 内使用最后良好状态，之后只使用有近期直接 RPC/探活成功证据的 Engine；Registry 失明只允许较短 `registry_blind_grace`。没有可信兼容容量时停止新准入并退出 LB READY，但任何软状态超时都不得批量触发 `SUSPECT -> deregister`、P/D unlink 或在飞请求清理。破坏性删除只由 Registry lease 失效、incarnation 替换或明确的部署生命周期操作触发。

## D17：State Stream 复用现有 Service 地址与 key 布局

现有 `service_name` 固定为 `ip:rpc_port`，继续作为 member key/value 和 `PushEngineState` 地址，避免破坏 xLLM Engine 对纯地址 value 的解析。V1 不迁移 master 选举 key；统一成员枚举必须按完整 key 排除 `XLLM:SERVICE:MASTER` 并按地址去重。新副本加载 Registry 并应用当前 master 的 FULL 后才允许 `accepting_new_requests`。

## D18：就绪与 listener 生命周期分离

HTTP/RPC listener 随进程存活，不再用 `has_available_instances()` 停启。`/livez` 表示存活，`/readyz` 与 request handler 共同读取 `accepting_new_requests`；Registry 短宽限耗尽、State Stream 失明且没有近期直接成功候选或 drain 时退出 LB READY，在飞请求继续完成。瞬时 queue/KV/credit 不足保持 READY 并返回稳定容量拒绝，不能触发全体 Service 同步摘流。master key 变化本身不触发降级，观测模式只由实际可见性和新鲜度决定，并使用同量纲双阈值与时间迟滞防抖。

## D19：集群 KV 索引是可丢失的路由提示

V2 复用 `GlobalKVCacheMgr`、State Stream 和统一选择接口交付集群级 KV-aware Router，不新增独立服务，不把 block 索引写入 etcd。Engine 事件驱动的索引只用于估算 P 侧 Prefill 复用与 D 侧分配/传输节省；Engine admission 仍是真实命中和容量权威。事件缺口、索引过期、超预算或收益不足时关闭 KV credit 并回退负载模型，不能拒绝本可正常执行的请求。

## D20：D 绑定时机受逐层 PUSH 三角约束

KV-aware 选 P、尽可能晚绑定 D、逐层 PUSH 与 Prefill 重叠三者不能同时最大化；逐层 PUSH 要求 Prefill 第一层开始前已有 D 目标 buffer。V1 选择 KV/负载感知的 P，并在 P scheduler 即将 admission 时绑定 D，这是保留逐层重叠前提下的最晚绑定点。只有放弃逐层重叠或引入另一种拉取协议时才重新评估更晚绑定，不因其他系统在完整 Prefill 后选 D 就直接改变当前路径。

## D21：V1 不做策略队列，V2 交付有界 flow control

V1 以稳定 `CAPACITY_EXHAUSTED`/`SLO_UNSATISFIABLE` 和客户端重试控制首版范围，不建设 Service 侧策略队列，M2 也不能隐含该能力。该选择会在过载时增加可见拒绝，且不能完整表达租户公平和优先级调度，因此不是最终目标。V2 交付有界、work-conserving 的优先级 band、租户 flow、flow 内 FCFS/EDF 和整池饱和门控；队列仍不持久化、不跨副本接管。V1 上线必须固定拒绝率产品门禁，若拒绝率或 `plan_failure_after_queue_wait_rate` 提前越界，可以把最小 BEST_EFFORT 队列作为 V1.x 独立变更评审，不能静默扩大 V1。

## D22：Engine 注册角色与逐请求 Prefill 模式分离

PREFILL/DECODE 的持久角色变更只能由 V3 慢环依次执行 drain、revoke 和新 incarnation 注册；V1 禁止多 Service 各自修改 MIX 本地视图。某个请求是否在 D 本地做 chunked Prefill 是 V2 快环候选规则，不改变 Engine 注册角色。只有 capability、混批隔离和 SLO 门禁通过时才启用；否则继续走远程 P。两者不得再用“固定角色”一并否决。

## D23：默认不同步跨 Service pending work

普通 Service 不交换请求 ownership 或请求级 peer RPC。每个副本只记录自己刚下发、尚未进入 State Stream 的 pending work，并以 least-load/top-prefix shortlist、同分随机化、保守 guard 和 Engine 原子 admission 缓解陈旧视图。代价是多副本并发下会低估其他副本的新增负载；若 `AddNewRequests` 冲突率在扩大 guard 后仍持续越过门禁，才重新评估只同步易失 aggregate/active-block prediction，不引入请求接管。

## D24：V1–V3 复用现有 etcd/brpc，不以 GIE 为内部协议

私有超节点栈、既有 Engine 对纯 `ip:port` Registry value 的依赖和 brpc 数据通路使自建 Registry/State Stream 在当前阶段成本最低。代价是无法直接复用 Gateway API Inference Extension 的 scorer、flow control 和 autoscaler 生态，V3 placement 需要自行实现。对外可增加 GIE Endpoint Picker adapter；当跨团队接入成本或生态复用收益超过内部协议适配成本时重新评估，但不要求改写 Engine 内部协议。

## D25：Engine lease 失效必须配套 self-fencing

Registry lease 失效证明旧 incarnation 不再是集群成员，不天然证明进程、DMA 和设备工作已物理停止。Engine 失去 lease ownership 时必须本地进入不可逆 `FENCED`，停止新 admission、输出和新 transfer，并通过 cancel/TTL/drain 收敛；网络恢复后使用新 incarnation。Service/P/D 以 incarnation 和 attempt 拒绝迟到结果。只有 self-fencing、部署终止或 Query/TTL 提供终态证明时才提前释放逻辑 reservation；已启动 DMA 仍按 transfer 终态或 quarantine 处理。

## D26：默认 full-history 与 strict-session 分开

默认请求携带完整历史，session/version/fingerprint 只是 KV 命中提示；不匹配、KV 缺失或 Store 故障均按 cache miss 完整 Prefill，不返回 session 冲突。只有显式 strict-session API 才使用单 writer、writer fence、manifest CAS 和稳定冲突错误。两种语义不能共用一个默认失败规则，manifest 也不能成为对话文本的唯一真相。

## D27：全系统单一 block hash 前像，多模态逐 block、存储 layout 只绑对象键

Router、Engine 本地 Prefix cache、KV 事件和 V2.5 Store 对象共用 08 §4 定义的同一个链式 block hash，任何一方不得私自增删前像输入。两条容易反复的具体规则：多模态摘要按 block 参与（放进 namespace 会使图像不同的请求连共享文本前缀也无法互认，链式结构本就能让分叉点之后自然分开）；`storage_kv_layout_digest` 绑定在 Store 对象键路径上而不进入哈希（折进哈希会让 Store 事件与 Router 计算的 hash 落入不同键空间，KVIndex 静默永远 miss，而 layout 隔离由对象键同样能够保证）。`positions` 不进前像，块序号由链深度隐含。代价是新增任何影响 KV 内容的因素都必须改 08 §4 并同步 bump `hash_version`，不能就地在某一侧扩展；这正是要付的代价。

## D28：Store Put 必须转移 backing ownership 或持有 read pin

丢弃 Store credit 只关闭路由收益，不证明 Put/DMA 终态。V2.5-S1 默认先复制到有界 snapshot/staging pool，本地 copy 完成后原 KV block 才能释放；零拷贝优化必须创建 StorePutHandle 并持有 live-block read pin，直到 Put/Query terminal。结果不明复用 transfer 的 cancel、poll、drain、quarantine 阶梯。handle 一旦创建不能因队列超时或 credit 丢弃而 erase；snapshot/pin/handle/quarantine 都有硬容量上限。02 §6.5 只保留通用 backing-memory 释放不变量，不把 Store connector 反向加入 V1。

## D29：公共计划显式区分聚合与 P/D，xLLM 本地 Prefill 不建第二套账本

Provider-neutral `ExecutionPlan` 允许 `AGGREGATED`、`REMOTE_PD`、`LOCAL_PREFILL_DECODE` 和 `PREFILL_ONLY`；每种 Provider 只发布已通过门禁的子集。`AGGREGATED` 表示一个 Runtime 实例完成整请求，是 vLLM-Ascend V1 首个模式，不伪装成 xLLM D。本地模式由 Service 直接提交显式支持 capability 的 xLLM D，D 在同一 allocator/scheduler 原子检查 mixed Prefill+Decode、KV、credit、slot 和 interference guard；不创建远程 reservation/BeginTransfer/FirstGeneration，但共用 request key、Query/cancel、TTL、tombstone 和输出 seq。选择成本必须加入对共驻 Decode 的 TPOT/SLO 外部性，不能只优化当前请求 TTFT。完整协议见 09/11。

## D30：V2 队列容量同时受内存、崩溃暴露和 drain deadline 约束

Service 队列仍不持久化、不跨副本接管，因此 `QUEUED + DISPATCHED` 是单 Service kill -9 的最大失败集合。队列 request/token/byte/tenant 上限必须同时满足 Service 内存和 `service_crash_request_budget`，上线同时报告客户端最终重试成功率。计划 drain 固定选择 `COMPLETE_QUEUED` 或 `RETRY_UNDISPATCHED`：前者的 deadline 覆盖最大排队等待和已提交请求剩余时限；后者只有在 Gateway/客户端已验证首 token 前重试时才可用，不能单独宣称无损。完整状态机和公式见 09。

## D31：V2.5 与 V3 并行、互不构成可用性依赖

阶段号不是全序依赖。V2.5 共享 KV 层与 V3 Placement 可以并行开发和独立上线；V3 在 Store 不可用时必须使用完整 `cache_loss_cost`。只有 Put/Query 和副本门禁已确认的 Prefix 对象才能降低缩容 cache loss，in-flight Put 或软索引不能抵扣。V2.5 能改善 V3 的缩容目标函数，但不是 autoscale、load/warmup/drain 的硬前置。

## D32：跨模式只允许一个 outcome 不明的执行资源持有

`max_unresolved_execution_holds_per_request=1` 是不可配置的跨阶段协议常量，同时覆盖 `REMOTE_D_RESERVATION`、`LOCAL_DECODE_SUBMISSION` 与 `AGGREGATED_EXECUTION`。RPC outcome 不明时，Query/cancel、TTL、self-fencing、进程终止或可执行的硬时间证明给出收敛证据前不得创建替代 attempt，包含跨三种模式切换。普通 P submission 不占用完整生成或 Decode KV/credit/slot，不计入该常量。已证明进入 `GenerationCommit/DECODING` 的旧 attempt 仍可在先 cancel 后发生有界重叠，但重叠必须计入 retry/device-time 浪费预算；已创建但尚未 Commit 的资源必须先 cancel 并证明 terminal。Service 在发送远程 P plan、本地 D submission 或聚合 Submit 前必须先在 `RequestContext` 上以 CAS 安装 hold；聚合路径的 holder 是已提交 Agent/Engine，Submit 结果不明时 `ABSENT` 不构成证明，未知 Cancel 必须安装 `CANCELLED_BEFORE_CREATE` fence。统一清理表与各模式时间边界见 02 §5.1/§5.3/§5.5。

## D33：seq=0 统一受 mode-specific GenerationCommit 屏障

`GenerationCommit` 是 Engine/Agent 本地原子、幂等的“允许交付 seq=0”屏障，不等于客户端已收到 token。屏障通过可不明 RPC 时必须可 Query 证明；`PREFILL_ONLY` 不强造 D Query，但同 key P submission 仍必须幂等。`AGGREGATED` 由 Provider Agent 原子接受完整请求并安装唯一输出 attempt；`REMOTE_PD` 仍必须等 P 获得 FirstGeneration ACK，仅有 D reservation 成功不够；`LOCAL_PREFILL_DECODE` 在同一临界区授予完整 mixed 资源、安装幂等 submission 并转入 `LOCAL_GENERATION_COMMITTED`；`PREFILL_ONLY` 由 P 原子准入无后续 D 的完整执行。这一抽象不弱化 V1 的 handoff 安全，也不把 V2 实现反向加入 V1 范围。

## D34：V2 队列复用 V1 观测模式，saturation 输出三态

V2 不为队列另建健康状态。saturation detector 只输出 `AVAILABLE | SATURATED | UNKNOWN`，陈旧或缺失状态不得强制当作饱和或可用。`OBSERVATION_STATE_BLIND` 按最后良好状态、直接成功证据和有界 probe 降级；`OBSERVATION_REGISTRY_BLIND` 宽限耗尽后停止入队/dispatch 并退出 LB READY。队列满单独出现时仍保持 READY 并快速容量拒绝；观测边界要求退出 READY 时，已 dispatch 请求继续、未 dispatch 队列请求返回稳定可重试错误。`UNKNOWN` 下的 dispatch rate 只能来自 probe 成功率的单侧置信下界并随样本衰减到 0；不存在非零配置 floor。rate 为 0 但按当前观测模式仍有 eligible 候选时，仅允许有界 BEST_EFFORT immediate probe；已有队列按原 band/tenant/flow 顺序优先于新到达请求，STRICT 稳定拒绝。没有 eligible 候选则退出 READY，不能用用户请求恢复健康状态。STATE_BLIND 宽限后 eligible 要求近期直接证据，REGISTRY_BLIND 宽限内仍按缓存成员规则。

## D35：本地 Prefill 与 Store 写穿共用 D 干扰预算

V2-L1 本地 Prefill 和 V2.5-S1 `COPY_ON_PUT`/write-back 都会占用 D 的内存带宽、scheduler 与 Decode TPOT headroom。两者必须在 D 本地共用原子 `DInterferenceBudget`，Store copy 的 bytes/time、带宽、staging headroom 和 TPOT 影响进入写穿策略；竞争时优先已准入 Decode，新 Put 放弃 credit，新本地 Prefill 回退远程路径。`PIN_ON_PUT` 的 pin bytes 和 Store/RDMA 读带宽也扣减同一预算，零拷贝不是例外。两路径只使用由当前 Decode 最严 SLO/profile 默认值派生的 `d_decode_tpot_guard(snapshot)`；Store/local share 是分类上限，当前快照下的绝对预测才是最终硬门禁。marginal delta 包含非线性交互，扣减和判定必须原子完成，且 `store_copy_tpot_share <= local_prefill_tpot_share <= 1`。上线必须做 `off/off`、`local-only`、`store-only`、`both-on` 四组同负载对照，显式测量非线性交互；`both-on` 未通过时两能力在同一 bucket 互斥或共同降额。

## D36：未知 holder 的取消必须留痕，时间证明只用本地硬 duration

远程 plan 的 Decode hold 以候选 incarnation 集合作为安全作用域；P 在回填前失联时，Service 只对该有界集合扇出。未知 key 的 Query `ABSENT` 不构成终态；Cancel 必须在 D 本地原子安装 `CANCELLED_BEFORE_CREATE` 否定 fence 后才 ACK，迟到 AddNewRequests 稳定拒绝。fence 使用 D 本地 monotonic TTL 和独立容量池；池满返回稳定 reason，D 进入 recovery-fence pressure 并停止新 Decode admission，不能伪造成功或挤占 outcome tombstone。fence TTL 覆盖 P submission RPC、P queue、D admission、AddNewRequests RPC hard lifetime 和 guard 的完整旧 plan 寿命。

Service 的时间兜底由 P submission RPC、P queue、D admission、AddNewRequests RPC hard lifetime、最大 reservation TTL 和扫描 jitter 的上界相加，只使用各组件本地 monotonic duration。若 transport/server handler 不能执行 RPC hard lifetime，就禁用时间证明，安全优先于重试活性。请求可以先失败，但有界 cleanup record 必须继续；它不保存请求内容、不恢复请求。Service 在 dispatch 前为每个 hold 预留 cleanup capacity token，避免请求结束时才发现表满；容量按受影响 attempt burst 而不是 P 故障事件数核算。Service 崩溃后由 D fence/TTL 自行收敛。

## D37：观测身份不参与执行正确性，O0 先于性能模型

Gateway/Service/P/D 透传 `global_request_id + trace_id` 做事件关联，资源协议仍只使用 `request_uid + attempt_seq + incarnation_id`。组件身份统一使用 Registry incarnation，每个 Admission attempt 恰好一个结构化终态。阶段 duration 由 owner 使用本地 monotonic clock 记录，异步事件允许丢失但必须计数；TPOT=0、负 ITL、跨时钟域相减或关联不完整的数据不得训练 M1/M2。观测系统不是请求状态存储，故障不能阻塞执行。

## D38：Admission 区分永久不可行与临时不足

现网 Engine 已能区分永久不可行与临时不足；V1 将其暴露为结构化三态结果和 per-rank block/credit 证据，不重复建设判断分支。若声明的开发基线尚无该分支，必须先取得产生日志的实际 build，或按线上既有语义移植，不能另造一套分类。新增能力是 Service 在下发前按真实 D profile 预判永久可行性；动态 headroom 不足才允许换 D 或有界退避。单请求预算失败必须原子回滚且只影响该请求。不同请求的 AddNewRequests 可按目标 D 异步/批量，但同请求候选仍受顺序和 Decode hold 约束。

## D39：基础 Decode Admission 保护已准入请求的 TPOT

线上证据表明普通 Decode 并发本身存在非线性衰减，因此 TPOT guard 不是本地 Prefill/Store 的专属能力。Service 使用完整模型做软选择；D 只执行由 CapacityProfile 编译、结合本地快照的版本化保守 guard，不在 allocator 临界区调用 M1/M2。BEST_EFFORT 可承担自身 SLO 风险，但不得突破已准入 Decode 的最严格 guard；V2 的 `DInterferenceBudget` 只在此基础上增加可选干扰账本。

## D40：lease 抖动与 incarnation 复活分开处理

Engine 使用 `OWNED -> OWNERSHIP_UNCERTAIN -> FENCED`。keepalive 暂时失败只停止新工作，不触发全池重载；UNCERTAIN 的本地 deadline 必须早于外部最早可能判定 lease 失效的时刻。键消失/被覆盖、deadline 到期或发现新 incarnation 时不可逆 FENCED；恢复只能使用新 incarnation，禁止复用旧注册 value。

## D41：可调度性属于 P-D pair，不只属于两个 Engine

`InstanceMgr` 以两端 incarnation 维护 `PENDING | READY | DEGRADED` LinkState，并周期对账 Registry 兼容 pair。单个 link 失败只降级该 pair，不能回滚其他健康 link；动态策略和 fallback 都必须同时检查两端 `IsSchedulable` 与 pair READY。

## D42：业务 deadline 必须在 Engine 本地终止

跨进程只传剩余 duration，接收方转换为本地 monotonic deadline。Service Watchdog/Cancel 是快速路径，P/D 本地检查是容量兜底；到期进入 `DEADLINE_EXCEEDED` 正常终态，停止新调度并释放资源，不能让客户端超时后的 Decode 继续占用 KV 和带宽。

## D43：跨仓协议兼容必须机械证明

优先使用单一 proto 来源；暂不能共源时，ProtocolContract 必须同时交付 descriptor compatibility CI、双向 golden wire test 和两侧 reserved tag。文档约定不能替代字段号冲突检查。

## D44：多引擎统一语义，不强制统一 wire

xLLM Service 依赖 ProviderDescriptor、Capability、EngineState、ExecutionPlan 和 attempt/fencing 语义；xLLM Native Adapter 继续使用 brpc/protobuf，vLLM-Ascend Agent 继续使用 OpenAI HTTP/SSE 与 vLLM 插件接口。禁止把 `backend_type` 特例扩散到 Scheduler，也禁止为追求表面统一 fork Runtime 内部协议。详细规格见 11。

## D45：能力不对称通过门禁表达，不伪造功能对齐

V1 的共同底座同时纳管 xLLM Native 和 vLLM-Ascend，但首版模式不要求对称：xLLM Native 交付严格 layerwise PUSH P/D，vLLM-Ascend 交付严格聚合模式。Provider 缺少 reservation、attempt query、Engine-local deadline、self-fencing 或兼容证明时，只能排除相应 STRICT mode 或留在显式 BEST_EFFORT 兼容 bucket，调度器不能用预测补偿硬语义。

## D46：Provider profile 是不可变兼容与建模边界

Registry 身份必须包含 Runtime/插件/硬件 Runtime 版本、模型/tokenizer/template、量化、拓扑、KV layout/Connector、scheduler 和 capability digest。字段变化创建新 profile/incarnation；CapacityProfile、trace、online residual 和 SLO 门禁按 `provider_id + provider_version + profile_digest + mode` 隔离，xLLM 与 vLLM-Ascend 数据不得混池。

## D47：跨 Provider P/D 默认 fail closed

P/D 兼容键包含两端 Provider/profile、模型 revision、KV layout、Connector protocol/version、transfer mode 和 topology transform。相同 Mooncake 名称不构成兼容证明。首版只允许各 Provider 内部经过测试的 pair；xLLM ↔ vLLM-Ascend P/D 只有专门转换和故障套件通过后才能增加。

## D48：vLLM-Ascend 必须由同机 Agent 约束数据入口和失效域

当前 sidecar lease 只证明 sidecar 存活，不能 fencing 原始 vLLM 端口。进入 STRICT 池时，Registry 地址必须指向 Provider Agent；原始端口仅允许 Agent/本机访问，Agent 与受控 vLLM 还必须属于同一重启/终止单元。`agent_fate_bound` 必须早于 Service 可创建替代 attempt 的最早时刻；只杀 Agent 后 vLLM 仍能接单的部署不得发布 `SELF_FENCING`。当前上游 `check_health()` 吞掉深层 NPU 异常，Agent 只能独立验证 `npu-smi` 或等待上游修复，在此之前不得发布 `DEEP_HEALTH`。

## D49：Provider 状态保留资源标签，缺失不是零

State Stream 保留每个 DP/rank/cache group 的原始资源语义。counter 可按定义求和；TP rank headroom 取最小值；DP KV ratio 保留逐 DP 或取选定/最坏值，禁止求和；延迟合并 histogram bucket delta 后求分位数。缺失值为 `UNKNOWN` 并触发能力降级，不能补 0 或把 interval average 命名为 max。

## D50：STRICT mode 必须有机械可判定的 capability 矩阵

Compatibility Resolver 固定检查 `required_capabilities(mode) ⊆ published_capabilities`。`ATTEMPT_QUERY/CANCEL_FENCE/ENGINE_LOCAL_DEADLINE/SELF_FENCING/DRAIN` 对全部当前 STRICT 模式必需，`NATIVE_RESERVATION` 仅远程 P/D 必需；可选状态能力缺失时写 `UNKNOWN` 并退出相关硬预测。未知 capability、未分类 mode 和提交屏障尚未定义的 `EPD` 一律不可注册，不能把要求散落在 Adapter 分支中。

## D51：执行 KV 布局与 Store 字节布局使用两个摘要

`kv_layout_digest` 覆盖执行/传输侧 cache dtype、block size、cache group、head/shard 映射和 Connector wire version，用于 P/D 与 topology transform 兼容；`storage_kv_layout_digest` 覆盖实际落盘字节布局与 Store serialization，用于对象键。两者都不进入 08 §4 的 block hash 前像。只有落盘布局完全由执行布局决定时，Provider 才可显式声明 `storage_kv_layout_digest = H(kv_layout_digest || store_serialization_version)`，不得隐含相等；只改前者必须拒绝 P/D，只改后者只改变 Store key，Router hash 不变。

## D52：绑定顺序由 Provider profile 声明

`ExecutionPlan` 使用有序角色项和 `selection_order=SINGLE|P_FIRST|D_FIRST`，由 `(provider_id, mode, transfer_mode)` 的 Descriptor 声明。xLLM layerwise PUSH 是 `P_FIRST`；vLLM-Ascend layerwise PUSH 是 `D_FIRST`，实际 P 由 Adapter 回填，且 Service 会失去 P 侧 KV-aware 选择。V1 仅执行前者和 vLLM-Ascend `SINGLE` 聚合模式，但 schema 必须能往返表达并稳定拒绝未开放的 `D_FIRST` profile。

## D53：成员资格丧失后立即停止新计划

只有带 revision 的权威 Registry DELETE/revoke 或 incarnation 变化能把实例置为 `MEMBERSHIP_LOST`；下一次选择立即排除，宽限仅收敛在飞请求。heartbeat、地址探活和 watch 恢复不是成员证明，不能恢复旧 incarnation；重新加入必须使用新 incarnation。watch 断连、list 歧义或探活失败走 `REGISTRY_BLIND`/状态降级，不能伪造成 DELETE。该决策与 D40 的 Engine self-fencing 配对，任一端缺失都不能形成 fencing 闭环。

## D54：首个产品版本直接交付 V2

不设置独立 V1 产品版本、发布包或先行上线里程碑。原 02 和既有决策中称为 V1 的 Provider SPI、State Stream、原子准入、attempt、deadline、fencing、资源回收和观测闭环全部重分类为 V2-B0 内部基础门；首发必须继续完成 08 的 V2 精确 HBM KV-aware、09 的有界流控/公平性/逐请求执行模式和 11 的多 Provider 门禁。只通过 V2-B0 不构成交付。该决策覆盖 D1、D13、D21 等条目中的旧版本/上线口径，但不改变其协议机制和正确性约束；V2.5 Store、V3 Placement 及后续阶段范围不变。

## D55：V2 实现采用 xLLM 风格和 CPU-first 完成度门禁

`xllm` 与 `xllm-service` 统一遵循 xLLM 项目级 custom code style 和 `.clang-format`，不维护第二套风格。除硬件专属 kernel、CANN/驱动和设备 DMA 外，全部可移植逻辑必须由 CPU 与 Torch CPU 测试覆盖；每个需求、状态迁移和支持矩阵项都要映射到可复现测试。每项功能随代码维护开发状态文档，准确列出支持范围、缺口、CPU/NPU 状态和验证证据。结构上禁止复制状态机、协议、资源账本和新旧双路径；未满足代码、测试、文档任一门禁的功能不得标记完成。完整规则见 00。

## D56：公共 Resolver 覆盖完整 V2 mode，而不是只覆盖基础远程路径

`required_capabilities(mode, transfer_mode)` 同时覆盖 `AGGREGATED`、
`REMOTE_PD`、`LOCAL_PREFILL_DECODE` 和 `PREFILL_ONLY`。本地 D 与 P-only
分别使用 `D_ONLY`、`P_ONLY` selection order；两者都在提交时绑定且不伪造
远程 reservation。`LOCAL_PREFILL_DECODE` 必须发布 mixed accounting 和
structured admission，`PREFILL_ONLY` 必须发布独立能力；二者与其他 STRICT
mode 一样要求 attempt、cancel fence、本地 deadline、self-fencing 和 drain。
未知 mode/transfer/capability 或未分类组合稳定 fail closed，能力规则只允许在
公共 Resolver 维护，禁止 Adapter 私有放宽。该决策补全 D50/D52 在原 V2-B0
范围内只列聚合和远程 P/D 的矩阵，不改变 D29/D32/D33 的资源与提交不变量。

## D57：STRICT 编码使用独立的组合 renderer digest

`ProviderDescriptor.model.renderer_digest` 是 tokenizer revision、template 与
影响渲染结果的配置共同形成的稳定摘要，不能用 `chat_template_digest` 或产品名
代替。`RequestCodec` 返回的 `EncodedRequest.renderer_digest` 必须在 STRICT 请求下
与 Descriptor 完全相等；模型 revision、请求 capability 和 API feature 也必须是
Descriptor 的子集。任一证据缺失或不相等都 fail closed。这样 Service 能机械证明
调度计数和 Provider 实际输入遵循同一渲染契约，而不是只记录一个不可校验字段。

## D58：`attempt_seq=0` 是合法首个 attempt，wire 必须保留 presence

`attempt_seq` 继续按 02 §3.1 从 0 单调递增。所有承载它的 proto3 消息使用
`optional uint64`：未出现表示协议字段缺失，出现且值为 0 表示合法首个 attempt。
校验器检查 presence 而不是 `value != 0`，避免把首个执行误判为非法，也避免把
旧发送方缺字段静默解释成首个 attempt。后续事件、资源键和输出 fencing 统一遵循
同一语义。

## D59：观测 producer 使用固定容量、非阻塞的本地事件 ring

请求执行线程在事件 schema 校验后只尝试一次获取 recorder 写锁；锁竞争立即丢弃并
计入 `dropped_contention`，容量耗尽立即丢弃并计入 `dropped_capacity`。ring 在构造时
按硬上限预分配，drain/exporter 可以等待锁，但 producer 不能等待 exporter，也不能
用动态队列把观测压力转成无界内存压力。身份字符串和 measurement boundary 在入队前
做固定长度门禁。丢事件只降低观测完整性，不改变执行结果；关联覆盖率和丢弃率不满足
门禁时，相关数据禁止训练 M1/M2。Admission RAII guard 对每次本地调用只生成一个终态
尝试，未显式结束时生成 `FAILED/MISSING_TERMINAL`；终态本身若因 ring 压力丢弃，仍由
上述丢弃计数显式暴露，不能阻塞或回滚请求执行。

## D60：请求关联使用单一不可变对象，旧 ID 只作兼容镜像

Service 在生成类 HTTP 请求入口构造一次 `RequestCorrelation`：有效的上游
`global_request_id/trace_id` 原样保留并标记来源，缺失或非法值由 Service 补齐；
W3C `traceparent` 只在结构、version、trace-id、parent-id 全部有效时提供 trace ID。
`request_uid` 始终由 Service 生成或由测试注入器提供经过校验的 UUIDv7，
`attempt_seq` 显式携带合法首值 0。长度、字符集、UUID version/variant 和来源 enum
均在边界 fail closed。

该对象从 Service 请求上下文复制到 Completion/Chat wire、xLLM `RequestParams`、
P 运行时 Request 和 P→D `DisaggRequest`，进入运行时后只提供 const 访问。
现有 `service_request_id/service_req_id` 不再生成第二个 ID，只作为
`correlation.request_uid` 的兼容镜像；xLLM API 与 D 接收边界在 correlation 出现时
强制二者相等。旧直连客户端未携带 correlation 时仍走显式 legacy 路径，P 不向 D
制造“存在但为空”的 correlation。vLLM 聚合路径通过 HTTP header 透传同一对象。
模型枚举等非生成控制请求不创建空身份。观测 ID 仍不参与资源正确性，执行键仍按
D37 使用 `request_uid + attempt_seq + incarnation_id`。
