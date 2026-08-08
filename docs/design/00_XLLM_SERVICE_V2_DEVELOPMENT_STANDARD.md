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

# xLLM Service V2 代码开发与交付规范

## 1. 定位与效力

- 状态：强制执行；适用于 `xllm` 与 `xllm-service` 两个仓库的 V2 开发。
- 首个交付版本：**V2**。不开发、不发布、不验收独立的 V1 产品版本。
- 设计基线：01 定义系统边界，02 定义 V2 必须包含的基础协议能力，
  08/09 定义 V2 KV-aware、流控和执行模式，11 定义多 Provider 接入。
- 本规范定义代码、测试、开发文档和完成度门禁。设计与实现发生冲突时，
  先修正文档并完成评审，禁止用代码事实静默改写设计。

本文使用“必须”“禁止”的条目都是合入与交付门禁，不是建议。

## 2. 版本与交付口径

### 2.1 首版直接交付完整 V2

原 02 文档中标为 V1 的 Provider SPI、State Stream、原子准入、attempt、
deadline、fencing、资源回收和观测闭环，统一解释为 **V2 基础能力**。这些
能力可以作为内部开发门先实现，但不能形成独立 V1 版本或对外宣称已交付。

V2 首版只有在以下范围同时完成后才能交付：

1. 01 中 V2 适用的系统边界、状态归属和故障不变量；
2. 02 中全部基础协议能力与 G-2、G-1、G0-G4/M0 门禁；
3. 11 中统一 Provider Contract、能力矩阵和 fail-closed Adapter；
4. 08 中 V2-K0 至 V2-K2 的精确 HBM KVIndex、事件恢复和 KV/load 联合选择；
5. 09 中策略感知有界流控、优先级/租户公平，以及按能力选择
   `REMOTE_PD`、`LOCAL_PREFILL_DECODE`、`PREFILL_ONLY`；
6. V2 支持矩阵中的 CPU 可验证项全部通过，NPU 专项项有明确待验状态和
   后续验证计划。

V2.5 的共享 Store、V3 Placement 及之后能力不因首版直接 V2 而被提前纳入。

### 2.2 禁止伪 V2

- 禁止只完成原 V1 基础能力便把版本标记为 V2 完成。
- 禁止用空实现、固定返回值、未消费的 proto 字段或只打日志的状态机占位。
- 禁止新建 `v1` 分支、V1 发布包、V1 API 或 V1 专属兼容路径。
- 内部可以按依赖拆分开发门，但名称使用 `V2-B*`、`V2-K*`、`V2-L*` 等
  能力门，不把开发门包装成产品版本。

## 3. 代码风格：严格服从 xLLM

### 3.1 唯一风格来源

所有新增、修改和重构代码必须遵循 xLLM 的项目级规范：

1. `xllm/.agents/skills/code-review/references/custom-code-style.md`；
2. xLLM 根目录 `.clang-format`；
3. 项目级规范未覆盖时，才使用 Google C++/Python Style Guide。

`xllm-service` 不建立另一套命名或格式规则。两个仓库的 `.clang-format`
必须保持语义一致；当前基线为 Google 风格、2 空格缩进、80 列、指针左对齐、
禁止参数与实参自动装箱。C++ 改动必须通过目标 xLLM 基线使用的
`clang-format` 检查。

### 3.2 必须执行的项目规则

- C++ namespace/function/local variable/file 使用 `snake_case`，类和结构体使用
  `PascalCase`，成员变量使用尾下划线，常量使用 `kPascalCase`。
- 新头文件使用 `#pragma once`，include 使用项目根相对路径；删除重复和未使用
  include。
- 普通整数优先使用固定宽度类型；禁止 C 风格强转；简单基础类型禁止滥用
  `auto`。
- 单参数构造函数使用 `explicit`；非继承类使用 `final`；override 只写
  `override`；带成员函数的数据类型使用 `class` 而不是 `struct`。
- 所有控制流使用花括号；文件局部符号放入匿名 namespace；已知容量的 vector
  先 `reserve()`，优先 `emplace_back()`。
- 所有权默认使用 `std::unique_ptr`；只有确有共享所有权时使用
  `std::shared_ptr`；裸指针只表达清晰的非 owning 引用。
- Torch C++ 优先使用 `torch::` API；断言使用 `CHECK`；不可恢复错误使用
  `LOG(FATAL)`，不得另造异常处理风格。
- 禁止新增 `FLAGS_` 全局依赖；配置通过构造参数或明确的 config 对象注入。
- Python 函数必须有完整类型标注，诊断输出使用 xLLM 共享 logger，禁止用
  `print()` 代替日志。
- 所有新文件必须带创建年份正确的 xLLM Apache 2.0 版权头。

格式化只是最低门槛。格式正确但命名、所有权、错误处理或模块边界违背上述
规则的代码仍不得合入。

## 4. CPU 与 Torch CPU 测试规范

### 4.1 CPU-first 可验证性

除真实 NPU kernel、CANN/驱动交互和设备间 DMA 外，所有 V2 控制逻辑必须能在
CPU 环境编译和执行。Provider、Registry、State Stream、clock、transport、
allocator 和 Engine admission 的硬件边界必须可注入 fake、in-memory 或
loopback 实现，不能把可移植逻辑藏在 `USE_NPU` 分支中。

新增 tensor 逻辑必须提供 Torch CPU 路径。共享算法先在 `torch::Tensor` CPU
或 PyTorch CPU 上证明 shape、dtype、layout、数值和边界语义，再进入 NPU 专项
验证。禁止以“最终运行在 NPU”为理由跳过 CPU 测试。

### 4.2 功能覆盖要求

“CPU 测试覆盖全”按功能与状态语义验收，不以单一行覆盖率代替。每个新增或
修改的需求必须在开发文档中映射到测试，并至少覆盖：

- 正常成功路径；
- 非法输入、永久不可行和临时容量不足；
- 空输入、单元素、最大允许边界和越界；
- timeout、cancel、重复提交、RPC 结果不明和幂等重放；
- lifecycle、attempt、hold、reservation、transfer 的每个允许状态迁移；
- stale state、incarnation 变化、fencing、drain 和资源回收；
- 并发竞争、顺序变化以及失败后的资源账本不变量；
- capability/provider/mode 支持矩阵中每个声明支持或稳定拒绝的组合。

Torch CPU 测试还必须覆盖：

- dtype、shape、stride/contiguous、device 和输出所有权；
- 空 tensor、非连续 tensor、边界 shape 和不支持 dtype；
- 与明确 reference 实现的数值对比，并按 dtype 给出显式容差；
- 确定性要求、异常输入，以及需要 backward 时的梯度正确性；
- 与设备无关的序列化、分片、block/layout 和跨 rank 元数据计算。

硬件专属测试可以延后到 NPU，但必须在支持矩阵中标为 `NPU_PENDING`，并说明
无法由 CPU 证明的硬件不变量、所需环境和验收用例。`NPU_PENDING` 不得掩盖
本可在 CPU 上验证的逻辑。

### 4.3 测试层级与合入门禁

每项功能按适用范围提供：

1. 纯函数/数据结构单元测试；
2. 状态机、协议、序列化和兼容性测试；
3. fake Engine/Provider 的组件测试；
4. loopback 多 Service/P/D 的 CPU 集成测试；
5. Torch CPU reference/parity 测试；
6. 故障注入、资源泄漏和确定性回归测试。

合入前必须满足：

- 两仓受影响 CPU 目标成功编译和动态链接；
- 所有新增测试及受影响既有测试通过，无静默 skip；
- 测试失败、flaky 或缺失必须记录为未完成，禁止把失败基线算作通过；
- 测试可以离线复现，不依赖真实 etcd、在线模型服务或 NPU；
- 修复缺陷时先增加能够稳定复现缺陷的测试，再提交修复。

## 5. 开发文档与完成度

每个可独立识别的 V2 功能必须和代码一起维护开发状态文档，统一放在
`docs/design/implementation/<feature>_STATUS.md`，模板见
[V2 功能开发状态文档](./implementation/README.md)。没有开发状态文档、文档与代码
不一致或测试映射缺失时，该功能视为未完成。

状态文档至少包含：

```text
功能名称与 owner
关联设计章节和 Requirement ID
本次支持范围 / 明确不支持范围
代码入口、核心接口和跨仓依赖
Provider / mode / model / dtype / topology 支持矩阵
状态：PLANNED | IN_PROGRESS | PARTIAL | CPU_VERIFIED |
      NPU_PENDING | VERIFIED | BLOCKED
需求 -> CPU test -> Torch CPU test -> NPU test 的追踪表
已知缺口、风险、兼容与回滚方式
性能/容量基线和观测指标
最近验证的 commit、命令、环境和结果
下一步及达到 VERIFIED 仍缺少的证据
```

状态定义：

- `PARTIAL`：只有部分路径或门禁完成，不能对外宣称支持。
- `CPU_VERIFIED`：全部 CPU/Torch CPU 需求已通过，硬件专项尚未完成。
- `NPU_PENDING`：只剩文档列明的硬件专项证据，不包含可移植逻辑欠账。
- `VERIFIED`：设计范围、CPU、NPU、故障和文档门禁全部完成。

每次代码变更必须同步更新支持矩阵、测试结果和剩余缺口。禁止使用“基本支持”
“大致完成”等不可验收措辞。

## 6. 结构清晰与去冗余

- 一个状态、协议字段和资源账本只允许一个权威实现；Adapter 只做 Provider
  适配，不能复制 Scheduler、admission 或生命周期状态机。
- 公共领域模型放在明确的公共模块。禁止在 `xllm`、`xllm-service` 及
  `third_party/xllm` 三处复制同一 proto、枚举、错误码或序列化逻辑。
- 跨仓协议优先单一 proto 来源；暂时不能共源时必须有 descriptor compatibility
  CI、双向 golden wire test 和 reserved tag。
- 函数和类保持单一职责；解析、决策、副作用、状态持久化和传输边界应可分别
  测试。复杂分支优先拆成具名领域操作，不堆叠条件和布尔开关。
- 删除被替代的旧路径、死 RPC、无消费者字段和重复 helper。兼容路径必须有
  明确 owner、退出条件和测试，不能无限期与新路径双写。
- 不为未来版本预建抽象。只有当前 V2 至少两个真实调用方需要共享语义时才提取
  公共层；否则保持最小、直接和可测试实现。
- 性能优化不能复制 correctness 路径。规则 fallback、预测模型和 KV-aware
  score 必须复用同一过滤、能力门禁和 Engine 硬准入。

## 7. V2 Definition of Done

一个功能只有同时满足以下条件才可标为完成：

1. 设计章节、协议、不变量和明确不做项已确认；
2. 实现符合 xLLM 风格，结构清晰，无重复实现和临时占位；
3. 全部可移植逻辑已由 CPU/Torch CPU 测试覆盖并通过；
4. 跨仓 wire、错误码、状态和兼容性有机械测试；
5. 开发状态文档已更新，支持范围、缺口和验证证据真实可复现；
6. 资源释放、失败收敛、并发和回滚路径已验证；
7. NPU 专项已通过，或在开发阶段准确标记 `NPU_PENDING`；
8. 没有新增测试失败、未解释 warning、flaky 或业务源码脏改动。

整个首发版本只有在 02/08/09/11 的 V2 范围全部达到上述门禁后，才能称为
“V2 已交付”。
