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

# V2 功能开发状态文档

本目录记录 V2 功能的真实支持范围和完善情况。每项功能使用一个
`<feature>_STATUS.md`，并遵循
[V2 代码开发与交付规范](../00_XLLM_SERVICE_V2_DEVELOPMENT_STANDARD.md)第 5 节。

状态文档使用以下模板：

```markdown
# <功能名称>

## 基本信息

- Owner：
- 状态：PLANNED | IN_PROGRESS | PARTIAL | CPU_VERIFIED | NPU_PENDING |
  VERIFIED | BLOCKED
- 关联设计/Requirement ID：
- 最近验证 commit、环境和日期：

## 支持范围

| Provider | Mode | Model/Profile | 支持状态 | 限制与证据 |
| --- | --- | --- | --- | --- |

## 实现

- 代码入口与核心接口：
- 状态、资源和错误码权威位置：
- 跨仓协议与依赖：
- 明确不支持范围：

## 需求与测试追踪

| Requirement ID | CPU test | Torch CPU test | NPU test | 结果 |
| --- | --- | --- | --- | --- |

## 完善情况

- 已完成：
- 已知缺口/风险：
- 回滚与兼容：
- 性能、容量和观测证据：
- 达到 VERIFIED 仍需完成：
```

没有对应测试证据的支持项不得标记 `CPU_VERIFIED` 或 `VERIFIED`；文档中的
`PARTIAL`、`NPU_PENDING` 和 `BLOCKED` 必须列出具体缺口及关闭条件。
