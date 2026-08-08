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

# xLLM Service Coding Agent Instructions

Before editing or reviewing code in this repository, read and follow:

1. `docs/design/00_XLLM_SERVICE_V2_DEVELOPMENT_STANDARD.md`;
2. `third_party/xllm/.agents/skills/code-review/references/custom-code-style.md`;
3. the repository root `.clang-format`.

The first product delivery is the complete V2 scope. There is no independent
V1 delivery. Requirements historically labelled V1 are V2-B0 internal
prerequisites and must not be reported as a completed product version.

All portable behavior must be implemented and verified on CPU, including
Torch CPU tests where tensor logic is involved. Every feature change must
update its implementation status documentation, support matrix, test mapping,
known gaps, and CPU/NPU verification status.

Keep one authoritative implementation for each protocol, state machine, and
resource ledger. Do not duplicate xLLM logic in xllm-service or add parallel
compatibility paths without an explicit owner, exit condition, and tests.
