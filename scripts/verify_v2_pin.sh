#!/usr/bin/env bash

# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
xllm_path="${repo_root}/third_party/xllm"

if [[ ! -d "${xllm_path}/.git" && ! -f "${xllm_path}/.git" ]]; then
  echo "third_party/xllm is not initialized" >&2
  exit 1
fi

indexed_revision="$(git -C "${repo_root}" ls-files -s third_party/xllm | awk '{print $2}')"
checked_out_revision="$(git -C "${xllm_path}" rev-parse HEAD)"

if [[ -z "${indexed_revision}" || "${indexed_revision}" != "${checked_out_revision}" ]]; then
  echo "third_party/xllm does not match the indexed gitlink" >&2
  echo "indexed=${indexed_revision:-missing}" >&2
  echo "checked_out=${checked_out_revision}" >&2
  exit 1
fi

required_protocols=(
  "xllm/proto/observability.proto"
  "xllm/proto/provider.proto"
)

for protocol in "${required_protocols[@]}"; do
  if [[ ! -f "${xllm_path}/${protocol}" ]]; then
    echo "pinned xLLM is missing required protocol: ${protocol}" >&2
    exit 1
  fi
done

echo "V2 pin verified: ${checked_out_revision}"
