# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm-service/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from online_cluster_stress import is_expected_abrupt_loss_error


def _fenced_error(**overrides: object) -> str:
    payload: dict[str, object] = {
        "accepted": True,
        "replayed": True,
        "state": "ATTEMPT_LIFECYCLE_STATE_CANCELLED",
        "reason": "ADMISSION_REASON_INTERNAL_ERROR",
        "request_uid": "request-1",
        "attempt_seq": 0,
        "incarnation_id": "incarnation-1",
    }
    payload.update(overrides)
    return (
        "HTTP 500: [service][E1010]HTTP/1.1 409 Conflict: "
        + json.dumps(payload, separators=(",", ":"))
    )


def test_abrupt_loss_accepts_known_transport_failures() -> None:
    for detail in (
        "Connection refused",
        "Connection reset by peer",
        "backend instance is not available",
    ):
        assert is_expected_abrupt_loss_error(f"HTTP 500: {detail}")


def test_abrupt_loss_accepts_exact_runtime_fence_proof() -> None:
    assert is_expected_abrupt_loss_error(_fenced_error())


def test_abrupt_loss_rejects_other_conflicts() -> None:
    assert not is_expected_abrupt_loss_error(
        _fenced_error(reason="ADMISSION_REASON_ENGINE_DRAINING")
    )
    assert not is_expected_abrupt_loss_error(
        _fenced_error(state="ATTEMPT_LIFECYCLE_STATE_RUNNING")
    )
    assert not is_expected_abrupt_loss_error(_fenced_error(replayed=False))


def test_abrupt_loss_rejects_malformed_or_incomplete_proof() -> None:
    assert not is_expected_abrupt_loss_error(
        "HTTP 500: HTTP/1.1 409 Conflict: {\"accepted\":true"
    )
    assert not is_expected_abrupt_loss_error(
        _fenced_error(incarnation_id="")
    )
    assert not is_expected_abrupt_loss_error(_fenced_error(extra="field"))


def test_abrupt_loss_rejects_unrelated_failures() -> None:
    assert not is_expected_abrupt_loss_error("HTTP 503: overloaded")
