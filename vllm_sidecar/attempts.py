# Copyright 2026 The xLLM Authors. All Rights Reserved.
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
"""Bounded, incarnation-scoped attempt ledger for the aggregated Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
import time
from typing import Callable, Protocol


_MAX_IDENTITY_BYTES = 256
_MAX_UINT64 = (1 << 64) - 1


def _valid_identity(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and len(value.encode("utf-8")) <= _MAX_IDENTITY_BYTES
    )


def _valid_attempt_seq(value: int) -> bool:
    return type(value) is int and 0 <= value <= _MAX_UINT64


def _positive_finite(value: object) -> bool:
    if type(value) not in (int, float) or value <= 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


class Closable(Protocol):
    def close(self) -> None: ...


@dataclass(frozen=True)
class AttemptKey:
    request_uid: str
    attempt_seq: int


@dataclass
class AttemptRecord:
    key: AttemptKey
    incarnation_id: str
    state: str
    reason: str
    deadline: float
    capacity_kind: str
    terminal_at: float | None = None
    upstream: Closable | None = field(default=None, repr=False)


@dataclass(frozen=True)
class AttemptResult:
    accepted: bool
    replayed: bool
    state: str
    reason: str


class AttemptLedger:
    def __init__(
        self,
        max_records: int = 8192,
        terminal_ttl_seconds: float = 60.0,
        max_cancel_fences: int | None = None,
        negative_fence_ttl_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_cancel_fences is None:
            max_cancel_fences = max_records
        if (
            type(max_records) is not int
            or max_records <= 0
            or type(max_cancel_fences) is not int
            or max_cancel_fences <= 0
            or not _positive_finite(terminal_ttl_seconds)
            or not _positive_finite(negative_fence_ttl_seconds)
        ):
            raise ValueError("attempt ledger limits must be positive")
        self._max_records = max_records
        self._max_cancel_fences = max_cancel_fences
        self._terminal_ttl = terminal_ttl_seconds
        self._negative_fence_ttl = negative_fence_ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._records: dict[AttemptKey, AttemptRecord] = {}
        self._attempt_records = 0
        self._cancel_fences = 0
        self._incarnation_id = ""
        self._activation_enabled = False
        self._negative_fence_pressure = False
        self._accepting = False

    def activate(self, incarnation_id: str) -> None:
        if not _valid_identity(incarnation_id):
            raise ValueError("incarnation_id must be a bounded identity")
        to_close = []
        with self._lock:
            if incarnation_id != self._incarnation_id:
                to_close = self._fence_locked("ADMISSION_REASON_STALE_INCARNATION")
                self._records.clear()
                self._attempt_records = 0
                self._cancel_fences = 0
                self._negative_fence_pressure = False
                self._incarnation_id = incarnation_id
            self._activation_enabled = True
            self._accepting = not self._negative_fence_pressure
        self._close_all(to_close)

    def fence(self, reason: str = "ADMISSION_REASON_ENGINE_DRAINING") -> None:
        with self._lock:
            self._activation_enabled = False
            to_close = self._fence_locked(reason)
        self._close_all(to_close)

    def _fence_locked(self, reason: str) -> list[Closable]:
        self._accepting = False
        now = self._clock()
        to_close = []
        for record in self._records.values():
            if record.terminal_at is not None:
                continue
            record.state = "ATTEMPT_LIFECYCLE_STATE_CANCELLED"
            record.reason = reason
            record.terminal_at = now
            if record.upstream is not None:
                to_close.append(record.upstream)
                record.upstream = None
        return to_close

    @staticmethod
    def _close_all(items: list[Closable]) -> None:
        for item in items:
            try:
                item.close()
            except Exception:
                pass

    def accepting(self) -> bool:
        with self._lock:
            return self._accepting

    def incarnation_id(self) -> str:
        with self._lock:
            return self._incarnation_id

    def begin(
        self,
        request_uid: str,
        attempt_seq: int,
        incarnation_id: str,
        remaining_deadline_ms: int,
    ) -> AttemptResult:
        if (
            not _valid_identity(request_uid)
            or not _valid_attempt_seq(attempt_seq)
            or not _valid_identity(incarnation_id)
            or type(remaining_deadline_ms) is not int
            or remaining_deadline_ms <= 0
            or remaining_deadline_ms > _MAX_UINT64
        ):
            return AttemptResult(
                False,
                False,
                "ATTEMPT_LIFECYCLE_STATE_FAILED",
                "ADMISSION_REASON_INVALID_REQUEST",
            )
        now = self._clock()
        key = AttemptKey(request_uid, attempt_seq)
        with self._lock:
            self._reap_locked(now)
            if incarnation_id != self._incarnation_id:
                return AttemptResult(
                    False,
                    False,
                    "ATTEMPT_LIFECYCLE_STATE_FAILED",
                    "ADMISSION_REASON_STALE_INCARNATION",
                )
            previous = self._records.get(key)
            if previous is not None:
                return AttemptResult(
                    False, True, previous.state, previous.reason
                )
            if not self._accepting:
                return AttemptResult(
                    False,
                    False,
                    "ATTEMPT_LIFECYCLE_STATE_FAILED",
                    "ADMISSION_REASON_ENGINE_DRAINING",
                )
            if self._attempt_records >= self._max_records:
                return AttemptResult(
                    False,
                    False,
                    "ATTEMPT_LIFECYCLE_STATE_FAILED",
                    "ADMISSION_REASON_TOMBSTONE_CAPACITY",
                )
            self._records[key] = AttemptRecord(
                key=key,
                incarnation_id=self._incarnation_id,
                state="ATTEMPT_LIFECYCLE_STATE_GENERATION_COMMITTED",
                reason="ADMISSION_REASON_NONE",
                deadline=now + remaining_deadline_ms / 1000.0,
                capacity_kind="attempt",
            )
            self._attempt_records += 1
            return AttemptResult(
                True,
                False,
                "ATTEMPT_LIFECYCLE_STATE_GENERATION_COMMITTED",
                "ADMISSION_REASON_NONE",
            )

    def attach(
        self, key: AttemptKey, incarnation_id: str, upstream: Closable
    ) -> bool:
        with self._lock:
            record = self._records.get(key)
            if (
                record is None
                or record.incarnation_id != incarnation_id
                or record.terminal_at is not None
            ):
                return False
            record.upstream = upstream
            record.state = "ATTEMPT_LIFECYCLE_STATE_RUNNING"
            return True

    def finish(
        self, key: AttemptKey, incarnation_id: str, state: str, reason: str
    ) -> None:
        with self._lock:
            record = self._records.get(key)
            if (
                record is None
                or record.incarnation_id != incarnation_id
                or record.terminal_at is not None
            ):
                return
            record.state = state
            record.reason = reason
            record.terminal_at = self._clock()
            record.upstream = None

    def cancel(
        self, request_uid: str, attempt_seq: int, incarnation_id: str
    ) -> AttemptResult:
        if (
            not _valid_identity(request_uid)
            or not _valid_attempt_seq(attempt_seq)
            or not _valid_identity(incarnation_id)
        ):
            return AttemptResult(
                False,
                False,
                "ATTEMPT_LIFECYCLE_STATE_FAILED",
                "ADMISSION_REASON_INVALID_REQUEST",
            )
        key = AttemptKey(request_uid, attempt_seq)
        upstream = None
        with self._lock:
            now = self._clock()
            self._reap_locked(now)
            if incarnation_id != self._incarnation_id:
                return AttemptResult(
                    False,
                    False,
                    "ATTEMPT_LIFECYCLE_STATE_FAILED",
                    "ADMISSION_REASON_STALE_INCARNATION",
                )
            record = self._records.get(key)
            if record is None:
                if self._cancel_fences >= self._max_cancel_fences:
                    self._negative_fence_pressure = True
                    self._accepting = False
                    return AttemptResult(
                        False,
                        False,
                        "ATTEMPT_LIFECYCLE_STATE_FAILED",
                        "ADMISSION_REASON_CANCEL_FENCE_CAPACITY",
                    )
                record = AttemptRecord(
                    key=key,
                    incarnation_id=self._incarnation_id,
                    state="ATTEMPT_LIFECYCLE_STATE_CANCELLED_BEFORE_CREATE",
                    reason="ADMISSION_REASON_CANCELLED_BEFORE_CREATE",
                    deadline=now,
                    capacity_kind="cancel_fence",
                    terminal_at=now,
                )
                self._records[key] = record
                self._cancel_fences += 1
                if self._cancel_fences >= self._max_cancel_fences:
                    self._negative_fence_pressure = True
                    self._accepting = False
            elif record.terminal_at is None:
                record.state = "ATTEMPT_LIFECYCLE_STATE_CANCELLED"
                record.reason = "ADMISSION_REASON_CANCELLED"
                record.terminal_at = now
                upstream = record.upstream
                record.upstream = None
            result = AttemptResult(True, False, record.state, record.reason)
        self._close_all([upstream] if upstream is not None else [])
        return result

    def query(
        self, request_uid: str, attempt_seq: int, incarnation_id: str
    ) -> AttemptResult:
        if (
            not _valid_identity(request_uid)
            or not _valid_attempt_seq(attempt_seq)
            or not _valid_identity(incarnation_id)
        ):
            return AttemptResult(
                False,
                False,
                "ATTEMPT_LIFECYCLE_STATE_FAILED",
                "ADMISSION_REASON_INVALID_REQUEST",
            )
        key = AttemptKey(request_uid, attempt_seq)
        with self._lock:
            self._reap_locked(self._clock())
            if incarnation_id != self._incarnation_id:
                return AttemptResult(
                    False,
                    False,
                    "ATTEMPT_LIFECYCLE_STATE_FAILED",
                    "ADMISSION_REASON_STALE_INCARNATION",
                )
            record = self._records.get(key)
            if record is None:
                return AttemptResult(
                    False,
                    False,
                    "ATTEMPT_LIFECYCLE_STATE_ABSENT",
                    "ADMISSION_REASON_ATTEMPT_NOT_FOUND",
                )
            return AttemptResult(True, True, record.state, record.reason)

    def reap_expired(self) -> int:
        to_close = []
        with self._lock:
            now = self._clock()
            for record in self._records.values():
                if record.terminal_at is not None or now < record.deadline:
                    continue
                record.state = "ATTEMPT_LIFECYCLE_STATE_EXPIRED"
                record.reason = "ADMISSION_REASON_DEADLINE_EXCEEDED"
                record.terminal_at = now
                if record.upstream is not None:
                    to_close.append(record.upstream)
                    record.upstream = None
            self._reap_locked(now)
        self._close_all(to_close)
        return len(to_close)

    def _reap_locked(self, now: float) -> None:
        expired = [
            key
            for key, record in self._records.items()
            if record.terminal_at is not None
            and now - record.terminal_at
            >= (
                self._negative_fence_ttl
                if record.capacity_kind == "cancel_fence"
                else self._terminal_ttl
            )
        ]
        for key in expired:
            record = self._records.pop(key)
            if record.capacity_kind == "cancel_fence":
                self._cancel_fences -= 1
            else:
                self._attempt_records -= 1
        low_watermark = self._max_cancel_fences // 2
        if (
            self._negative_fence_pressure
            and self._cancel_fences <= low_watermark
        ):
            self._negative_fence_pressure = False
            self._accepting = self._activation_enabled

    def size(self) -> int:
        with self._lock:
            return len(self._records)

    def capacity_sizes(self) -> tuple[int, int]:
        with self._lock:
            return self._attempt_records, self._cancel_fences
