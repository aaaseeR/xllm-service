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
"""Incarnation-fenced V3 lifecycle controller for the aggregated Agent."""

from __future__ import annotations

from dataclasses import dataclass
import threading

from scripts.logger import logger

from .attempts import AttemptLedger, DrainSnapshot


_SCHEMA_VERSION = 1
_MAX_IDENTITY_BYTES = 256
_MAX_UINT64 = (1 << 64) - 1
_ACTIONS = {
    "PROVIDER_LIFECYCLE_ACTION_BEGIN_DRAIN",
    "PROVIDER_LIFECYCLE_ACTION_CANCEL_DRAIN",
}
_COMMAND_FIELDS = {
    "schema_version",
    "operation_id",
    "leader_incarnation",
    "leader_epoch",
    "desired_generation",
    "engine_uid",
    "engine_incarnation",
    "action",
}


def _valid_identity(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and len(value.encode("utf-8")) <= _MAX_IDENTITY_BYTES
    )


def _valid_uint64(value: object, *, positive: bool = False) -> bool:
    return (
        type(value) is int
        and (value > 0 if positive else value >= 0)
        and value <= _MAX_UINT64
    )


@dataclass(frozen=True)
class LifecycleCommand:
    schema_version: int
    operation_id: str
    leader_incarnation: str
    leader_epoch: int
    desired_generation: int
    engine_uid: str
    engine_incarnation: str
    action: str

    @classmethod
    def parse(cls, payload: object) -> LifecycleCommand:
        if not isinstance(payload, dict) or set(payload) != _COMMAND_FIELDS:
            raise ValueError("lifecycle command fields are invalid")
        command = cls(
            schema_version=payload.get("schema_version"),
            operation_id=payload.get("operation_id"),
            leader_incarnation=payload.get("leader_incarnation"),
            leader_epoch=payload.get("leader_epoch"),
            desired_generation=payload.get("desired_generation"),
            engine_uid=payload.get("engine_uid"),
            engine_incarnation=payload.get("engine_incarnation"),
            action=payload.get("action"),
        )
        if (
            command.schema_version != _SCHEMA_VERSION
            or not _valid_identity(command.operation_id)
            or not _valid_identity(command.leader_incarnation)
            or not _valid_uint64(command.leader_epoch, positive=True)
            or not _valid_uint64(command.desired_generation, positive=True)
            or not _valid_identity(command.engine_uid)
            or not _valid_identity(command.engine_incarnation)
            or command.action not in _ACTIONS
        ):
            raise ValueError("lifecycle command values are invalid")
        return command


@dataclass
class _LifecycleRecord:
    command: LifecycleCommand
    code: str
    message: str
    committed: bool = False


class LifecycleController:
    """Applies drain admission exactly once within an Engine incarnation."""

    def __init__(self, ledger: AttemptLedger, max_records: int = 4096) -> None:
        if type(max_records) is not int or max_records <= 0:
            raise ValueError("lifecycle record limit must be positive")
        self._ledger = ledger
        self._max_records = max_records
        self._lock = threading.Lock()
        self._engine_uid = ""
        self._engine_incarnation = ""
        self._active = False
        self._leader_incarnation = ""
        self._leader_epoch = 0
        self._highest_generation = 0
        self._records: dict[str, _LifecycleRecord] = {}
        self._drain_operation_id = ""

    def activate(self, engine_uid: str, engine_incarnation: str) -> None:
        if not _valid_identity(engine_uid) or not _valid_identity(
            engine_incarnation
        ):
            raise ValueError("Engine identity must be bounded and non-empty")
        with self._lock:
            if engine_incarnation != self._engine_incarnation:
                self._engine_uid = engine_uid
                self._engine_incarnation = engine_incarnation
                self._leader_incarnation = ""
                self._leader_epoch = 0
                self._highest_generation = 0
                self._records.clear()
                self._drain_operation_id = ""
            elif self._engine_uid and engine_uid != self._engine_uid:
                raise ValueError("engine_uid changed within one incarnation")
            self._active = True

    def fence(self) -> None:
        with self._lock:
            self._active = False

    def execute(self, payload: object) -> dict:
        command = LifecycleCommand.parse(payload)
        with self._lock:
            fenced = self._validate_fence_locked(command)
            if fenced is not None:
                return fenced
            previous = self._records.get(command.operation_id)
            if previous is not None:
                if previous.command != command:
                    return self._response_locked(
                        command,
                        "PROVIDER_LIFECYCLE_CODE_CONFLICT",
                        "operation id collides with another command",
                        replayed=True,
                    )
                self._refresh_record_locked(previous)
                return self._record_response_locked(previous, replayed=True)
            if (
                command.leader_epoch == self._leader_epoch
                and command.desired_generation == self._highest_generation
            ):
                return self._response_locked(
                    command,
                    "PROVIDER_LIFECYCLE_CODE_CONFLICT",
                    "desired generation already has another operation",
                )
            if len(self._records) >= self._max_records:
                return self._response_locked(
                    command,
                    "PROVIDER_LIFECYCLE_CODE_RETRYABLE_ERROR",
                    "lifecycle operation capacity is exhausted",
                )

            self._leader_incarnation = command.leader_incarnation
            self._leader_epoch = command.leader_epoch
            self._highest_generation = command.desired_generation
            if command.action == "PROVIDER_LIFECYCLE_ACTION_BEGIN_DRAIN":
                record = self._begin_drain_locked(command)
            else:
                record = self._cancel_drain_locked(command)
            self._records[command.operation_id] = record
            return self._record_response_locked(record, replayed=False)

    def query(self, payload: object) -> dict:
        command = LifecycleCommand.parse(payload)
        with self._lock:
            fenced = self._validate_fence_locked(command)
            if fenced is not None:
                return fenced
            record = self._records.get(command.operation_id)
            if record is None:
                return self._response_locked(
                    command,
                    "PROVIDER_LIFECYCLE_CODE_NOT_FOUND",
                    "lifecycle operation was not found",
                )
            if record.command != command:
                return self._response_locked(
                    command,
                    "PROVIDER_LIFECYCLE_CODE_CONFLICT",
                    "operation id collides with another command",
                    replayed=True,
                )
            self._refresh_record_locked(record)
            return self._record_response_locked(record, replayed=True)

    def engine_state(self) -> dict:
        with self._lock:
            if self._drain_operation_id:
                record = self._records.get(self._drain_operation_id)
                if record is not None:
                    self._refresh_record_locked(record)
            drain = self._ledger.drain_snapshot()
            if not self._active:
                lifecycle = "ENGINE_LIFECYCLE_FENCED"
            elif drain.admission_closed:
                lifecycle = "ENGINE_LIFECYCLE_DRAINING"
            else:
                lifecycle = "ENGINE_LIFECYCLE_READY"
            return {
                "lifecycle": lifecycle,
                "operation_id": self._drain_operation_id,
                "leader_incarnation": self._leader_incarnation,
                "leader_epoch": self._leader_epoch,
                "desired_generation": self._highest_generation,
                "drain": self._drain_json(drain),
            }

    def _validate_fence_locked(self, command: LifecycleCommand) -> dict | None:
        if (
            not self._active
            or command.engine_uid != self._engine_uid
            or command.engine_incarnation != self._engine_incarnation
        ):
            return self._response_locked(
                command,
                "PROVIDER_LIFECYCLE_CODE_FENCED",
                "lifecycle command targets a stale Engine incarnation",
            )
        if command.leader_epoch < self._leader_epoch or (
            command.leader_epoch == self._leader_epoch
            and self._leader_incarnation
            and command.leader_incarnation != self._leader_incarnation
        ):
            return self._response_locked(
                command,
                "PROVIDER_LIFECYCLE_CODE_FENCED",
                "lifecycle command is from a stale leader",
            )
        if (
            command.leader_epoch == self._leader_epoch
            and command.desired_generation < self._highest_generation
        ):
            return self._response_locked(
                command,
                "PROVIDER_LIFECYCLE_CODE_FENCED",
                "lifecycle command is from a stale desired generation",
            )
        return None

    def _begin_drain_locked(
        self, command: LifecycleCommand
    ) -> _LifecycleRecord:
        snapshot = self._ledger.begin_drain()
        committed = self._drain_complete(snapshot)
        record = _LifecycleRecord(
            command=command,
            code=(
                "PROVIDER_LIFECYCLE_CODE_SUCCEEDED"
                if committed
                else "PROVIDER_LIFECYCLE_CODE_IN_PROGRESS"
            ),
            message="drain complete" if committed else "drain in progress",
            committed=committed,
        )
        self._drain_operation_id = command.operation_id
        return record

    def _cancel_drain_locked(
        self, command: LifecycleCommand
    ) -> _LifecycleRecord:
        active_drain = self._records.get(self._drain_operation_id)
        if active_drain is not None:
            self._refresh_record_locked(active_drain)
            if active_drain.committed:
                return _LifecycleRecord(
                    command=command,
                    code="PROVIDER_LIFECYCLE_CODE_CONFLICT",
                    message="committed drain cannot be cancelled",
                )
            active_drain.code = "PROVIDER_LIFECYCLE_CODE_CONFLICT"
            active_drain.message = "drain cancelled by a newer generation"
        self._ledger.cancel_drain()
        self._drain_operation_id = ""
        return _LifecycleRecord(
            command=command,
            code="PROVIDER_LIFECYCLE_CODE_SUCCEEDED",
            message="drain cancelled",
            committed=True,
        )

    def _refresh_record_locked(self, record: _LifecycleRecord) -> None:
        if (
            record.command.action
            != "PROVIDER_LIFECYCLE_ACTION_BEGIN_DRAIN"
            or record.command.operation_id != self._drain_operation_id
            or record.code == "PROVIDER_LIFECYCLE_CODE_CONFLICT"
        ):
            return
        snapshot = self._ledger.drain_snapshot()
        record.committed = self._drain_complete(snapshot)
        record.code = (
            "PROVIDER_LIFECYCLE_CODE_SUCCEEDED"
            if record.committed
            else "PROVIDER_LIFECYCLE_CODE_IN_PROGRESS"
        )
        record.message = (
            "drain complete" if record.committed else "drain in progress"
        )

    def _record_response_locked(
        self, record: _LifecycleRecord, *, replayed: bool
    ) -> dict:
        return self._response_locked(
            record.command,
            record.code,
            record.message,
            replayed=replayed,
        )

    def _response_locked(
        self,
        command: LifecycleCommand,
        code: str,
        message: str,
        *,
        replayed: bool = False,
    ) -> dict:
        drain = self._ledger.drain_snapshot()
        if not self._active:
            lifecycle = "ENGINE_LIFECYCLE_FENCED"
        elif drain.admission_closed:
            lifecycle = "ENGINE_LIFECYCLE_DRAINING"
        else:
            lifecycle = "ENGINE_LIFECYCLE_READY"
        log = logger.debug
        if code in {
            "PROVIDER_LIFECYCLE_CODE_FENCED",
            "PROVIDER_LIFECYCLE_CODE_CONFLICT",
            "PROVIDER_LIFECYCLE_CODE_RETRYABLE_ERROR",
            "PROVIDER_LIFECYCLE_CODE_TERMINAL_ERROR",
            "PROVIDER_LIFECYCLE_CODE_INVALID",
        }:
            log = logger.warning
        elif code == "PROVIDER_LIFECYCLE_CODE_SUCCEEDED" and not replayed:
            log = logger.info
        log(
            "xllm_service_v3_agent_lifecycle operation_id=%r action=%s "
            "engine_uid=%r engine_incarnation=%r code=%s lifecycle=%s "
            "replayed=%s active_attempts=%d admission_closed=%s message=%r",
            command.operation_id,
            command.action,
            command.engine_uid,
            command.engine_incarnation,
            code,
            lifecycle,
            replayed,
            drain.active_attempts,
            drain.admission_closed,
            message,
        )
        return {
            "code": code,
            "operation_id": command.operation_id,
            "action": command.action,
            "leader_incarnation": command.leader_incarnation,
            "leader_epoch": command.leader_epoch,
            "desired_generation": command.desired_generation,
            "engine_uid": command.engine_uid,
            "engine_incarnation": command.engine_incarnation,
            "lifecycle": lifecycle,
            "drain": self._drain_json(drain),
            "replayed": replayed,
            "message": message,
        }

    @staticmethod
    def _drain_complete(snapshot: DrainSnapshot) -> bool:
        return snapshot.admission_closed and snapshot.active_attempts == 0

    @staticmethod
    def _drain_json(snapshot: DrainSnapshot) -> dict:
        return {
            "admission_closed": snapshot.admission_closed,
            "prefill_queue": 0,
            # Aggregated vLLM has no external KV transfer or reservation path.
            # Native/disaggregated Providers report those counters themselves.
            "active_transfers": 0,
            "active_reservations": 0,
            "decode_sequences": snapshot.active_attempts,
            "pending_output": 0,
            "pending_cleanup": 0,
        }
