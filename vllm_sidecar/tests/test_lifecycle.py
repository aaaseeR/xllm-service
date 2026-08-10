# Copyright 2026 The xLLM Authors. All Rights Reserved.

import requests

from vllm_sidecar.agent import AgentRuntime
from vllm_sidecar.attempts import AttemptKey, AttemptLedger
from vllm_sidecar.lifecycle import LifecycleController


_TOKEN = "test-internal-token"


def _command(
    operation_id: str,
    generation: int,
    action: str = "PROVIDER_LIFECYCLE_ACTION_BEGIN_DRAIN",
    leader_epoch: int = 10,
    leader_incarnation: str = "leader-1",
    engine_uid: str = "engine-1",
    engine_incarnation: str = "inc-1",
) -> dict:
    return {
        "schema_version": 1,
        "operation_id": operation_id,
        "leader_incarnation": leader_incarnation,
        "leader_epoch": leader_epoch,
        "desired_generation": generation,
        "engine_uid": engine_uid,
        "engine_incarnation": engine_incarnation,
        "action": action,
    }


def _controller() -> tuple[AttemptLedger, LifecycleController]:
    ledger = AttemptLedger(max_records=8, terminal_ttl_seconds=60.0)
    controller = LifecycleController(ledger, max_records=8)
    ledger.activate("inc-1")
    controller.activate("engine-1", "inc-1")
    return ledger, controller


def test_drain_closes_admission_without_cancelling_active_attempt() -> None:
    ledger, controller = _controller()
    assert ledger.begin("request-1", 0, "inc-1", 1000).accepted

    command = _command("drain-1", 1)
    response = controller.execute(command)
    assert response["code"] == "PROVIDER_LIFECYCLE_CODE_IN_PROGRESS"
    assert response["drain"]["admission_closed"] is True
    assert response["drain"]["decode_sequences"] == 1
    assert not ledger.accepting()
    assert ledger.query("request-1", 0, "inc-1").state == (
        "ATTEMPT_LIFECYCLE_STATE_GENERATION_COMMITTED"
    )
    assert not ledger.begin("request-2", 0, "inc-1", 1000).accepted

    # A successful etcd keepalive for the same incarnation must not reopen a
    # planned drain.
    ledger.activate("inc-1")
    controller.activate("engine-1", "inc-1")
    assert not ledger.accepting()
    replay = controller.execute(command)
    assert replay["replayed"] is True
    assert replay["drain"]["decode_sequences"] == 1

    ledger.finish(
        AttemptKey("request-1", 0),
        "inc-1",
        "ATTEMPT_LIFECYCLE_STATE_DONE",
        "ADMISSION_REASON_ATTEMPT_TERMINAL",
    )
    complete = controller.query(command)
    assert complete["code"] == "PROVIDER_LIFECYCLE_CODE_SUCCEEDED"
    assert complete["drain"]["decode_sequences"] == 0


def test_newer_generation_can_cancel_only_an_uncommitted_drain() -> None:
    ledger, controller = _controller()
    assert ledger.begin("request-1", 0, "inc-1", 1000).accepted
    assert controller.execute(_command("drain-1", 1))["code"] == (
        "PROVIDER_LIFECYCLE_CODE_IN_PROGRESS"
    )

    cancel = controller.execute(
        _command(
            "cancel-2",
            2,
            action="PROVIDER_LIFECYCLE_ACTION_CANCEL_DRAIN",
        )
    )
    assert cancel["code"] == "PROVIDER_LIFECYCLE_CODE_SUCCEEDED"
    assert ledger.accepting()
    assert ledger.query("request-1", 0, "inc-1").state == (
        "ATTEMPT_LIFECYCLE_STATE_GENERATION_COMMITTED"
    )
    assert controller.query(_command("drain-1", 1))["code"] == (
        "PROVIDER_LIFECYCLE_CODE_FENCED"
    )

    ledger.finish(
        AttemptKey("request-1", 0),
        "inc-1",
        "ATTEMPT_LIFECYCLE_STATE_DONE",
        "ADMISSION_REASON_ATTEMPT_TERMINAL",
    )
    committed = _command("drain-3", 3)
    assert controller.execute(committed)["code"] == (
        "PROVIDER_LIFECYCLE_CODE_SUCCEEDED"
    )
    conflict = controller.execute(
        _command(
            "cancel-4",
            4,
            action="PROVIDER_LIFECYCLE_ACTION_CANCEL_DRAIN",
        )
    )
    assert conflict["code"] == "PROVIDER_LIFECYCLE_CODE_CONFLICT"
    assert not ledger.accepting()


def test_lifecycle_fences_stale_leader_generation_and_incarnation() -> None:
    _, controller = _controller()
    first = _command("drain-1", 1)
    assert controller.execute(first)["code"] == (
        "PROVIDER_LIFECYCLE_CODE_SUCCEEDED"
    )
    assert controller.execute(first)["replayed"] is True

    stale_leader = _command(
        "stale-leader",
        2,
        leader_epoch=9,
        leader_incarnation="leader-old",
    )
    assert controller.execute(stale_leader)["code"] == (
        "PROVIDER_LIFECYCLE_CODE_FENCED"
    )
    stale_incarnation = _command(
        "stale-engine", 2, engine_incarnation="inc-old"
    )
    assert controller.execute(stale_incarnation)["code"] == (
        "PROVIDER_LIFECYCLE_CODE_FENCED"
    )

    controller.fence()
    assert controller.query(first)["code"] == "PROVIDER_LIFECYCLE_CODE_FENCED"


def test_lifecycle_http_endpoint_requires_auth_and_preserves_schema() -> None:
    agent = AgentRuntime(
        "127.0.0.1:0",
        "http://127.0.0.1:1",
        internal_token=_TOKEN,
    )
    agent.start()
    agent.activate("inc-1", "engine-1")
    base = "http://" + agent.listen_address
    command = _command("drain-http", 1)
    try:
        assert requests.post(
            base + "/v1/internal/lifecycle/execute",
            json=command,
            timeout=2.0,
        ).status_code == 401
        response = requests.post(
            base + "/v1/internal/lifecycle/execute",
            json=command,
            headers={"X-Internal-Token": _TOKEN},
            timeout=2.0,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["code"] == "PROVIDER_LIFECYCLE_CODE_SUCCEEDED"
        assert body["operation_id"] == command["operation_id"]
        assert body["leader_epoch"] == command["leader_epoch"]
        assert body["engine_incarnation"] == command["engine_incarnation"]

        invalid = dict(command)
        invalid["leader_epoch"] = True
        assert requests.post(
            base + "/v1/internal/lifecycle/query",
            json=invalid,
            headers={"X-Internal-Token": _TOKEN},
            timeout=2.0,
        ).status_code == 400
    finally:
        agent.stop()
