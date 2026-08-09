# Copyright 2026 The xLLM Authors. All Rights Reserved.

from concurrent.futures import ThreadPoolExecutor

from vllm_sidecar.attempts import AttemptKey, AttemptLedger


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class Closable:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_attempt_lifecycle_deadline_cancel_fence_and_capacity() -> None:
    clock = Clock()
    ledger = AttemptLedger(max_records=2, terminal_ttl_seconds=5.0, clock=clock)
    assert not ledger.begin("r0", 0, "inc-1", 1000).accepted

    ledger.activate("inc-1")
    assert ledger.begin("r1", 0, "inc-1", 1000).accepted
    upstream = Closable()
    assert ledger.attach(AttemptKey("r1", 0), "inc-1", upstream)
    assert (
        ledger.query("r1", 0, "inc-1").state
        == "ATTEMPT_LIFECYCLE_STATE_RUNNING"
    )

    cancelled = ledger.cancel("r1", 0, "inc-1")
    assert cancelled.state == "ATTEMPT_LIFECYCLE_STATE_CANCELLED"
    assert upstream.closed
    before_create = ledger.cancel("r2", 0, "inc-1")
    assert before_create.state == "ATTEMPT_LIFECYCLE_STATE_CANCELLED_BEFORE_CREATE"
    assert not ledger.begin("r2", 0, "inc-1", 1000).accepted
    assert not ledger.begin("r3", 0, "inc-1", 1000).accepted

    clock.now += 5.0
    assert ledger.begin("r3", 0, "inc-1", 1000).accepted
    expiring = Closable()
    assert ledger.attach(AttemptKey("r3", 0), "inc-1", expiring)
    clock.now += 1.0
    assert ledger.reap_expired() == 1
    assert expiring.closed
    assert (
        ledger.query("r3", 0, "inc-1").state
        == "ATTEMPT_LIFECYCLE_STATE_EXPIRED"
    )


def test_only_one_concurrent_submit_wins() -> None:
    ledger = AttemptLedger(max_records=64, terminal_ttl_seconds=10.0)
    ledger.activate("inc-1")
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(
            executor.map(
                lambda _: ledger.begin("same", 7, "inc-1", 1000), range(64)
            )
        )
    assert sum(result.accepted for result in results) == 1
    assert sum(result.replayed for result in results) == 63


def test_new_incarnation_fences_and_closes_old_attempt() -> None:
    ledger = AttemptLedger(max_records=4, terminal_ttl_seconds=10.0)
    ledger.activate("inc-1")
    assert ledger.begin("r1", 0, "inc-1", 1000).accepted
    upstream = Closable()
    assert ledger.attach(AttemptKey("r1", 0), "inc-1", upstream)

    ledger.activate("inc-2")
    assert upstream.closed
    assert (
        ledger.query("r1", 0, "inc-2").state
        == "ATTEMPT_LIFECYCLE_STATE_ABSENT"
    )
    assert ledger.incarnation_id() == "inc-2"


def test_old_incarnation_cannot_mutate_reused_attempt_key() -> None:
    ledger = AttemptLedger(max_records=4, terminal_ttl_seconds=10.0)
    ledger.activate("inc-1")
    assert ledger.begin("same", 1, "inc-1", 1000).accepted

    ledger.activate("inc-2")
    assert ledger.begin("same", 1, "inc-2", 1000).accepted
    stale_upstream = Closable()
    assert not ledger.attach(AttemptKey("same", 1), "inc-1", stale_upstream)
    ledger.finish(
        AttemptKey("same", 1),
        "inc-1",
        "ATTEMPT_LIFECYCLE_STATE_DONE",
        "ADMISSION_REASON_ATTEMPT_TERMINAL",
    )
    assert (
        ledger.query("same", 1, "inc-2").state
        == "ATTEMPT_LIFECYCLE_STATE_GENERATION_COMMITTED"
    )
    assert ledger.cancel("same", 1, "inc-1").reason == (
        "ADMISSION_REASON_STALE_INCARNATION"
    )
