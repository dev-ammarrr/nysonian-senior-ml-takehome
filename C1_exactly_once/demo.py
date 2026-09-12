"""
C1 concurrent tests + demo scenarios.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from circuit_breaker import CircuitBreaker
from dependency import VerificationDependency
from idempotency import IdempotencyStore
from refund_service import RefundService


def section(t: str):
    print(f"\n{'='*72}\n  {t}\n{'='*72}\n")


async def test_concurrent_duplicates_issue_once():
    store = IdempotencyStore(":memory:")
    dep = VerificationDependency(delay_sec=0.1, fail_rate=0.0)
    breaker = CircuitBreaker(failure_threshold=5)
    svc = RefundService(store, dep, breaker)

    key = "idem-concurrent-1"
    results = await asyncio.gather(
        *[svc.issue_refund(key, "ord_1", 2500) for _ in range(20)]
    )
    successes = [r for r in results if r.body.get("outcome") == "success"]
    duplicates = [r for r in results if r.body.get("outcome") == "duplicate"]
    in_progress = [r for r in results if r.body.get("outcome") == "in_progress"]

    assert svc.refunds_issued == 1
    assert len(successes) == 1
    assert len(duplicates) + len(in_progress) == 19
    # Eventually duplicates should win after first completes — poll once more
    final = await svc.issue_refund(key, "ord_1", 2500)
    assert final.body["outcome"] == "duplicate"
    assert final.body["refund"]["refund_id"] == successes[0].body["refund"]["refund_id"]
    print("✓ concurrent duplicates → exactly one refund issued")


async def test_in_progress_while_slow_upstream():
    store = IdempotencyStore(":memory:")
    dep = VerificationDependency(delay_sec=0.3)
    breaker = CircuitBreaker()
    svc = RefundService(store, dep, breaker)
    key = "idem-slow"

    async def first():
        return await svc.issue_refund(key, "ord_2", 1000)

    async def second():
        await asyncio.sleep(0.05)  # start after first claimed
        return await svc.issue_refund(key, "ord_2", 1000)

    r1, r2 = await asyncio.gather(first(), second())
    outcomes = {r1.body["outcome"], r2.body["outcome"]}
    assert "success" in outcomes
    assert "in_progress" in outcomes or "duplicate" in outcomes
    print("✓ while upstream slow, peer gets in_progress/duplicate — not a second refund")
    assert svc.refunds_issued == 1


async def test_circuit_open_not_permanent_duplicate():
    store = IdempotencyStore(":memory:")
    dep = VerificationDependency(delay_sec=0.01, force_fail=True)
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout_sec=0.2)
    svc = RefundService(store, dep, breaker, max_retries=0)

    key = "idem-circuit"
    # Trip the breaker
    for _ in range(3):
        await svc.issue_refund(f"warm-{_}", "ord", 1000)

    assert breaker.stats().state.value in ("open", "half_open")

    # Force open
    for _ in range(5):
        breaker.record_failure()

    r_open = await svc.issue_refund(key, "ord_3", 1000)
    assert r_open.status_code == 503
    assert r_open.body["outcome"] == "circuit_open"

    # Heal dependency + wait for recovery
    dep.force_fail = False
    await asyncio.sleep(0.25)

    r_retry = await svc.issue_refund(key, "ord_3", 1000)
    assert r_retry.body["outcome"] == "success", r_retry.body
    assert svc.refunds_issued >= 1
    print("✓ circuit_open does NOT permanently burn idempotency key")


async def test_healthy_requests_during_dependency_slowness():
    store = IdempotencyStore(":memory:")
    dep = VerificationDependency(delay_sec=0.5)
    breaker = CircuitBreaker(failure_threshold=50)
    svc = RefundService(store, dep, breaker)

    async def slow_refund():
        return await svc.issue_refund("slow-1", "ord_slow", 1000)

    async def many_health():
        outs = []
        for _ in range(30):
            outs.append(await svc.health())
            await asyncio.sleep(0.01)
        return outs

    refund_task = asyncio.create_task(slow_refund())
    health_task = asyncio.create_task(many_health())
    refund_res, health_res = await asyncio.gather(refund_task, health_task)

    assert refund_res.status_code == 200
    assert all(h.status_code == 200 for h in health_res)
    assert len(health_res) == 30
    print("✓ health endpoint keeps responding while refund waits on slow dependency")


async def main():
    section("C1: Exactly-Once Refund + Circuit Breaker")
    print("Framing: POST /refunds with Idempotency-Key")
    print("Upstream: slow verification API (mocked)")
    print()

    await test_concurrent_duplicates_issue_once()
    await test_in_progress_while_slow_upstream()
    await test_circuit_open_not_permanent_duplicate()
    await test_healthy_requests_during_dependency_slowness()

    section("Caller-visible outcomes (decisions)")
    print("1) duplicate detected → 200 + outcome=duplicate + cached refund body")
    print("   Why: clients must be able to retry safely after timeouts.")
    print("2) circuit open → 503 + outcome=circuit_open (failed_retryable in store)")
    print("   Why: not 200; caller should back off; key must remain reclaimable.")
    print("3) genuine success → 200 + outcome=success + refund_id")
    print("4) in_progress → 409 (concurrent twin while upstream still running)")
    print("   Why: do not invent a second refund; tell caller to wait/retry.")

    section("All C1 checks passed")


if __name__ == "__main__":
    asyncio.run(main())
