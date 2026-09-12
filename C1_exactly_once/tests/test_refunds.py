import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from circuit_breaker import CircuitBreaker
from dependency import VerificationDependency
from idempotency import IdempotencyStore
from refund_service import RefundService


async def test_exactly_once():
    svc = RefundService(
        IdempotencyStore(":memory:"),
        VerificationDependency(delay_sec=0.05),
        CircuitBreaker(),
    )
    results = await asyncio.gather(
        *[svc.issue_refund("k1", "o1", 999) for _ in range(15)]
    )
    assert svc.refunds_issued == 1
    assert sum(1 for r in results if r.body.get("outcome") == "success") == 1


async def test_circuit_retryable():
    dep = VerificationDependency(delay_sec=0.01, force_fail=True)
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout_sec=0.15)
    svc = RefundService(IdempotencyStore(":memory:"), dep, breaker, max_retries=0)
    for i in range(3):
        await svc.issue_refund(f"trip-{i}", "o", 100)
    for _ in range(3):
        breaker.record_failure()
    r1 = await svc.issue_refund("same", "o", 100)
    assert r1.status_code == 503
    dep.force_fail = False
    await asyncio.sleep(0.2)
    r2 = await svc.issue_refund("same", "o", 100)
    assert r2.body["outcome"] == "success"


async def test_health_under_load():
    svc = RefundService(
        IdempotencyStore(":memory:"),
        VerificationDependency(delay_sec=0.2),
        CircuitBreaker(failure_threshold=100),
    )
    refund = asyncio.create_task(svc.issue_refund("h1", "o", 100))
    healths = await asyncio.gather(*[svc.health() for _ in range(20)])
    await refund
    assert all(h.status_code == 200 for h in healths)


def run():
    asyncio.run(test_exactly_once())
    print("✓ test_exactly_once")
    asyncio.run(test_circuit_retryable())
    print("✓ test_circuit_retryable")
    asyncio.run(test_health_under_load())
    print("✓ test_health_under_load")
    print("\nAll C1 unit tests passed!")


if __name__ == "__main__":
    run()
