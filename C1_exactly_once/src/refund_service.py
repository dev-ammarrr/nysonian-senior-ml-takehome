"""
C1 — Refund endpoint: exactly-once + resilient to slow upstream.

Framing: agent calls POST /refunds with Idempotency-Key to issue a refund.
Upstream: slow verification API (2–10s, timeouts, rate limits).

Coexistence rules we enforce:
1. Duplicate guard is held as in_progress for the duration of upstream.
   Concurrent retries get 409 in_progress (not a fake success).
2. Circuit-open does NOT permanently consume the idempotency key —
   status=failed_retryable so a later attempt can proceed after recovery.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from circuit_breaker import CircuitBreaker, CircuitOpenError
from dependency import VerificationDependency
from idempotency import IdempotencyStore


@dataclass
class ApiResponse:
    status_code: int
    body: Dict[str, Any]


class RefundService:
    def __init__(
        self,
        store: IdempotencyStore,
        dependency: VerificationDependency,
        breaker: CircuitBreaker,
        max_retries: int = 2,
        backoff_sec: float = 0.05,
    ):
        self.store = store
        self.dependency = dependency
        self.breaker = breaker
        self.max_retries = max_retries
        self.backoff_sec = backoff_sec
        self.refunds_issued = 0
        self._inflight_healthy = 0  # for proving unrelated requests still work

    async def issue_refund(
        self,
        idempotency_key: str,
        order_id: str,
        amount_cents: int,
    ) -> ApiResponse:
        claim, cached = self.store.try_begin(idempotency_key)

        if claim == "duplicate_success":
            # cached is the full success body from mark_success
            refund = None
            if isinstance(cached, dict):
                refund = cached.get("refund", cached)
            return ApiResponse(
                200,
                {
                    "outcome": "duplicate",
                    "refund": refund,
                    "decision": cached.get("decision") if isinstance(cached, dict) else None,
                    "message": "idempotent replay of successful refund",
                },
            )

        if claim == "in_progress":
            return ApiResponse(
                409,
                {
                    "outcome": "in_progress",
                    "message": "same logical refund is already being processed; retry later",
                },
            )

        # claim == proceed
        if not self.breaker.allow():
            body = {
                "outcome": "circuit_open",
                "message": "verification dependency unhealthy; try again after recovery",
            }
            self.store.mark_failed_retryable(idempotency_key, body)
            return ApiResponse(503, body)

        try:
            approved = await self._verify_with_retry(order_id, amount_cents)
        except CircuitOpenError:
            body = {
                "outcome": "circuit_open",
                "message": "verification dependency unhealthy; try again after recovery",
            }
            self.store.mark_failed_retryable(idempotency_key, body)
            return ApiResponse(503, body)
        except Exception as e:
            self.breaker.record_failure()
            body = {
                "outcome": "upstream_error",
                "message": str(e),
            }
            self.store.mark_failed_retryable(idempotency_key, body)
            return ApiResponse(502, body)

        if not approved.approved:
            # Business decline is terminal success of the *decision*, no money moved
            body = {
                "outcome": "success",
                "refund": None,
                "decision": "declined",
                "reason": approved.reason,
            }
            self.store.mark_success(idempotency_key, body)
            self.breaker.record_success()
            return ApiResponse(200, body)

        refund_id = f"rfnd_{uuid.uuid4().hex[:10]}"
        refund = {
            "refund_id": refund_id,
            "order_id": order_id,
            "amount_cents": amount_cents,
            "status": "issued",
        }
        self.refunds_issued += 1
        body = {"outcome": "success", "refund": refund, "decision": "approved"}
        self.store.mark_success(idempotency_key, body)
        self.breaker.record_success()
        return ApiResponse(200, body)

    async def _verify_with_retry(self, order_id: str, amount_cents: int):
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            if not self.breaker.allow():
                raise CircuitOpenError("circuit open")
            try:
                result = await asyncio.wait_for(
                    self.dependency.verify_refund(order_id, amount_cents),
                    timeout=self.dependency.timeout_sec,
                )
                return result
            except Exception as e:
                last_err = e
                self.breaker.record_failure()
                if attempt < self.max_retries:
                    await asyncio.sleep(self.backoff_sec * (2 ** attempt))
        assert last_err is not None
        raise last_err

    async def health(self) -> ApiResponse:
        """Unaffected by slow dependency — proves no thread-pool collapse."""
        self._inflight_healthy += 1
        await asyncio.sleep(0.001)
        return ApiResponse(
            200,
            {
                "ok": True,
                "circuit": self.breaker.stats().state.value,
                "refunds_issued": self.refunds_issued,
            },
        )
