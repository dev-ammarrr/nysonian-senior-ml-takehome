"""
C1 — Slow / flaky verification dependency (stand-in for LLM or KYC API).
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class VerifyResult:
    approved: bool
    reason: str


class VerificationDependency:
    """
    Configurable slow dependency for tests.
    """

    def __init__(
        self,
        delay_sec: float = 0.05,
        fail_rate: float = 0.0,
        timeout_sec: float = 2.0,
        force_fail: bool = False,
        hang: bool = False,
    ):
        self.delay_sec = delay_sec
        self.fail_rate = fail_rate
        self.timeout_sec = timeout_sec
        self.force_fail = force_fail
        self.hang = hang
        self.call_count = 0

    async def verify_refund(self, order_id: str, amount_cents: int) -> VerifyResult:
        self.call_count += 1
        if self.hang:
            await asyncio.sleep(self.timeout_sec + 5)
        await asyncio.sleep(self.delay_sec)
        if self.force_fail or random.random() < self.fail_rate:
            raise TimeoutError(f"verification timeout for {order_id}")
        if amount_cents > 500_00:
            return VerifyResult(False, "amount_too_high")
        return VerifyResult(True, "ok")
