"""
C1 — Circuit breaker for a slow/flaky upstream dependency.

States: closed → open → half_open → closed
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitStats:
    state: CircuitState
    failures: int
    successes: int
    opened_at: float | None


class CircuitOpenError(Exception):
    pass


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_sec: float = 2.0,
        half_open_successes: int = 1,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self.half_open_successes = half_open_successes
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._half_open_successes = 0
        self._opened_at: float | None = None
        self._lock = Lock()

    def allow(self) -> bool:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if (
                    self._opened_at is not None
                    and time.time() - self._opened_at >= self.recovery_timeout_sec
                ):
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_successes = 0
                    return True
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.half_open_successes:
                    self._state = CircuitState.CLOSED
                    self._failures = 0
                    self._opened_at = None
            else:
                self._failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = time.time()
            elif self._failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.time()

    def stats(self) -> CircuitStats:
        with self._lock:
            return CircuitStats(
                state=self._state,
                failures=self._failures,
                successes=self._half_open_successes,
                opened_at=self._opened_at,
            )
