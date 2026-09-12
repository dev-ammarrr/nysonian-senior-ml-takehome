"""
C1 — Atomic idempotency store (SQLite UNIQUE) for exactly-once semantics.

States:
  in_progress      — lease held while upstream runs
  succeeded        — terminal; duplicates return cached body
  failed_retryable — circuit open / upstream fail; later attempts may reclaim
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class IdemStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED_RETRYABLE = "failed_retryable"


@dataclass
class IdemRecord:
    key: str
    status: IdemStatus
    response_json: Optional[str]
    created_at: float
    updated_at: float


class IdempotencyStore:
    def __init__(self, db_path: str = ":memory:", lease_sec: float = 30.0):
        # check_same_thread=False for concurrent asyncio.to_thread / threads in tests
        self.db_path = db_path
        self.lease_sec = lease_sec
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS idempotency (
                key TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                response_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()
        self._lock = __import__("threading").Lock()

    def close(self):
        self._conn.close()

    def _get(self, key: str) -> Optional[IdemRecord]:
        cur = self._conn.execute(
            "SELECT key, status, response_json, created_at, updated_at FROM idempotency WHERE key=?",
            (key,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return IdemRecord(
            key=row[0],
            status=IdemStatus(row[1]),
            response_json=row[2],
            created_at=row[3],
            updated_at=row[4],
        )

    def try_begin(self, key: str) -> tuple[str, Optional[Dict[str, Any]]]:
        """
        Atomically claim the key for work.

        Returns:
          ("proceed", None) — caller should run upstream
          ("duplicate_success", body) — return cached success
          ("in_progress", None) — another request holds the lease
        """
        now = time.time()
        with self._lock:
            existing = self._get(key)
            if existing is None:
                self._conn.execute(
                    "INSERT INTO idempotency(key, status, response_json, created_at, updated_at) VALUES (?,?,?,?,?)",
                    (key, IdemStatus.IN_PROGRESS.value, None, now, now),
                )
                self._conn.commit()
                return "proceed", None

            if existing.status == IdemStatus.SUCCEEDED:
                body = json.loads(existing.response_json or "{}")
                return "duplicate_success", body

            if existing.status == IdemStatus.IN_PROGRESS:
                # Expired lease → reclaim (crash safety)
                if now - existing.updated_at > self.lease_sec:
                    self._conn.execute(
                        "UPDATE idempotency SET status=?, updated_at=? WHERE key=?",
                        (IdemStatus.IN_PROGRESS.value, now, key),
                    )
                    self._conn.commit()
                    return "proceed", None
                return "in_progress", None

            if existing.status == IdemStatus.FAILED_RETRYABLE:
                self._conn.execute(
                    "UPDATE idempotency SET status=?, response_json=NULL, updated_at=? WHERE key=?",
                    (IdemStatus.IN_PROGRESS.value, now, key),
                )
                self._conn.commit()
                return "proceed", None

            return "in_progress", None

    def mark_success(self, key: str, body: Dict[str, Any]) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "UPDATE idempotency SET status=?, response_json=?, updated_at=? WHERE key=?",
                (IdemStatus.SUCCEEDED.value, json.dumps(body), now, key),
            )
            self._conn.commit()

    def mark_failed_retryable(self, key: str, body: Optional[Dict[str, Any]] = None) -> None:
        """
        Critical for circuit-open coexistence: do NOT treat as terminal success.
        A later attempt after recovery must be able to reclaim the key.
        """
        now = time.time()
        with self._lock:
            self._conn.execute(
                "UPDATE idempotency SET status=?, response_json=?, updated_at=? WHERE key=?",
                (
                    IdemStatus.FAILED_RETRYABLE.value,
                    json.dumps(body or {}),
                    now,
                    key,
                ),
            )
            self._conn.commit()

    def heartbeat(self, key: str) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "UPDATE idempotency SET updated_at=? WHERE key=? AND status=?",
                (now, key, IdemStatus.IN_PROGRESS.value),
            )
            self._conn.commit()
