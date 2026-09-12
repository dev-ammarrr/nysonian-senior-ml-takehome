"""
A2 — Side-effect store (source of truth independent of tool return values).

In production this would be: warehouse DB, payment ledger, CRM ticket store, etc.
The monitor never trusts the tool's HTTP/status response — it reads this store.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Optional
import time


@dataclass
class WarehouseAction:
    action_id: str
    order_id: str
    sku: str
    action_type: str  # e.g. "create_return_label"
    params: Dict
    created_at: float = field(default_factory=time.time)


class SideEffectStore:
    """Independent system of record for warehouse actions."""

    def __init__(self):
        self._actions: Dict[str, WarehouseAction] = {}
        self._lock = Lock()

    def write(self, action: WarehouseAction) -> None:
        with self._lock:
            self._actions[action.action_id] = action

    def get(self, action_id: str) -> Optional[WarehouseAction]:
        with self._lock:
            return self._actions.get(action_id)

    def exists(self, action_id: str) -> bool:
        return self.get(action_id) is not None

    def count(self) -> int:
        with self._lock:
            return len(self._actions)

    def clear(self) -> None:
        with self._lock:
            self._actions.clear()
