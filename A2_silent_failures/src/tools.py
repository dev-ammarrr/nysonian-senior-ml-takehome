"""
A2 — Tool integrations (healthy vs silently broken).

The broken tool is the failure mode from the assignment:
returns success status, workflow logs "completed", but no real side effect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol
import time
import uuid

from store import SideEffectStore, WarehouseAction


@dataclass
class ToolResult:
    success: bool
    action_id: str
    message: str
    latency_ms: float


class WarehouseTool(Protocol):
    name: str

    def create_return_label(self, order_id: str, sku: str, reason: str) -> ToolResult:
        ...


class HealthyWarehouseTool:
    """Correct integration: reports success AND writes the side effect."""

    name = "healthy_warehouse"

    def __init__(self, store: SideEffectStore):
        self.store = store

    def create_return_label(self, order_id: str, sku: str, reason: str) -> ToolResult:
        start = time.time()
        action_id = f"act_{uuid.uuid4().hex[:12]}"
        self.store.write(
            WarehouseAction(
                action_id=action_id,
                order_id=order_id,
                sku=sku,
                action_type="create_return_label",
                params={"reason": reason},
            )
        )
        return ToolResult(
            success=True,
            action_id=action_id,
            message="return label created",
            latency_ms=round((time.time() - start) * 1000, 2),
        )


class SilentlyBrokenWarehouseTool:
    """
    Returns success without writing to the warehouse store.
    This is the class of bug that can run for months unnoticed.
    """

    name = "silently_broken_warehouse"

    def __init__(self, store: SideEffectStore):
        # Store is present to mirror the healthy interface, but intentionally unused.
        self.store = store

    def create_return_label(self, order_id: str, sku: str, reason: str) -> ToolResult:
        start = time.time()
        action_id = f"act_{uuid.uuid4().hex[:12]}"
        # BUG: no self.store.write(...)
        return ToolResult(
            success=True,
            action_id=action_id,
            message="return label created",
            latency_ms=round((time.time() - start) * 1000, 2),
        )


class WrongParamsWarehouseTool:
    """Writes a side effect but with wrong params (another silent failure mode)."""

    name = "wrong_params_warehouse"

    def __init__(self, store: SideEffectStore):
        self.store = store

    def create_return_label(self, order_id: str, sku: str, reason: str) -> ToolResult:
        start = time.time()
        action_id = f"act_{uuid.uuid4().hex[:12]}"
        self.store.write(
            WarehouseAction(
                action_id=action_id,
                order_id=order_id,
                sku="WRONG_SKU",  # silent corruption
                action_type="create_return_label",
                params={"reason": reason},
            )
        )
        return ToolResult(
            success=True,
            action_id=action_id,
            message="return label created",
            latency_ms=round((time.time() - start) * 1000, 2),
        )
