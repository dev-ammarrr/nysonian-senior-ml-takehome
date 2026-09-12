"""
A2 — Detection layer: verified-effect monitoring + canaries + gap stats.

Senior framing:
- Never trust the calling workflow's "step completed" log.
- Independently verify the effect showed up where it should.
- Run synthetic canaries through the real pipeline continuously.
- Alert on gap between reported_success_rate and verified_success_rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from collections import deque
from typing import Deque, Dict, List, Optional, Any
from datetime import datetime
import time
import json

from store import SideEffectStore
from tools import ToolResult, WarehouseTool


@dataclass
class VerificationRecord:
    action_id: str
    order_id: str
    expected_sku: str
    reported_success: bool
    verified: bool
    failure_reason: Optional[str]
    is_canary: bool
    tool_name: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict:
        return asdict(self)


class VerifiedEffectMonitor:
    """
    After a tool claims success, check the side-effect store independently.

    Checks:
    1. Record exists for action_id
    2. order_id matches
    3. sku matches expected (catches wrong-params silent bugs)
    """

    def __init__(self, store: SideEffectStore, window_size: int = 500):
        self.store = store
        self.records: Deque[VerificationRecord] = deque(maxlen=window_size)
        self.alerts: List[Dict[str, Any]] = []

    def verify(
        self,
        result: ToolResult,
        order_id: str,
        expected_sku: str,
        tool_name: str,
        is_canary: bool = False,
    ) -> VerificationRecord:
        failure_reason = None
        verified = False

        if not result.success:
            failure_reason = "tool_reported_failure"
        else:
            action = self.store.get(result.action_id)
            if action is None:
                failure_reason = "missing_side_effect"
            elif action.order_id != order_id:
                failure_reason = "order_id_mismatch"
            elif action.sku != expected_sku:
                failure_reason = "sku_mismatch"
            elif action.action_type != "create_return_label":
                failure_reason = "action_type_mismatch"
            else:
                verified = True

        record = VerificationRecord(
            action_id=result.action_id,
            order_id=order_id,
            expected_sku=expected_sku,
            reported_success=result.success,
            verified=verified,
            failure_reason=failure_reason,
            is_canary=is_canary,
            tool_name=tool_name,
        )
        self.records.append(record)

        # Alert on silent failure: reported success but not verified
        if result.success and not verified:
            alert = {
                "severity": datetime.utcnow().isoformat(),
                "type": "silent_failure",
                "action_id": result.action_id,
                "order_id": order_id,
                "tool_name": tool_name,
                "reason": failure_reason,
                "is_canary": is_canary,
            }
            self.alerts.append(alert)

        return record

    def gap_stats(self) -> Dict[str, float]:
        """
        Statistical monitoring: reported success vs independently verified success.
        A persistent gap is the smoking gun for silent failures.
        """
        if not self.records:
            return {
                "n": 0,
                "reported_success_rate": 0.0,
                "verified_success_rate": 0.0,
                "gap": 0.0,
            }

        n = len(self.records)
        reported = sum(1 for r in self.records if r.reported_success) / n
        verified = sum(1 for r in self.records if r.verified) / n
        return {
            "n": n,
            "reported_success_rate": round(reported, 4),
            "verified_success_rate": round(verified, 4),
            "gap": round(reported - verified, 4),
            "silent_failures": sum(
                1 for r in self.records if r.reported_success and not r.verified
            ),
        }

    def should_page(self, gap_threshold: float = 0.02, min_samples: int = 20) -> bool:
        """
        Tune false alarms via gap_threshold + min_samples.
        Too low → noise; too high → months of silent damage.
        """
        stats = self.gap_stats()
        return stats["n"] >= min_samples and stats["gap"] >= gap_threshold


class CanaryRunner:
    """
    Continuously send synthetic transactions through the real tool path.
    Known expected outcomes → immediate detection without waiting for customers.
    """

    def __init__(self, tool: WarehouseTool, monitor: VerifiedEffectMonitor):
        self.tool = tool
        self.monitor = monitor
        self.canary_count = 0

    def run_once(self) -> VerificationRecord:
        self.canary_count += 1
        order_id = f"canary_order_{self.canary_count}"
        sku = f"CANARY_SKU_{self.canary_count}"
        result = self.tool.create_return_label(order_id, sku, reason="canary")
        return self.monitor.verify(
            result=result,
            order_id=order_id,
            expected_sku=sku,
            tool_name=self.tool.name,
            is_canary=True,
        )

    def run_n(self, n: int) -> List[VerificationRecord]:
        return [self.run_once() for _ in range(n)]


class AgentWorkflow:
    """
    Minimal agent step that "successfully completes" based on tool return value —
    exactly the blind spot we are detecting.
    """

    def __init__(self, tool: WarehouseTool, monitor: VerifiedEffectMonitor):
        self.tool = tool
        self.monitor = monitor
        self.workflow_log: List[Dict] = []

    def handle_return_request(self, order_id: str, sku: str, reason: str) -> Dict:
        result = self.tool.create_return_label(order_id, sku, reason)
        # Blind logging — what production often does
        self.workflow_log.append(
            {
                "step": "create_return_label",
                "status": "completed" if result.success else "failed",
                "action_id": result.action_id,
            }
        )
        verification = self.monitor.verify(
            result=result,
            order_id=order_id,
            expected_sku=sku,
            tool_name=self.tool.name,
            is_canary=False,
        )
        return {
            "workflow_status": "completed" if result.success else "failed",
            "verified": verification.verified,
            "failure_reason": verification.failure_reason,
            "action_id": result.action_id,
        }


def export_report(monitor: VerifiedEffectMonitor, path: str) -> Dict:
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "gap_stats": monitor.gap_stats(),
        "should_page": monitor.should_page(),
        "alerts": monitor.alerts[-50:],
        "recent_failures": [
            r.to_dict()
            for r in list(monitor.records)[-50:]
            if r.reported_success and not r.verified
        ],
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return report
