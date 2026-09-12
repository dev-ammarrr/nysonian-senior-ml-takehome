"""
A2 demo: show a silently broken tool looking healthy, then detection catching it.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from store import SideEffectStore
from tools import (
    HealthyWarehouseTool,
    SilentlyBrokenWarehouseTool,
    WrongParamsWarehouseTool,
)
from monitor import AgentWorkflow, CanaryRunner, VerifiedEffectMonitor, export_report


def section(title: str):
    print(f"\n{'='*72}\n  {title}\n{'='*72}\n")


def run_traffic(workflow: AgentWorkflow, n: int, prefix: str):
    for i in range(n):
        workflow.handle_return_request(
            order_id=f"{prefix}_order_{i}",
            sku=f"SKU_{i % 5}",
            reason="customer_return",
        )


def main():
    section("A2: Silent Failure Detection Demo")

    store = SideEffectStore()
    monitor = VerifiedEffectMonitor(store, window_size=1000)

    # --- Healthy baseline ---
    section("1) Healthy tool (baseline)")
    healthy = HealthyWarehouseTool(store)
    healthy_wf = AgentWorkflow(healthy, monitor)
    run_traffic(healthy_wf, 30, "healthy")
    canary_h = CanaryRunner(healthy, monitor)
    canary_h.run_n(10)
    stats = monitor.gap_stats()
    print(f"Gap stats after healthy traffic: {stats}")
    print(f"Should page? {monitor.should_page(min_samples=20)}")
    assert stats["gap"] == 0.0
    assert not monitor.should_page(min_samples=20)
    print("✓ Healthy path: reported == verified, no page")

    # --- Silently broken ---
    section("2) Silently broken tool (returns success, writes nothing)")
    broken = SilentlyBrokenWarehouseTool(store)
    broken_wf = AgentWorkflow(broken, monitor)
    before_alerts = len(monitor.alerts)
    run_traffic(broken_wf, 25, "broken")
    canary_b = CanaryRunner(broken, monitor)
    canary_results = canary_b.run_n(10)

    silent = [r for r in canary_results if r.reported_success and not r.verified]
    print(f"Canary silent failures caught: {len(silent)}/{len(canary_results)}")
    print(f"Example alert: {monitor.alerts[-1] if monitor.alerts else None}")
    stats = monitor.gap_stats()
    print(f"Gap stats after broken traffic: {stats}")
    print(f"Should page? {monitor.should_page(min_samples=20)}")
    assert len(silent) == 10
    assert all(r.failure_reason == "missing_side_effect" for r in silent)
    assert monitor.should_page(min_samples=20)
    assert len(monitor.alerts) > before_alerts
    print("✓ Broken tool caught via verified-effect + canaries")

    # --- Wrong params ---
    section("3) Wrong-params tool (writes, but corrupted SKU)")
    store2 = SideEffectStore()
    monitor2 = VerifiedEffectMonitor(store2)
    wrong = WrongParamsWarehouseTool(store2)
    wf2 = AgentWorkflow(wrong, monitor2)
    out = wf2.handle_return_request("ord_x", "SKU_REAL", "return")
    print(f"Workflow status (blind): {out['workflow_status']}")
    print(f"Verified: {out['verified']} reason={out['failure_reason']}")
    assert out["workflow_status"] == "completed"
    assert out["verified"] is False
    assert out["failure_reason"] == "sku_mismatch"
    print("✓ Param corruption caught (workflow still thought it succeeded)")

    section("4) False-alarm tradeoff knobs")
    print("gap_threshold=0.02, min_samples=20  → page when gap real and stable")
    print("If tuned wrong (threshold too low): noise from rare race/replication lag")
    print("If tuned wrong (threshold too high): silent failures burn for weeks")
    print("How you'd know: canary failure rate stays high while alerts stay quiet")

    report = export_report(monitor, "detection_report.json")
    print(f"\nReport written to detection_report.json (gap={report['gap_stats']['gap']})")
    section("Demo Complete — silent failures caught without reading transcripts")


if __name__ == "__main__":
    main()
