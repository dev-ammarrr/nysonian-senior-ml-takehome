import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from store import SideEffectStore
from tools import HealthyWarehouseTool, SilentlyBrokenWarehouseTool, WrongParamsWarehouseTool
from monitor import AgentWorkflow, CanaryRunner, VerifiedEffectMonitor


def test_healthy_verified():
    store = SideEffectStore()
    monitor = VerifiedEffectMonitor(store)
    wf = AgentWorkflow(HealthyWarehouseTool(store), monitor)
    out = wf.handle_return_request("o1", "SKU1", "r")
    assert out["workflow_status"] == "completed"
    assert out["verified"] is True
    assert monitor.gap_stats()["gap"] == 0.0


def test_silent_break_caught():
    store = SideEffectStore()
    monitor = VerifiedEffectMonitor(store)
    tool = SilentlyBrokenWarehouseTool(store)
    wf = AgentWorkflow(tool, monitor)
    out = wf.handle_return_request("o2", "SKU2", "r")
    assert out["workflow_status"] == "completed"  # looks healthy
    assert out["verified"] is False
    assert out["failure_reason"] == "missing_side_effect"
    assert len(monitor.alerts) == 1


def test_canary_detects_broken_tool():
    store = SideEffectStore()
    monitor = VerifiedEffectMonitor(store)
    canary = CanaryRunner(SilentlyBrokenWarehouseTool(store), monitor)
    results = canary.run_n(5)
    assert all(r.reported_success and not r.verified for r in results)
    assert monitor.should_page(gap_threshold=0.02, min_samples=5)


def test_wrong_params_caught():
    store = SideEffectStore()
    monitor = VerifiedEffectMonitor(store)
    wf = AgentWorkflow(WrongParamsWarehouseTool(store), monitor)
    out = wf.handle_return_request("o3", "EXPECTED", "r")
    assert out["verified"] is False
    assert out["failure_reason"] == "sku_mismatch"


def test_gap_opens_only_after_failures():
    store = SideEffectStore()
    monitor = VerifiedEffectMonitor(store)
    healthy = AgentWorkflow(HealthyWarehouseTool(store), monitor)
    for i in range(20):
        healthy.handle_return_request(f"h{i}", "S", "r")
    assert not monitor.should_page(min_samples=20)

    broken = AgentWorkflow(SilentlyBrokenWarehouseTool(store), monitor)
    for i in range(20):
        broken.handle_return_request(f"b{i}", "S", "r")
    assert monitor.should_page(min_samples=20)
    assert monitor.gap_stats()["gap"] > 0.3


if __name__ == "__main__":
    test_healthy_verified()
    print("✓ test_healthy_verified")
    test_silent_break_caught()
    print("✓ test_silent_break_caught")
    test_canary_detects_broken_tool()
    print("✓ test_canary_detects_broken_tool")
    test_wrong_params_caught()
    print("✓ test_wrong_params_caught")
    test_gap_opens_only_after_failures()
    print("✓ test_gap_opens_only_after_failures")
    print("\nAll A2 tests passed!")
