"""
Basic tests for the routing system.

In production, this would include:
- Unit tests for each component
- Integration tests for full routing flow
- Load tests for latency/throughput
- Chaos tests for LLM failure handling
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from router import ToolRouter
from classifier import ToolClassifier
from measurement import MeasurementSystem


def test_classifier_basic():
    """Test classifier produces predictions with confidence."""
    classifier = ToolClassifier()
    
    message = "The package arrived completely crushed."
    tool, confidence, scores = classifier.predict(message)
    
    assert tool in ["refund_damaged_on_arrival", "warranty_claim", "return_wrong_item"]
    assert 0.0 <= confidence <= 1.0
    assert len(scores) > 0


def test_router_high_confidence():
    """Test router uses classifier for clear keyword cases."""
    router = ToolRouter(enable_llm=False)

    message = "The package arrived completely crushed and broken on arrival."
    decision = router.route(message)

    assert decision.decision_path == "classifier_only"
    assert decision.tool == "refund_damaged_on_arrival"
    assert decision.confidence > 0.0


def test_router_metrics():
    """Test router tracks metrics correctly."""
    router = ToolRouter(enable_llm=False)
    
    messages = [
        "Package arrived broken",
        "Item stopped working after 3 months",
        "Wrong item sent"
    ]
    
    for msg in messages:
        router.route(msg)
    
    metrics = router.get_metrics()
    assert metrics["total_requests"] == 3
    assert metrics["classifier_only"] + metrics["llm_verified"] + metrics["human_review"] == 3


def test_measurement_disagreement():
    """Test disagreement detection."""
    measurement = MeasurementSystem()
    
    # Create mock decisions with disagreements
    from router import RoutingDecision
    
    decisions = [
        RoutingDecision(
            message_id="1",
            message="test",
            tool="refund_damaged_on_arrival",
            confidence=0.8,
            decision_path="llm_verified",
            classifier_prediction="warranty_claim",
            llm_prediction="refund_damaged_on_arrival"
        )
    ]
    
    disagreements = measurement.detect_disagreements(decisions)
    assert len(disagreements) == 1
    assert disagreements[0].classifier_tool != disagreements[0].llm_tool


def test_confidence_thresholds():
    """Test confidence thresholds work correctly."""
    from config import HIGH_CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD
    
    assert LOW_CONFIDENCE_THRESHOLD < HIGH_CONFIDENCE_THRESHOLD
    assert 0.0 <= LOW_CONFIDENCE_THRESHOLD <= 1.0
    assert 0.0 <= HIGH_CONFIDENCE_THRESHOLD <= 1.0


if __name__ == "__main__":
    print("Running tests...")
    test_classifier_basic()
    print("✓ test_classifier_basic")
    
    test_router_high_confidence()
    print("✓ test_router_high_confidence")
    
    test_router_metrics()
    print("✓ test_router_metrics")
    
    test_measurement_disagreement()
    print("✓ test_measurement_disagreement")
    
    test_confidence_thresholds()
    print("✓ test_confidence_thresholds")
    
    print("\nAll tests passed! ✓")
