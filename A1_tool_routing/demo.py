"""
Demonstration of the tool routing system with measurement.

This script demonstrates:
1. Router with confidence-gated escalation
2. Disagreement detection
3. Confidence calibration monitoring
4. Rollout safety metrics
"""
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from router import ToolRouter, RoutingDecision
from measurement import MeasurementSystem
from config import HIGH_CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD, LLM_PROVIDER


def load_test_messages():
    """Load test messages from mock data."""
    with open('data/mock_messages.json', 'r') as f:
        return json.load(f)


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def demo_routing_system():
    """Demonstrate the routing system."""
    print_section("A1: Tool Routing System Demo")
    
    print(f"Configuration:")
    print(f"  LLM Provider: {LLM_PROVIDER}")
    print(f"  High Confidence Threshold: {HIGH_CONFIDENCE_THRESHOLD}")
    print(f"  Low Confidence Threshold: {LOW_CONFIDENCE_THRESHOLD}")
    print()
    
    # Load test data
    test_data = load_test_messages()
    print(f"Loaded {len(test_data)} test messages\n")
    
    # Initialize router (with LLM disabled for fast demo)
    # Set enable_llm=True to test with actual LLM calls
    print("NOTE: Running with LLM disabled for fast demo.")
    print("      Set enable_llm=True in code to test with OpenAI/Groq.\n")
    router = ToolRouter(enable_llm=False)
    
    # Process messages
    print_section("Routing Decisions")
    decisions = []
    
    for i, test_case in enumerate(test_data[:10]):  # Process first 10 for demo
        message = test_case["message"]
        expected = test_case["expected_tool"]
        
        decision = router.route(message, message_id=test_case["id"])
        decisions.append(decision)
        
        # Print decision
        match_symbol = "✓" if decision.tool == expected else "✗"
        print(f"{i+1}. {match_symbol} Message: {message[:60]}...")
        print(f"   Predicted: {decision.tool} (conf: {decision.confidence:.3f})")
        print(f"   Expected:  {expected}")
        print(f"   Path: {decision.decision_path}, Latency: {decision.latency_ms:.1f}ms")
        
        if decision.classifier_prediction != decision.tool:
            print(f"   ⚠ Disagreement: Classifier said {decision.classifier_prediction}")
        print()
    
    # Show metrics
    print_section("Routing Metrics")
    metrics = router.get_metrics()
    print(f"Total Requests: {metrics['total_requests']}")
    print(f"  Classifier Only: {metrics['classifier_only']} ({metrics['classifier_only_pct']}%)")
    print(f"  LLM Verified: {metrics['llm_verified']} ({metrics['llm_verified_pct']}%)")
    print(f"  Human Review: {metrics['human_review']} ({metrics['human_review_pct']}%)")
    
    # Calculate accuracy (using expected_tool as pseudo ground truth)
    ground_truth = {tc["id"]: tc["expected_tool"] for tc in test_data[:10]}
    correct = sum(1 for d in decisions if d.tool == ground_truth[d.message_id])
    accuracy = correct / len(decisions)
    print(f"\nClassifier Accuracy: {accuracy*100:.1f}%")
    
    return router, decisions, ground_truth


def demo_measurement_system(decisions, ground_truth):
    """Demonstrate measurement without ground truth."""
    print_section("Measurement System (Part 2)")
    
    measurement = MeasurementSystem()
    
    # 1. Disagreement detection
    print("1. Disagreement Detection")
    disagreements = measurement.detect_disagreements(decisions)
    disagreement_rate = measurement.compute_disagreement_rate(decisions)
    print(f"   Disagreements found: {len(disagreements)}")
    print(f"   Disagreement rate: {disagreement_rate*100:.2f}%")
    
    if disagreements:
        print("\n   Example disagreement:")
        d = disagreements[0]
        print(f"   Message: {d.message[:60]}...")
        print(f"   Classifier: {d.classifier_tool} (conf: {d.classifier_confidence:.3f})")
        print(f"   LLM: {d.llm_tool}")
        print(f"   Reasoning: {d.llm_reasoning}")
    print()
    
    # 2. Confidence calibration
    print("2. Confidence Calibration Monitoring")
    confidence_buckets = measurement.monitor_confidence_calibration(decisions, ground_truth)
    
    if confidence_buckets:
        print("   Confidence Range | Total | Accuracy")
        print("   " + "-"*40)
        for bucket_label, bucket in sorted(confidence_buckets.items()):
            print(f"   {bucket.confidence_range:>15} | {bucket.total_predictions:>5} | {bucket.accuracy*100:>5.1f}%")
    else:
        print("   (Not enough data for calibration analysis)")
    print()
    
    # 3. Audit sample generation
    print("3. Audit Sample Generation")
    audit_sample = measurement.generate_audit_sample(decisions, sample_size=5, strategy="high_risk")
    print(f"   Generated {len(audit_sample)} high-risk samples for human review")
    print("   Strategy: Prioritize disagreements and low-confidence predictions")
    print()
    
    # Export report
    report_path = "measurement_report.json"
    report = measurement.export_report(report_path, decisions)
    print(f"   Full report exported to: {report_path}")
    
    return measurement


def demo_rollout_safety():
    """Demonstrate rollout safety metrics (Part 3)."""
    print_section("Rollout Safety (Part 3)")
    
    print("Safe Rollout Strategy:")
    print()
    print("Phase 1: Shadow Mode (Week 1)")
    print("  - New router runs alongside old router")
    print("  - Logs decisions but doesn't act")
    print("  - Collect disagreement data")
    print("  - Monitor metrics:")
    print("    • Disagreement rate < 5%")
    print("    • No latency regression (p95 < 2x old router)")
    print("    • LLM failure rate < 1%")
    print()
    
    print("Phase 2: Canary (5% traffic, 3-5 days)")
    print("  - Route 5% of traffic to new router")
    print("  - Auto-rollback triggers:")
    print("    • Correction rate > baseline + 2σ")
    print("    • Cost per message > baseline + 20%")
    print("    • p95 latency > 2 seconds")
    print("    • Human review queue > 10%")
    print()
    
    print("Phase 3: Staged Rollout")
    print("  - 5% → 25% → 50% → 100%")
    print("  - 2-3 days per stage")
    print("  - Monitor same metrics at each stage")
    print("  - Rollback window: 15 minutes")
    print()
    
    print("Key Metrics Dashboard:")
    metrics_example = {
        "disagreement_rate": 0.034,
        "confidence_calibration_error": 0.052,
        "human_review_rate": 0.087,
        "cost_per_message": 0.0023,
        "p50_latency_ms": 31,
        "p95_latency_ms": 487,
        "p99_latency_ms": 1243
    }
    for metric, value in metrics_example.items():
        print(f"  {metric:.<35} {value}")


def main():
    """Run full demonstration."""
    try:
        # Demo routing
        router, decisions, ground_truth = demo_routing_system()
        
        # Demo measurement
        measurement = demo_measurement_system(decisions, ground_truth)
        
        # Demo rollout safety
        demo_rollout_safety()
        
        print_section("Demo Complete")
        print("Next steps:")
        print("1. Set up .env file with API keys")
        print("2. Run with enable_llm=True to test LLM verification")
        print("3. Review measurement_report.json for detailed metrics")
        print("4. See README.md for full documentation and decisions")
        print()
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nIf you see import errors, install dependencies:")
        print("  pip install -r requirements.txt")
        print("\nIf you see LLM errors, check .env configuration")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
