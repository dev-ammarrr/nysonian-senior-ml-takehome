"""
Measurement system for detecting misclassification without ground truth labels.

Implements three complementary signals:
1. Disagreement detection (classifier vs LLM)
2. Confidence calibration monitoring
3. Outcome-based feedback (simulated)

Design Decision: Use multiple weak signals rather than waiting for perfect labels
- Disagreement catches distribution drift and edge cases immediately
- Confidence monitoring catches model degradation
- Outcome tracking provides lagging but high-quality signal
"""
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json
from datetime import datetime

from router import RoutingDecision


@dataclass
class DisagreementRecord:
    """Record when classifier and LLM disagree."""
    message_id: str
    message: str
    classifier_tool: str
    classifier_confidence: float
    llm_tool: str
    llm_reasoning: str
    timestamp: str
    
    def to_dict(self) -> Dict:
        return {
            "message_id": self.message_id,
            "message": self.message,
            "classifier_tool": self.classifier_tool,
            "classifier_confidence": self.classifier_confidence,
            "llm_tool": self.llm_tool,
            "llm_reasoning": self.llm_reasoning,
            "timestamp": self.timestamp
        }


@dataclass
class ConfidenceBucket:
    """Track accuracy within confidence buckets."""
    confidence_range: str  # e.g., "0.80-0.85"
    total_predictions: int
    correct_predictions: int
    accuracy: float


class MeasurementSystem:
    """
    Measure real-world classification quality without labeled ground truth.
    """
    
    def __init__(self):
        self.disagreements: List[DisagreementRecord] = []
        self.confidence_buckets = defaultdict(lambda: {"total": 0, "correct": 0})
        
        # Simulated outcome tracker (in production, this would integrate with actual business outcomes)
        self.outcome_feedback: Dict[str, bool] = {}  # message_id -> was_correct
    
    def detect_disagreements(self, decisions: List[RoutingDecision]) -> List[DisagreementRecord]:
        """
        Detect cases where classifier and LLM disagree.
        
        This is a FREE signal that catches:
        - Edge cases the classifier struggles with
        - Distribution drift
        - Model degradation
        
        In production, disagreements would be:
        1. Logged for analysis
        2. Sampled for human review
        3. Used to retrain/fine-tune classifier
        """
        disagreements = []
        
        for decision in decisions:
            if decision.decision_path == "llm_verified":
                if decision.classifier_prediction != decision.llm_prediction:
                    disagreement = DisagreementRecord(
                        message_id=decision.message_id,
                        message=decision.message,
                        classifier_tool=decision.classifier_prediction,
                        classifier_confidence=decision.classifier_confidence,
                        llm_tool=decision.llm_prediction,
                        llm_reasoning=decision.llm_reasoning or "",
                        timestamp=decision.timestamp
                    )
                    disagreements.append(disagreement)
        
        self.disagreements.extend(disagreements)
        return disagreements
    
    def compute_disagreement_rate(self, decisions: List[RoutingDecision]) -> float:
        """
        Compute disagreement rate as a proxy for misclassification.
        
        Assumption: When classifier and LLM disagree, LLM is usually correct
        (given it has more context and reasoning capability).
        
        High disagreement rate indicates:
        - Classifier needs retraining
        - New types of messages appearing
        - Confidence thresholds need tuning
        """
        llm_verified = [d for d in decisions if d.decision_path == "llm_verified"]
        if not llm_verified:
            return 0.0
        
        disagreed = sum(
            1 for d in llm_verified 
            if d.classifier_prediction != d.llm_prediction
        )
        
        return disagreed / len(llm_verified)
    
    def monitor_confidence_calibration(
        self, 
        decisions: List[RoutingDecision],
        ground_truth: Optional[Dict[str, str]] = None
    ) -> Dict[str, ConfidenceBucket]:
        """
        Monitor if confidence scores are well-calibrated.
        
        Well-calibrated means: predictions with 80% confidence should be correct 80% of the time.
        
        Args:
            decisions: Routing decisions to analyze
            ground_truth: Optional dict of message_id -> correct_tool (from human audits)
        
        Returns:
            Confidence buckets with accuracy metrics
        """
        # Use LLM predictions as pseudo-ground-truth when real labels unavailable
        if ground_truth is None:
            ground_truth = {
                d.message_id: d.llm_prediction 
                for d in decisions 
                if d.llm_prediction is not None
            }
        
        # Bucket predictions by confidence
        buckets = defaultdict(lambda: {"total": 0, "correct": 0})
        
        for decision in decisions:
            if decision.classifier_prediction and decision.classifier_confidence:
                # Define bucket (0.1 width)
                bucket_idx = int(decision.classifier_confidence * 10) / 10
                bucket_label = f"{bucket_idx:.1f}-{bucket_idx+0.1:.1f}"
                
                buckets[bucket_label]["total"] += 1
                
                # Check if correct (if we have ground truth)
                if decision.message_id in ground_truth:
                    correct_tool = ground_truth[decision.message_id]
                    if decision.classifier_prediction == correct_tool:
                        buckets[bucket_label]["correct"] += 1
        
        # Convert to ConfidenceBucket objects
        result = {}
        for bucket_label, counts in sorted(buckets.items()):
            accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
            result[bucket_label] = ConfidenceBucket(
                confidence_range=bucket_label,
                total_predictions=counts["total"],
                correct_predictions=counts["correct"],
                accuracy=accuracy
            )
        
        return result
    
    def simulate_outcome_feedback(
        self, 
        decision: RoutingDecision, 
        was_corrected: bool
    ):
        """
        Simulate downstream outcome feedback.
        
        In production, this would track:
        - Was the action reversed by a human?
        - Did customer escalate/complain after?
        - Did the action fail validation downstream?
        
        This is a LAGGING but HIGH-QUALITY signal (days/weeks delay).
        """
        self.outcome_feedback[decision.message_id] = not was_corrected
    
    def compute_correction_rate(self) -> float:
        """
        Compute rate of actions that needed human correction.
        
        In production, this is the GOLD STANDARD metric, but it's:
        - Lagging (days to weeks)
        - Sparse (only 1-5% of cases get reviewed)
        - Expensive (human time)
        
        Use this to validate disagreement/confidence signals.
        """
        if not self.outcome_feedback:
            return 0.0
        
        incorrect = sum(1 for correct in self.outcome_feedback.values() if not correct)
        return incorrect / len(self.outcome_feedback)
    
    def generate_audit_sample(
        self, 
        decisions: List[RoutingDecision], 
        sample_size: int = 100,
        strategy: str = "stratified"
    ) -> List[RoutingDecision]:
        """
        Generate a sample for human audit.
        
        Strategies:
        - random: Simple random sample
        - stratified: Sample proportional to decision_path
        - high_risk: Focus on low-confidence and disagreements
        
        Design Decision: Use 'high_risk' strategy to maximize value of limited human review.
        """
        if strategy == "random":
            import random
            return random.sample(decisions, min(sample_size, len(decisions)))
        
        elif strategy == "stratified":
            # Sample proportionally from each decision path
            import random
            samples = []
            paths = set(d.decision_path for d in decisions)
            per_path = sample_size // len(paths)
            
            for path in paths:
                path_decisions = [d for d in decisions if d.decision_path == path]
                samples.extend(random.sample(path_decisions, min(per_path, len(path_decisions))))
            
            return samples[:sample_size]
        
        elif strategy == "high_risk":
            # Prioritize:
            # 1. Disagreements (most informative)
            # 2. Low confidence classifier predictions
            # 3. Random sample of rest
            import random
            
            high_priority = [
                d for d in decisions 
                if (d.decision_path == "llm_verified" and 
                    d.classifier_prediction != d.llm_prediction)
            ]
            
            medium_priority = [
                d for d in decisions
                if (d.decision_path == "classifier_only" and 
                    d.classifier_confidence < 0.90)
            ]
            
            low_priority = [
                d for d in decisions
                if d not in high_priority and d not in medium_priority
            ]
            
            sample = []
            sample.extend(high_priority[:sample_size // 2])  # 50% from disagreements
            sample.extend(medium_priority[:(sample_size // 3)])  # 33% from low confidence
            remaining = sample_size - len(sample)
            sample.extend(random.sample(low_priority, min(remaining, len(low_priority))))
            
            return sample[:sample_size]
        
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def export_report(self, filepath: str, decisions: List[RoutingDecision]):
        """Export comprehensive measurement report."""
        disagreements = self.detect_disagreements(decisions)
        disagreement_rate = self.compute_disagreement_rate(decisions)
        confidence_buckets = self.monitor_confidence_calibration(decisions)
        
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "total_decisions": len(decisions),
            "disagreement_rate": round(disagreement_rate, 4),
            "total_disagreements": len(disagreements),
            "confidence_calibration": {
                bucket: {
                    "range": cb.confidence_range,
                    "total": cb.total_predictions,
                    "correct": cb.correct_predictions,
                    "accuracy": round(cb.accuracy, 4)
                }
                for bucket, cb in confidence_buckets.items()
            },
            "disagreements_sample": [d.to_dict() for d in disagreements[:10]],  # First 10
            "correction_rate": round(self.compute_correction_rate(), 4)
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
