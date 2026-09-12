"""
Main routing system with confidence-gated escalation.

This implements the core architecture:
    Message → Classifier → (if confident) Execute
                         → (if uncertain) LLM Verify
                         → (if very uncertain) Human Review

Design Decision: Single entry point with explicit escalation paths
- Makes monitoring easier (all decisions logged at one layer)
- Clear SLA boundaries (fast path vs slow path)
- Graceful degradation (can disable LLM if it's down)
"""
from typing import Dict, Optional, Literal
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from classifier import ToolClassifier
from llm_verifier import LLMVerifier
from config import (
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    ENABLE_LLM_VERIFICATION,
    ENABLE_HUMAN_ESCALATION
)


@dataclass
class RoutingDecision:
    """Structured routing decision with full audit trail."""
    message_id: str
    message: str
    tool: str
    confidence: float
    decision_path: Literal["classifier_only", "llm_verified", "human_review"]
    classifier_prediction: Optional[str] = None
    classifier_confidence: Optional[float] = None
    llm_prediction: Optional[str] = None
    llm_reasoning: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: str = None
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ToolRouter:
    """
    Main routing system with confidence-based escalation.
    """
    
    def __init__(self, enable_llm: bool = ENABLE_LLM_VERIFICATION):
        self.classifier = ToolClassifier()
        self.llm_verifier = LLMVerifier() if enable_llm else None
        self.enable_llm = enable_llm
        
        # Thresholds
        self.high_threshold = HIGH_CONFIDENCE_THRESHOLD
        self.low_threshold = LOW_CONFIDENCE_THRESHOLD
        
        # Metrics (in production, this would be sent to monitoring system)
        self.metrics = {
            "total_requests": 0,
            "classifier_only": 0,
            "llm_verified": 0,
            "human_review": 0,
            "llm_failures": 0
        }
    
    def route(self, message: str, message_id: Optional[str] = None) -> RoutingDecision:
        """
        Route a customer message to the appropriate tool.
        
        Decision flow:
        1. Classifier predicts tool + confidence
        2. If confidence >= HIGH_THRESHOLD: use classifier prediction
        3. Elif confidence >= LOW_THRESHOLD: escalate to LLM verification
        4. Else: escalate to human review
        """
        import time
        start_time = time.time()
        
        if message_id is None:
            message_id = f"msg_{int(time.time() * 1000)}"
        
        self.metrics["total_requests"] += 1
        
        # Step 1: Classifier prediction
        classifier_tool, classifier_conf, all_scores = self.classifier.predict(message)
        
        # Step 2: Decision gate
        if classifier_conf >= self.high_threshold:
            # High confidence: use classifier directly
            decision = RoutingDecision(
                message_id=message_id,
                message=message,
                tool=classifier_tool,
                confidence=classifier_conf,
                decision_path="classifier_only",
                classifier_prediction=classifier_tool,
                classifier_confidence=classifier_conf,
                latency_ms=round((time.time() - start_time) * 1000, 2),
                metadata={"all_scores": all_scores}
            )
            self.metrics["classifier_only"] += 1

        elif classifier_conf >= self.low_threshold:
            # Medium confidence: verify with LLM when available
            if self.enable_llm and self.llm_verifier:
                llm_result = self.llm_verifier.verify(message, classifier_tool)

                if "error" in llm_result:
                    self.metrics["llm_failures"] += 1
                    # Fallback to classifier on LLM failure
                    decision = RoutingDecision(
                        message_id=message_id,
                        message=message,
                        tool=classifier_tool,
                        confidence=classifier_conf,
                        decision_path="classifier_only",
                        classifier_prediction=classifier_tool,
                        classifier_confidence=classifier_conf,
                        latency_ms=round((time.time() - start_time) * 1000, 2),
                        metadata={"llm_error": llm_result["error"], "all_scores": all_scores}
                    )
                    self.metrics["classifier_only"] += 1
                else:
                    decision = RoutingDecision(
                        message_id=message_id,
                        message=message,
                        tool=llm_result["tool"],
                        confidence=llm_result["confidence"],
                        decision_path="llm_verified",
                        classifier_prediction=classifier_tool,
                        classifier_confidence=classifier_conf,
                        llm_prediction=llm_result["tool"],
                        llm_reasoning=llm_result["reasoning"],
                        latency_ms=round((time.time() - start_time) * 1000, 2),
                        metadata={
                            "llm_latency_ms": llm_result["latency_ms"],
                            "llm_provider": llm_result["provider"],
                            "all_scores": all_scores
                        }
                    )
                    self.metrics["llm_verified"] += 1
            else:
                # LLM disabled: use classifier rather than dumping everything to human
                decision = RoutingDecision(
                    message_id=message_id,
                    message=message,
                    tool=classifier_tool,
                    confidence=classifier_conf,
                    decision_path="classifier_only",
                    classifier_prediction=classifier_tool,
                    classifier_confidence=classifier_conf,
                    latency_ms=round((time.time() - start_time) * 1000, 2),
                    metadata={"llm_disabled": True, "all_scores": all_scores}
                )
                self.metrics["classifier_only"] += 1

        else:
            # Low confidence: human review
            decision = RoutingDecision(
                message_id=message_id,
                message=message,
                tool="escalate_to_specialist",
                confidence=classifier_conf,
                decision_path="human_review",
                classifier_prediction=classifier_tool,
                classifier_confidence=classifier_conf,
                latency_ms=round((time.time() - start_time) * 1000, 2),
                metadata={"reason": "confidence_too_low", "all_scores": all_scores}
            )
            self.metrics["human_review"] += 1

        return decision
    
    def route_batch(self, messages: list[str]) -> list[RoutingDecision]:
        """Route multiple messages."""
        return [self.route(msg, f"msg_{i}") for i, msg in enumerate(messages)]
    
    def get_metrics(self) -> Dict:
        """Get current routing metrics."""
        total = self.metrics["total_requests"]
        if total == 0:
            return self.metrics
        
        return {
            **self.metrics,
            "classifier_only_pct": round(100 * self.metrics["classifier_only"] / total, 2),
            "llm_verified_pct": round(100 * self.metrics["llm_verified"] / total, 2),
            "human_review_pct": round(100 * self.metrics["human_review"] / total, 2),
            "llm_failure_rate": round(100 * self.metrics["llm_failures"] / total, 2) if self.enable_llm else 0
        }
    
    def export_decision_log(self, decisions: list[RoutingDecision], filepath: str):
        """Export decisions to JSON for analysis."""
        with open(filepath, 'w') as f:
            json.dump([d.to_dict() for d in decisions], f, indent=2)
