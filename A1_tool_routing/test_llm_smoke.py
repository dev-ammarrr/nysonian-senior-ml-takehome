"""
Quick LLM-path smoke test for A1.
Uses OPENAI/GROQ from .env. Forces mid-confidence messages into LLM verification.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from router import ToolRouter
from config import LLM_PROVIDER, HIGH_CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD


AMBIGUOUS = [
    "Something's wrong with my order, the thing doesn't seem right.",
    "It broke and I'm not sure if it was already like that.",
    "Please help with this product issue ASAP.",
]


def main():
    print(f"Provider={LLM_PROVIDER} thresholds=[{LOW_CONFIDENCE_THRESHOLD}, {HIGH_CONFIDENCE_THRESHOLD}]")
    router = ToolRouter(enable_llm=True)

    # Temporarily lower high threshold so clear cases can still hit classifier,
    # but force a couple of soft messages through LLM by monkey-lowering confidence path.
    for msg in AMBIGUOUS:
        # Call LLM verifier directly to prove provider works
        result = router.llm_verifier.verify(msg)
        print(f"LLM: tool={result['tool']} conf={result['confidence']:.2f} "
              f"latency={result['latency_ms']:.0f}ms reason={result.get('reasoning','')[:80]}")
        if "error" in result:
            raise SystemExit(f"LLM failed: {result['error']}")

    # Full route path on a clear + ambiguous mix
    decisions = router.route_batch([
        "The package arrived completely crushed.",
        "Something is off with this order, please check.",
    ])
    for d in decisions:
        print(f"ROUTE path={d.decision_path} tool={d.tool} conf={d.confidence:.3f}")

    print("A1 LLM smoke test OK")


if __name__ == "__main__":
    main()
