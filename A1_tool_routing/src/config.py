"""
Configuration module for tool routing system.
Handles environment variables and threshold settings.
"""
import os
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# LLM Provider Configuration
LLM_PROVIDER: Literal["openai", "groq"] = os.getenv("LLM_PROVIDER", "groq")

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

# Confidence Thresholds
HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.85"))
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.60"))

# Feature Flags
ENABLE_LLM_VERIFICATION = os.getenv("ENABLE_LLM_VERIFICATION", "true").lower() == "true"
ENABLE_HUMAN_ESCALATION = os.getenv("ENABLE_HUMAN_ESCALATION", "true").lower() == "true"

# Tool Definitions
TOOL_DEFINITIONS = {
    "refund_damaged_on_arrival": {
        "description": "Issue refund for items damaged during shipping",
        "cost_profile": "high",
        "keywords": ["broken on arrival", "damaged package", "crushed", "shattered on delivery"]
    },
    "warranty_claim": {
        "description": "Process warranty claim for items broken after normal use",
        "cost_profile": "medium",
        "keywords": ["stopped working", "broke after", "warranty", "defective"]
    },
    "return_wrong_item": {
        "description": "Process return for wrong item sent",
        "cost_profile": "medium",
        "keywords": ["wrong item", "incorrect product", "not what I ordered"]
    },
    "decline_customer_damage": {
        "description": "Decline claim for customer-caused damage",
        "cost_profile": "low",
        "keywords": ["dropped it", "my fault", "accidentally"]
    },
    "warehouse_replacement": {
        "description": "Send replacement from warehouse without return",
        "cost_profile": "high",
        "keywords": ["send another", "need replacement urgently"]
    },
    "escalate_to_specialist": {
        "description": "Escalate complex cases to human specialist",
        "cost_profile": "low",
        "keywords": ["speak to manager", "this is unacceptable", "legal action"]
    }
}

TOOL_NAMES = list(TOOL_DEFINITIONS.keys())
