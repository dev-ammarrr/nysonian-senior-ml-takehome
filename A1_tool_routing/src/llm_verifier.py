"""
LLM verification layer for ambiguous cases.

Supports both OpenAI and Groq with structured output.

Design Decision: Use structured output (JSON mode) instead of parsing free text because:
1. More reliable at scale (no parsing failures)
2. Forces the LLM to commit to a tool choice
3. Can extract reasoning for debugging/auditing
4. Better latency (shorter outputs)
"""
import json
from typing import Dict, Optional, Literal
from openai import OpenAI
from groq import Groq

from config import (
    LLM_PROVIDER, 
    OPENAI_API_KEY, 
    OPENAI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    TOOL_DEFINITIONS,
    TOOL_NAMES
)


class LLMVerifier:
    """
    LLM-based verification for ambiguous tool routing decisions.
    Supports OpenAI and Groq with provider switching via env variable.
    """
    
    def __init__(self, provider: Optional[Literal["openai", "groq"]] = None):
        self.provider = provider or LLM_PROVIDER
        
        if self.provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set in environment")
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_MODEL
        elif self.provider == "groq":
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not set in environment")
            self.client = Groq(api_key=GROQ_API_KEY)
            self.model = GROQ_MODEL
        else:
            raise ValueError(f"Unknown provider: {self.provider}. Must be 'openai' or 'groq'")
        
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with tool definitions."""
        tools_desc = "\n".join([
            f"- {tool}: {TOOL_DEFINITIONS[tool]['description']}"
            for tool in TOOL_NAMES
        ])
        
        return f"""You are a customer service routing assistant. Given a customer message, select the most appropriate tool to handle their issue.

Available tools:
{tools_desc}

Rules:
1. If the item was damaged/broken ON ARRIVAL (during shipping), use refund_damaged_on_arrival
2. If it broke AFTER normal use, use warranty_claim
3. If they received the WRONG item, use return_wrong_item
4. If damage was customer's fault, use decline_customer_damage
5. If they're angry or threatening legal action, use escalate_to_specialist
6. When in doubt between options, prefer the tool that's safer for customer satisfaction

Respond ONLY with valid JSON in this format:
{{
    "tool": "<tool_name>",
    "confidence": <0.0 to 1.0>,
    "reasoning": "<brief explanation>"
}}"""
    
    def verify(self, message: str, classifier_prediction: Optional[str] = None) -> Dict:
        """
        Verify tool selection using LLM.
        
        Args:
            message: Customer message
            classifier_prediction: Optional prediction from classifier (for comparison)
        
        Returns:
            Dict with keys: tool, confidence, reasoning, latency_ms
        """
        import time
        start_time = time.time()
        
        user_prompt = f"Customer message: \"{message}\""
        if classifier_prediction:
            user_prompt += f"\n\nNote: A lightweight classifier predicted '{classifier_prediction}'. Verify if this is correct."
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,  # Low temperature for consistency
                    max_tokens=150
                )
                result_text = response.choices[0].message.content
            
            elif self.provider == "groq":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=150
                )
                result_text = response.choices[0].message.content
            
            # Parse response
            result = json.loads(result_text)
            
            # Validate tool name
            if result["tool"] not in TOOL_NAMES:
                raise ValueError(f"Invalid tool name: {result['tool']}")
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "tool": result["tool"],
                "confidence": float(result.get("confidence", 0.9)),
                "reasoning": result.get("reasoning", ""),
                "latency_ms": round(latency_ms, 2),
                "provider": self.provider,
                "model": self.model
            }
        
        except Exception as e:
            # Fallback on LLM failure
            latency_ms = (time.time() - start_time) * 1000
            return {
                "tool": classifier_prediction or "escalate_to_specialist",
                "confidence": 0.3,
                "reasoning": f"LLM verification failed: {str(e)}",
                "latency_ms": round(latency_ms, 2),
                "provider": self.provider,
                "model": self.model,
                "error": str(e)
            }
    
    def batch_verify(self, messages: list[str]) -> list[Dict]:
        """Verify multiple messages (sequential for simplicity)."""
        return [self.verify(msg) for msg in messages]
