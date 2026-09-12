"""
Local classifier for tool routing with confidence scoring.

Uses TF-IDF + cosine similarity (sklearn) combined with keyword matching.

Design Decision: sklearn TF-IDF instead of sentence-transformers/BERT because:
1. Zero-shot from tool descriptions — no labeled training data needed
2. ~5ms inference on CPU, no GPU / torch install required
3. Easy to iterate on tool definitions without retraining
4. Reliable dependency story for assessment + production cold-start
5. Can swap for embeddings later once we have labeled data to justify the ops cost

At 50k msgs/day, TF-IDF is enough to shrink LLM volume; the escalation gate
matters more than classifier sophistication.
"""
import numpy as np
from typing import Dict, Tuple, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from config import TOOL_DEFINITIONS, TOOL_NAMES


# Extra exemplars so TF-IDF sees real customer phrasing, not just tool descriptions.
# In production these would come from disagreement logs / audited outcomes.
TOOL_EXEMPLARS = {
    "refund_damaged_on_arrival": [
        "package arrived completely crushed item shattered",
        "broken on arrival damaged during shipping",
        "opened the box item already damaged",
        "delivery driver dropped package item broken",
        "arrived crushed shattered on delivery",
    ],
    "warranty_claim": [
        "used for months then stopped working",
        "broke after normal use defective",
        "product stopped working still under warranty",
        "working fine then stopped after weeks",
        "broke after I used it seems defective",
    ],
    "return_wrong_item": [
        "not what I ordered wrong item sent",
        "sent me blue but I ordered red",
        "received someone else's order",
        "incorrect product wrong color",
        "not what was shown on website",
    ],
    "decline_customer_damage": [
        "I accidentally dropped it screen cracked",
        "my fault spilled water broke it",
        "tried to fix it myself now broken",
        "I dropped it is this covered",
        "customer caused damage accidentally",
    ],
    "warehouse_replacement": [
        "need replacement urgently send another",
        "send another one right away ASAP",
        "need this replaced urgently warehouse",
        "send replacement without return urgently",
    ],
    "escalate_to_specialist": [
        "speak to manager this is unacceptable",
        "contacting my lawyer legal action",
        "filing complaint with consumer protection",
        "extremely disappointed want manager",
    ],
}


class ToolClassifier:
    """
    Lightweight classifier that combines TF-IDF similarity with keyword matching.
    Returns both prediction and calibrated confidence score.
    """

    def __init__(self):
        self.tool_corpus = []
        self.corpus_tool_index: List[str] = []

        for tool in TOOL_NAMES:
            base = (
                f"{TOOL_DEFINITIONS[tool]['description']} "
                f"{' '.join(TOOL_DEFINITIONS[tool]['keywords'])}"
            )
            exemplars = TOOL_EXEMPLARS.get(tool, [])
            for text in [base] + exemplars:
                self.tool_corpus.append(text)
                self.corpus_tool_index.append(tool)

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
            sublinear_tf=True,
        )
        self.tool_vectors = self.vectorizer.fit_transform(self.tool_corpus)
        self.keyword_patterns = self._compile_keyword_patterns()

    def _compile_keyword_patterns(self) -> Dict[str, List[re.Pattern]]:
        patterns = {}
        for tool in TOOL_NAMES:
            keywords = TOOL_DEFINITIONS[tool]["keywords"]
            patterns[tool] = [
                re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
                for kw in keywords
            ]
        return patterns

    def _keyword_match_score(self, message: str) -> Dict[str, float]:
        scores = {tool: 0.0 for tool in TOOL_NAMES}
        for tool in TOOL_NAMES:
            match_count = sum(
                1 for pattern in self.keyword_patterns[tool] if pattern.search(message)
            )
            if match_count > 0:
                # Strong boost for any keyword hit — these are high-precision phrases
                scores[tool] = min(0.55 + 0.25 * match_count, 1.0)
        return scores

    def _semantic_similarity_score(self, message: str) -> Dict[str, float]:
        message_vector = self.vectorizer.transform([message])
        similarities = cosine_similarity(message_vector, self.tool_vectors)[0]

        # Max similarity across all exemplars for each tool
        scores = {tool: 0.0 for tool in TOOL_NAMES}
        for sim, tool in zip(similarities, self.corpus_tool_index):
            scores[tool] = max(scores[tool], float(sim))
        return scores

    def predict(self, message: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict tool with confidence score.

        Returns:
            tool_name, confidence [0,1], all_scores
        """
        keyword_scores = self._keyword_match_score(message)
        semantic_scores = self._semantic_similarity_score(message)

        # Keywords are high-precision; TF-IDF covers paraphrase cases
        combined_scores = {
            tool: 0.45 * semantic_scores[tool] + 0.55 * keyword_scores[tool]
            for tool in TOOL_NAMES
        }

        top_tool = max(combined_scores, key=combined_scores.get)
        top_score = combined_scores[top_tool]

        sorted_scores = sorted(combined_scores.values(), reverse=True)
        second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        margin = top_score - second_score

        # Keyword hits get a modest bump (precision signal) without saturating to 1.0
        keyword_boost = 0.08 if keyword_scores[top_tool] >= 0.55 else 0.0

        # Calibrate for TF-IDF score ranges (typically 0.2–0.8).
        # Keep headroom so medium band still routes to LLM verification.
        confidence = (
            0.50 * top_score
            + 0.35 * min(margin * 2.0, 1.0)
            + keyword_boost
            + 0.10
        )
        confidence = float(min(max(confidence, 0.0), 0.98))

        return top_tool, confidence, combined_scores

    def batch_predict(self, messages: List[str]) -> List[Tuple[str, float, Dict[str, float]]]:
        return [self.predict(msg) for msg in messages]
