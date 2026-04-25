"""
relevance_scorer.py — CRAG-style document relevance evaluator.

Uses a cross-encoder (``cross-encoder/ms-marco-MiniLM-L-6-v2``) to score
each retrieved passage against the query, then maps the maximum score to
one of three CRAG actions: CORRECT · AMBIGUOUS · INCORRECT.
"""

from typing import List, Tuple
from sentence_transformers import CrossEncoder


class RelevanceScorer:
    """Score passages against a query and decide the CRAG action.

    Attributes:
        model_name: Cross-encoder identifier.
        model:      Loaded ``CrossEncoder`` instance.
    """

    # CRAG action thresholds (from the paper)
    CORRECT_THRESHOLD: float = 0.7
    INCORRECT_THRESHOLD: float = 0.3

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Load the cross-encoder model.

        Args:
            model_name: HuggingFace cross-encoder model identifier.
        """
        self.model_name = model_name
        print(f"[relevance] Loading cross-encoder '{model_name}' …")
        self.model = CrossEncoder(model_name, max_length=512)
        print("[relevance] Cross-encoder loaded.")

    def score_documents(self, query: str, passages: List[str]) -> List[float]:
        """Score each passage for relevance to *query*.

        Args:
            query:    The user question.
            passages: List of passage texts.

        Returns:
            List of float scores (one per passage), each roughly in [0, 1]
            thanks to the sigmoid applied by the cross-encoder.
        """
        if not passages:
            return []
        pairs = [[query, p] for p in passages]
        raw_scores = self.model.predict(pairs)
        # ms-marco cross-encoder outputs logits; apply sigmoid for [0,1]
        import numpy as np
        scores = (1.0 / (1.0 + np.exp(-np.array(raw_scores)))).tolist()
        return scores

    def classify_action(self, scores: List[float]) -> str:
        """Map the score distribution to a CRAG action.

        Decision logic (following Yan et al., 2024):
            * max(scores) > 0.7  → ``"CORRECT"``
            * max(scores) < 0.3  → ``"INCORRECT"``
            * otherwise          → ``"AMBIGUOUS"``

        Args:
            scores: Per-passage relevance scores from :meth:`score_documents`.

        Returns:
            One of ``"CORRECT"``, ``"INCORRECT"``, ``"AMBIGUOUS"``.
        """
        if not scores:
            return "INCORRECT"
        max_score = max(scores)
        if max_score > self.CORRECT_THRESHOLD:
            return "CORRECT"
        elif max_score < self.INCORRECT_THRESHOLD:
            return "INCORRECT"
        else:
            return "AMBIGUOUS"

    def evaluate(
        self, query: str, passages: List[str]
    ) -> Tuple[List[float], str]:
        """Convenience wrapper: score *and* classify in one call.

        Args:
            query:    User question.
            passages: Retrieved passage texts.

        Returns:
            ``(scores, action)`` tuple.
        """
        scores = self.score_documents(query, passages)
        action = self.classify_action(scores)
        return scores, action


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    scorer = RelevanceScorer()
    q = "Who invented the telephone?"
    docs = [
        "Alexander Graham Bell is widely credited with inventing the telephone.",
        "The Great Wall of China is visible from space according to popular myth.",
    ]
    scores, action = scorer.evaluate(q, docs)
    for doc, s in zip(docs, scores):
        print(f"  score={s:.3f}  {doc[:60]}…")
    print(f"  → action: {action}")
