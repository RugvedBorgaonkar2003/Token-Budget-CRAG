"""
baseline_crag.py — Fixed-budget Corrective RAG pipeline.

Implements the original CRAG loop (Yan et al., 2024) with hard-coded
retrieval parameters K=5 and N=2.  Serves as the **control** against
which AT-CRAG is compared.
"""

import time
from typing import Dict, List, Tuple

from at_crag.retriever.embedder import Embedder
from at_crag.retriever.faiss_index import FAISSIndex
from at_crag.evaluator.relevance_scorer import RelevanceScorer
from at_crag.generator.reader import Generator


class BaselineCRAG:
    """End-to-end fixed-budget CRAG pipeline.

    The pipeline caches all heavy components (embedder, FAISS index,
    cross-encoder, generator) so they are loaded once and reused across
    calls.

    Attributes:
        embedder: Dense query encoder.
        index:    Populated FAISS index.
        scorer:   Cross-encoder relevance scorer.
        generator: Flan-T5-base answer generator.
    """

    def __init__(
        self,
        embedder: Embedder,
        index: FAISSIndex,
        scorer: RelevanceScorer,
        generator: Generator,
    ):
        """Inject pre-loaded components.

        Args:
            embedder:  Initialised :class:`Embedder`.
            index:     Populated :class:`FAISSIndex`.
            scorer:    Initialised :class:`RelevanceScorer`.
            generator: Initialised :class:`Generator`.
        """
        self.embedder = embedder
        self.index = index
        self.scorer = scorer
        self.generator = generator

    def run(
        self,
        query: str,
        K: int = 5,
        N: int = 2,
    ) -> Dict:
        """Execute the baseline CRAG pipeline on a single query.

        Algorithm:
            1. Encode the query.
            2. Retrieve top-K passages from FAISS.
            3. Score passages with the cross-encoder.
            4. Based on the CRAG action:
               - CORRECT  → pass top passages to the generator.
               - INCORRECT → rewrite query and retry (up to N iterations).
               - AMBIGUOUS → take only the top-3 scored passages.
            5. Generate the answer.

        Args:
            query: User question.
            K:     Number of passages to retrieve (fixed).
            N:     Maximum corrective iterations (fixed).

        Returns:
            Dict with keys: ``answer``, ``action``, ``iterations_used``,
            ``total_passages_seen``, ``latency_seconds``.
        """
        t0 = time.perf_counter()

        current_query = query
        total_passages_seen = 0
        final_action = "INCORRECT"
        context_passages: List[str] = []

        for iteration in range(1, N + 1):
            # 1. Encode
            q_emb = self.embedder.encode_query(current_query)

            # 2. Retrieve
            results = self.index.retrieve(q_emb, K)
            passages = [p["text"] for p, _ in results]
            total_passages_seen += len(passages)

            # 3. Score
            scores, action = self.scorer.evaluate(current_query, passages)
            final_action = action

            # 4. Decide
            if action == "CORRECT":
                # Use all retrieved passages (sorted by score desc)
                ranked = sorted(zip(scores, passages), reverse=True)
                context_passages = [p for _, p in ranked]
                break

            elif action == "AMBIGUOUS":
                # Take only top-3 scored passages
                ranked = sorted(zip(scores, passages), reverse=True)
                context_passages = [p for _, p in ranked[:3]]
                break

            else:  # INCORRECT
                if iteration < N:
                    # Rewrite query for next iteration
                    current_query = f"detailed explanation of {query}"
                else:
                    # Last iteration — use whatever we have
                    ranked = sorted(zip(scores, passages), reverse=True)
                    context_passages = [p for _, p in ranked[:3]]

        # 5. Generate
        if not context_passages:
            context_passages = ["No relevant information found."]

        answer = self.generator.generate_answer(query, context_passages)

        latency = time.perf_counter() - t0

        return {
            "answer": answer,
            "action": final_action,
            "iterations_used": iteration,
            "total_passages_seen": total_passages_seen,
            "latency_seconds": round(latency, 4),
            "K": K,
            "N": N,
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def create_baseline_pipeline(index_dir: str) -> BaselineCRAG:
    """Build a ready-to-use baseline CRAG pipeline.

    Args:
        index_dir: Path to the saved FAISS index directory.

    Returns:
        Fully initialised :class:`BaselineCRAG` instance.
    """
    embedder = Embedder()
    index = FAISSIndex()
    index.load(index_dir)
    scorer = RelevanceScorer()
    generator = Generator()
    return BaselineCRAG(embedder, index, scorer, generator)
