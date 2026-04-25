"""
at_crag.py — Adaptive Token-Budget Corrective RAG pipeline.

Identical to :mod:`baseline_crag` except that the retrieval budget
(K documents, N correction iterations) is decided dynamically per query
by the complexity controller.  This is the **only structural difference**
and isolates the independent variable cleanly.
"""

import time
from typing import Dict, List

from at_crag.retriever.embedder import Embedder
from at_crag.retriever.faiss_index import FAISSIndex
from at_crag.evaluator.relevance_scorer import RelevanceScorer
from at_crag.generator.reader import Generator
from at_crag.complexity.classifier import ComplexityClassifier
from at_crag.controller.budget_controller import assign_budget


class AdaptiveCRAG:
    """End-to-end adaptive-budget CRAG pipeline.

    Attributes:
        embedder:   Dense query encoder.
        index:      Populated FAISS index.
        scorer:     Cross-encoder relevance scorer.
        generator:  Flan-T5-base answer generator.
        classifier: Query complexity classifier.
    """

    def __init__(
        self,
        embedder: Embedder,
        index: FAISSIndex,
        scorer: RelevanceScorer,
        generator: Generator,
        classifier: ComplexityClassifier,
    ):
        """Inject pre-loaded components.

        Args:
            embedder:   Initialised :class:`Embedder`.
            index:      Populated :class:`FAISSIndex`.
            scorer:     Initialised :class:`RelevanceScorer`.
            generator:  Initialised :class:`Generator`.
            classifier: Trained :class:`ComplexityClassifier`.
        """
        self.embedder = embedder
        self.index = index
        self.scorer = scorer
        self.generator = generator
        self.classifier = classifier

    def run(self, query: str) -> Dict:
        """Execute the AT-CRAG pipeline on a single query.

        Algorithm:
            1. Predict query complexity (0 / 1 / 2).
            2. Assign budget: (K, N) = assign_budget(complexity_level).
            3. Run the CRAG loop with dynamic K and N.
            4. Generate answer.

        The CRAG loop is **identical** to :meth:`BaselineCRAG.run` —
        only the source of K and N differs.

        Args:
            query: User question.

        Returns:
            Dict with keys: ``answer``, ``action``, ``complexity_level``,
            ``complexity_label``, ``K``, ``N``, ``iterations_used``,
            ``total_passages_seen``, ``latency_seconds``.
        """
        t0 = time.perf_counter()

        # --- Step 1-2: Complexity → Budget --------------------------------
        complexity_level = self.classifier.predict_complexity(query)
        complexity_label = ComplexityClassifier.LABELS[complexity_level]
        budget = assign_budget(complexity_level)
        K = budget["K"]
        N = budget["N"]

        # --- Step 3: CRAG Loop (identical to baseline) --------------------
        current_query = query
        total_passages_seen = 0
        final_action = "INCORRECT"
        context_passages: List[str] = []
        iteration = 0

        for iteration in range(1, N + 1):
            # Encode
            q_emb = self.embedder.encode_query(current_query)

            # Retrieve
            results = self.index.retrieve(q_emb, K)
            passages = [p["text"] for p, _ in results]
            total_passages_seen += len(passages)

            # Score
            scores, action = self.scorer.evaluate(current_query, passages)
            final_action = action

            # Decide
            if action == "CORRECT":
                ranked = sorted(zip(scores, passages), reverse=True)
                context_passages = [p for _, p in ranked]
                break

            elif action == "AMBIGUOUS":
                ranked = sorted(zip(scores, passages), reverse=True)
                context_passages = [p for _, p in ranked[:3]]
                break

            else:  # INCORRECT
                if iteration < N:
                    current_query = f"detailed explanation of {query}"
                else:
                    ranked = sorted(zip(scores, passages), reverse=True)
                    context_passages = [p for _, p in ranked[:3]]

        # --- Step 4: Generate ---------------------------------------------
        if not context_passages:
            context_passages = ["No relevant information found."]

        answer = self.generator.generate_answer(query, context_passages)

        latency = time.perf_counter() - t0

        return {
            "answer": answer,
            "action": final_action,
            "complexity_level": complexity_level,
            "complexity_label": complexity_label,
            "K": K,
            "N": N,
            "iterations_used": iteration,
            "total_passages_seen": total_passages_seen,
            "latency_seconds": round(latency, 4),
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def create_adaptive_pipeline(index_dir: str, classifier_dir: str | None = None) -> AdaptiveCRAG:
    """Build a ready-to-use AT-CRAG pipeline.

    Args:
        index_dir:      Path to the saved FAISS index directory.
        classifier_dir: Path to the saved complexity classifier.
                        If ``None``, uses the default location.

    Returns:
        Fully initialised :class:`AdaptiveCRAG` instance.
    """
    embedder = Embedder()
    index = FAISSIndex()
    index.load(index_dir)
    scorer = RelevanceScorer()
    generator = Generator()
    classifier = ComplexityClassifier(model_dir=classifier_dir)
    classifier.load()
    return AdaptiveCRAG(embedder, index, scorer, generator, classifier)
