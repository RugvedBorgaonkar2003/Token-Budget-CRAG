"""
faiss_index.py — Build, persist and query a FAISS flat-L2 index.

The index is built once from passage embeddings and saved to disk so
subsequent runs skip the expensive encoding step.
"""

import os
from typing import List, Dict, Tuple

import faiss
import numpy as np


class FAISSIndex:
    """Thin wrapper around a ``faiss.IndexFlatL2`` index.

    Attributes:
        dim:       Embedding dimensionality (default 384).
        index:     The underlying FAISS index.
        passages:  The passage metadata list (aligned by row id).
    """

    def __init__(self, dim: int = 384):
        """Create an empty index.

        Args:
            dim: Vector dimensionality — must match the embedder output.
        """
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.passages: List[Dict] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def add_embeddings(self, embeddings: np.ndarray, passages: List[Dict]) -> None:
        """Add pre-computed embeddings and their passage metadata.

        Args:
            embeddings: ``(N, dim)`` float32 matrix.
            passages:   Corresponding list of ``{id, text, title}`` dicts.
        """
        assert embeddings.shape[0] == len(passages), (
            f"Embedding rows ({embeddings.shape[0]}) != passages ({len(passages)})"
        )
        self.index.add(embeddings.astype(np.float32))
        self.passages.extend(passages)
        print(f"[faiss] Index now contains {self.index.ntotal:,} vectors.")

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------
    def retrieve(
        self, query_embedding: np.ndarray, K: int = 5
    ) -> List[Tuple[Dict, float]]:
        """Return the top-K nearest passages for a query embedding.

        Args:
            query_embedding: ``(1, dim)`` float32 vector.
            K:               Number of neighbours to retrieve.

        Returns:
            List of ``(passage_dict, L2_distance)`` tuples sorted by
            ascending distance (best match first).
        """
        distances, indices = self.index.search(query_embedding.astype(np.float32), K)
        results: List[Tuple[Dict, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue  # fewer than K vectors in index
            results.append((self.passages[idx], float(dist)))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, directory: str) -> None:
        """Save the FAISS index and passage list to *directory*.

        Args:
            directory: Target folder (created if absent).
        """
        os.makedirs(directory, exist_ok=True)
        index_path = os.path.join(directory, "faiss.index")
        meta_path = os.path.join(directory, "passages.npy")

        faiss.write_index(self.index, index_path)
        np.save(meta_path, self.passages, allow_pickle=True)
        print(f"[faiss] Saved index ({self.index.ntotal:,} vectors) to {directory}")

    def load(self, directory: str) -> None:
        """Load a previously saved index from *directory*.

        Args:
            directory: Folder containing ``faiss.index`` and ``passages.npy``.
        """
        index_path = os.path.join(directory, "faiss.index")
        meta_path = os.path.join(directory, "passages.npy")

        self.index = faiss.read_index(index_path)
        self.passages = np.load(meta_path, allow_pickle=True).tolist()
        self.dim = self.index.d
        print(f"[faiss] Loaded index with {self.index.ntotal:,} vectors from {directory}")


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    idx = FAISSIndex(dim=4)
    vecs = np.random.rand(10, 4).astype(np.float32)
    meta = [{"id": str(i), "text": f"passage {i}", "title": "test"} for i in range(10)]
    idx.add_embeddings(vecs, meta)

    query = np.random.rand(1, 4).astype(np.float32)
    results = idx.retrieve(query, K=3)
    for passage, dist in results:
        print(f"  {passage['id']}  dist={dist:.4f}")
