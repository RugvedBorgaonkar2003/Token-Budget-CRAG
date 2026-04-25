"""
embedder.py — Dense passage / query encoder using Sentence-Transformers.

Wraps ``all-MiniLM-L6-v2`` (384-dim, ~80 MB, CPU-friendly) behind a clean
API used by the FAISS indexing and retrieval modules.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


class Embedder:
    """Encodes text (passages or queries) into dense 384-d vectors.

    Attributes:
        model_name: HuggingFace model identifier.
        model:      Loaded SentenceTransformer instance.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialise the embedding model.

        Args:
            model_name: Any model supported by ``sentence-transformers``.
                        Defaults to ``all-MiniLM-L6-v2`` (384-dim, CPU-fast).
        """
        self.model_name = model_name
        print(f"[embedder] Loading model '{model_name}' …")
        self.model = SentenceTransformer(model_name)
        print("[embedder] Model loaded.")

    def encode_passages(
        self,
        passages: List[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode a list of passage strings into an embedding matrix.

        Args:
            passages:       List of raw text passages.
            batch_size:     Encoding batch size (tune for your RAM).
            show_progress:  Show tqdm progress bar.

        Returns:
            ``np.ndarray`` of shape ``(len(passages), embedding_dim)``.
        """
        embeddings = self.model.encode(
            passages,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string.

        Args:
            query: The user question.

        Returns:
            ``np.ndarray`` of shape ``(1, embedding_dim)``.
        """
        emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb.astype(np.float32)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    enc = Embedder()
    q = enc.encode_query("Who invented the telephone?")
    print(f"Query embedding shape: {q.shape}")

    p = enc.encode_passages(["Alexander Graham Bell invented the telephone."])
    print(f"Passage embedding shape: {p.shape}")
