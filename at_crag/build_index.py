"""
build_index.py — One-time script to build the FAISS index from Wikipedia.

Run this ONCE before evaluation:

    python -m at_crag.build_index                  # default: 50K articles
    python -m at_crag.build_index --limit 1000     # tiny dev build
"""

import os
import sys
import argparse
import random

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

from at_crag.retriever.corpus import load_wikipedia_subset
from at_crag.retriever.embedder import Embedder
from at_crag.retriever.faiss_index import FAISSIndex


def build(limit: int = 50_000, batch_size: int = 256) -> None:
    """Download corpus, encode passages, build and save FAISS index.

    Args:
        limit:      Max Wikipedia articles to download.
        batch_size: Embedding batch size.
    """
    index_dir = os.path.join(_PROJECT_ROOT, "data", "faiss_index")

    if os.path.exists(os.path.join(index_dir, "faiss.index")):
        print(f"[build] Index already exists at {index_dir} — skipping.")
        print("[build] Delete the folder to rebuild.")
        return

    # 1. Corpus
    passages = load_wikipedia_subset(limit=limit)
    texts = [p["text"] for p in passages]

    # 2. Embed
    embedder = Embedder()
    print(f"[build] Encoding {len(texts):,} passages (batch_size={batch_size}) …")
    embeddings = embedder.encode_passages(texts, batch_size=batch_size)
    print(f"[build] Embedding matrix: {embeddings.shape}")

    # 3. Index
    index = FAISSIndex(dim=embeddings.shape[1])
    index.add_embeddings(embeddings, passages)
    index.save(index_dir)
    print("[build] Done ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index")
    parser.add_argument("--limit", type=int, default=50_000,
                        help="Number of Wikipedia articles (default: 50000)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Embedding batch size (default: 256)")
    args = parser.parse_args()
    build(limit=args.limit, batch_size=args.batch_size)
