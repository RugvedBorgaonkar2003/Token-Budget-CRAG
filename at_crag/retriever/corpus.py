"""
corpus.py — Load and chunk a Wikipedia subset into retrieval passages.

Uses the HuggingFace `datasets` library to stream a slice of Wikipedia.
Each article is split into ~100-word passages with a 20-word overlap so
that no single passage is too long for the embedding model while still
preserving cross-boundary context.
"""

import os
import json
import random
from typing import List, Dict

from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(42)

# Default artefact paths (relative to project root)
_DEFAULT_CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def _chunk_text(text: str, title: str, chunk_size: int = 100, overlap: int = 20) -> List[Dict]:
    """Split *text* into word-level chunks.

    Args:
        text:       Full article body.
        title:      Article title (attached to every passage for provenance).
        chunk_size: Target number of words per passage.
        overlap:    Number of overlapping words between consecutive passages.

    Returns:
        List of dicts ``{id, text, title}`` — one per passage.
    """
    words = text.split()
    passages: List[Dict] = []
    start = 0
    passage_id = 0
    while start < len(words):
        end = start + chunk_size
        passage_text = " ".join(words[start:end])
        passages.append({
            "id": f"{title}_{passage_id}",
            "text": passage_text,
            "title": title,
        })
        passage_id += 1
        start += chunk_size - overlap  # slide window
    return passages


def load_wikipedia_subset(
    limit: int = 50_000,
    chunk_size: int = 100,
    overlap: int = 20,
    cache_dir: str | None = None,
) -> List[Dict]:
    """Load the first *limit* Wikipedia articles and chunk them.

    If a cached JSON file exists on disk the corpus is loaded from there
    instead of re-downloading and re-chunking.

    Args:
        limit:      Maximum number of articles to download.
        chunk_size: Words per passage.
        overlap:    Word overlap between consecutive passages.
        cache_dir:  Directory for the cache file.  Falls back to
                    ``data/processed/`` inside the project tree.

    Returns:
        List of passage dicts ``{id, text, title}``.
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"wiki_passages_{limit}.json")

    # ---- fast path: load from cache ----
    if os.path.exists(cache_path):
        print(f"[corpus] Loading cached passages from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---- slow path: download + chunk ----
    print(f"[corpus] Downloading first {limit:,} Wikipedia articles …")
    ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

    all_passages: List[Dict] = []
    for i, article in enumerate(tqdm(ds, total=limit, desc="Chunking articles")):
        if i >= limit:
            break
        title = article.get("title", f"article_{i}")
        body = article.get("text", "")
        if not body.strip():
            continue
        all_passages.extend(_chunk_text(body, title, chunk_size, overlap))

    print(f"[corpus] Created {len(all_passages):,} passages from {limit:,} articles.")

    # persist
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(all_passages, f)
    print(f"[corpus] Saved to {cache_path}")

    return all_passages


# ---------------------------------------------------------------------------
# Quick smoke-test when executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    passages = load_wikipedia_subset(limit=100)  # tiny test
    print(f"Total passages: {len(passages)}")
    if passages:
        print("Sample:", passages[0])
