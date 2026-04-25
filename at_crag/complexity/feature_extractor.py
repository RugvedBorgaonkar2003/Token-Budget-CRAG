"""
feature_extractor.py — Hand-crafted feature extraction for query complexity.

Produces a fixed-length numeric vector from a raw query string.  These
features are consumed by the complexity classifier to decide whether a
query is SIMPLE, MEDIUM, or COMPLEX *before* retrieval happens.
"""

import re
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Tiny POS-like verb list (covers ~90 % of common English question verbs).
# Used instead of a full POS tagger to stay CPU-light.
# ---------------------------------------------------------------------------
_COMMON_VERBS = frozenset([
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "say", "said", "get", "got", "make", "made",
    "go", "went", "gone", "take", "took", "taken",
    "come", "came", "know", "knew", "known",
    "see", "saw", "seen", "give", "gave", "given",
    "find", "found", "think", "thought", "tell", "told",
    "become", "became", "show", "showed", "shown",
    "leave", "left", "call", "called", "describe",
    "explain", "compare", "define", "list", "name",
])

# Question-type keywords (one-hot encoded)
_Q_TYPES = ["what", "who", "when", "where", "how", "why", "which"]

# Negation words
_NEGATIONS = frozenset(["not", "never", "no", "without", "except", "neither", "nor"])

# Conjunctions that hint at multi-hop reasoning
_CONJUNCTIONS = frozenset(["and", "or", "but", "also", "both", "either"])

# Comparative keywords
_COMPARATIVES = frozenset([
    "compare", "comparison", "difference", "differences",
    "versus", "vs", "than", "better", "worse",
    "more", "less", "most", "least",
])


def extract_features(query: str) -> np.ndarray:
    """Extract a feature vector from a raw query string.

    Feature layout (15 dimensions total):
        0       — query_length          (token count)
        1       — num_entities          (capitalised words, proxy for NER)
        2-8     — question_type one-hot (what/who/when/where/how/why/which)
        9       — has_negation          (binary)
        10      — conjunction_count     (count of and/or/but …)
        11      — verb_count            (count of common verbs)
        12      — is_comparative        (binary)
        13      — avg_word_length       (mean characters per token)
        14      — has_superlative       (binary: most/least/best/worst)

    Args:
        query: The raw user question.

    Returns:
        ``np.ndarray`` of shape ``(15,)`` with dtype ``float32``.
    """
    tokens: List[str] = query.split()
    tokens_lower: List[str] = [t.lower().strip("?.,!;:'\"") for t in tokens]

    # 0 — query length
    query_length = len(tokens)

    # 1 — entity proxy (capitalised words, skip first word & stopwords)
    num_entities = sum(
        1 for t in tokens[1:] if t and t[0].isupper()
    )

    # 2-8 — question type one-hot
    first_word = tokens_lower[0] if tokens_lower else ""
    q_type_vec = [1.0 if first_word == qt else 0.0 for qt in _Q_TYPES]

    # 9 — negation
    has_negation = float(any(t in _NEGATIONS for t in tokens_lower))

    # 10 — conjunction count
    conjunction_count = sum(1 for t in tokens_lower if t in _CONJUNCTIONS)

    # 11 — verb count
    verb_count = sum(1 for t in tokens_lower if t in _COMMON_VERBS)

    # 12 — comparative
    is_comparative = float(any(t in _COMPARATIVES for t in tokens_lower))

    # 13 — average word length
    avg_word_length = (
        np.mean([len(t) for t in tokens]) if tokens else 0.0
    )

    # 14 — superlative
    _SUPERLATIVES = {"most", "least", "best", "worst", "largest", "smallest",
                     "highest", "lowest", "greatest", "first", "last"}
    has_superlative = float(any(t in _SUPERLATIVES for t in tokens_lower))

    feature_vec = np.array(
        [query_length, num_entities]
        + q_type_vec
        + [has_negation, conjunction_count, verb_count, is_comparative,
           avg_word_length, has_superlative],
        dtype=np.float32,
    )
    return feature_vec


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    queries = [
        "Who invented the telephone?",
        "What is the difference between RNA and DNA in terms of structure and function?",
        "When was the Battle of Hastings?",
    ]
    for q in queries:
        vec = extract_features(q)
        print(f"[{vec.shape}] {q}")
        print(f"  features = {vec.tolist()}\n")
