"""
metrics.py — QA evaluation metrics and latency measurement.

Provides Exact Match (EM), token-level F1, and a latency helper used
by both the baseline and adaptive evaluation harnesses.
"""

import re
import time
import string
from typing import Callable, Tuple, Any, List


def _normalize(text: str) -> str:
    """Lower-case, strip punctuation and articles for fair comparison.

    Args:
        text: Raw answer string.

    Returns:
        Normalised string.
    """
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def exact_match(prediction: str, gold_answers: str | List[str]) -> bool:
    """Check whether the prediction exactly matches any gold answer.

    Args:
        prediction:   Model-generated answer.
        gold_answers: Single string or list of acceptable gold answers.

    Returns:
        ``True`` if the normalised prediction equals any normalised gold.
    """
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]

    pred_norm = _normalize(prediction)
    return any(_normalize(g) == pred_norm for g in gold_answers)


def token_f1(prediction: str, gold_answers: str | List[str]) -> float:
    """Compute token-level F1 between prediction and the best-matching gold.

    Standard QA F1: overlap of word tokens between predicted and gold
    answer, expressed as the harmonic mean of precision and recall.

    Args:
        prediction:   Model-generated answer.
        gold_answers: Single string or list of acceptable gold answers.

    Returns:
        F1 score in [0.0, 1.0].  Returns the maximum across all golds.
    """
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]

    pred_tokens = _normalize(prediction).split()
    best_f1 = 0.0

    for gold in gold_answers:
        gold_tokens = _normalize(gold).split()

        common = set(pred_tokens) & set(gold_tokens)
        num_common = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)

        if num_common == 0:
            continue

        precision = num_common / len(pred_tokens) if pred_tokens else 0.0
        recall = num_common / len(gold_tokens) if gold_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        best_f1 = max(best_f1, f1)

    return best_f1


def measure_latency(fn: Callable, *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    """Call *fn* and measure wall-clock time.

    Args:
        fn:     Callable to time.
        *args:  Positional arguments forwarded to *fn*.
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        ``(result, elapsed_seconds)`` tuple.
    """
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("EM:", exact_match("Alexander Graham Bell", "alexander graham bell"))
    print("EM:", exact_match("Bell", "Alexander Graham Bell"))
    print("F1:", token_f1("Alexander Graham Bell", "Alexander Graham Bell"))
    print("F1:", token_f1("Graham Bell", "Alexander Graham Bell"))
    print("F1:", token_f1("completely wrong answer", "Alexander Graham Bell"))
