"""
run_eval.py — Evaluation harness for Baseline CRAG vs. AT-CRAG.

Loads 500 TriviaQA validation questions, runs both pipelines, logs
per-question results to JSON, prints a summary table, and optionally
runs an ablation study broken down by complexity tier.

Usage::

    python -m at_crag.eval.run_eval              # full 500-question eval
    python -m at_crag.eval.run_eval --num 50      # quick dev run
    python -m at_crag.eval.run_eval --ablation    # include ablation table
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path for relative imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from at_crag.retriever.embedder import Embedder
from at_crag.retriever.faiss_index import FAISSIndex
from at_crag.evaluator.relevance_scorer import RelevanceScorer
from at_crag.generator.reader import Generator
from at_crag.complexity.classifier import ComplexityClassifier
from at_crag.controller.budget_controller import assign_budget
from at_crag.pipeline.baseline_crag import BaselineCRAG
from at_crag.pipeline.at_crag import AdaptiveCRAG
from at_crag.eval.metrics import exact_match, token_f1

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "experiments", "results")
_INDEX_DIR = os.path.join(_PROJECT_ROOT, "data", "faiss_index")


def _load_questions(num: int = 500) -> List[Dict]:
    """Load TriviaQA validation questions.

    Args:
        num: Number of questions to load.

    Returns:
        List of dicts with keys ``question`` and ``gold_answers``.
    """
    print(f"[eval] Loading {num} TriviaQA validation questions …")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation", streaming=True)

    questions: List[Dict] = []
    for i, row in enumerate(ds):
        if i >= num:
            break
        gold = row.get("answer", {})
        aliases = gold.get("aliases", [])
        value = gold.get("value", "")
        all_answers = list(set([value] + aliases)) if value else aliases
        if not all_answers:
            continue
        questions.append({
            "question": row["question"],
            "gold_answers": all_answers,
        })

    print(f"[eval] Loaded {len(questions)} questions.")
    return questions


def _init_shared_components() -> dict:
    """Load all heavy components once and return as a dict."""
    embedder = Embedder()
    index = FAISSIndex()
    index.load(_INDEX_DIR)
    scorer = RelevanceScorer()
    generator = Generator()
    classifier = ComplexityClassifier()
    classifier.load()
    return {
        "embedder": embedder,
        "index": index,
        "scorer": scorer,
        "generator": generator,
        "classifier": classifier,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(num: int = 500, run_ablation_flag: bool = False) -> None:
    """Run both pipelines on *num* TriviaQA questions and log results.

    Args:
        num: Number of evaluation questions.
        run_ablation_flag: If ``True``, also run the ablation study.
    """
    questions = _load_questions(num)
    components = _init_shared_components()

    baseline = BaselineCRAG(
        embedder=components["embedder"],
        index=components["index"],
        scorer=components["scorer"],
        generator=components["generator"],
    )
    adaptive = AdaptiveCRAG(
        embedder=components["embedder"],
        index=components["index"],
        scorer=components["scorer"],
        generator=components["generator"],
        classifier=components["classifier"],
    )

    results: List[Dict] = []

    for item in tqdm(questions, desc="Evaluating"):
        q = item["question"]
        gold = item["gold_answers"]

        # --- Baseline CRAG ---
        bl = baseline.run(q, K=5, N=2)
        bl_em = exact_match(bl["answer"], gold)
        bl_f1 = token_f1(bl["answer"], gold)

        # --- AT-CRAG ---
        at = adaptive.run(q)
        at_em = exact_match(at["answer"], gold)
        at_f1 = token_f1(at["answer"], gold)

        results.append({
            "question": q,
            "gold_answers": gold,
            # baseline
            "baseline_answer": bl["answer"],
            "baseline_em": int(bl_em),
            "baseline_f1": round(bl_f1, 4),
            "baseline_latency": bl["latency_seconds"],
            "baseline_action": bl["action"],
            "baseline_iterations": bl["iterations_used"],
            # adaptive
            "at_crag_answer": at["answer"],
            "at_crag_em": int(at_em),
            "at_crag_f1": round(at_f1, 4),
            "at_crag_latency": at["latency_seconds"],
            "at_crag_action": at["action"],
            "at_crag_complexity": at["complexity_label"],
            "at_crag_K": at["K"],
            "at_crag_N": at["N"],
            "at_crag_iterations": at["iterations_used"],
        })

    # ---- Save ------------------------------------------------------------
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(_RESULTS_DIR, f"run_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[eval] Results saved to {out_path}")

    # ---- Summary table ---------------------------------------------------
    df = pd.DataFrame(results)
    summary = {
        "Method": ["Baseline CRAG", "AT-CRAG"],
        "EM": [
            round(df["baseline_em"].mean(), 4),
            round(df["at_crag_em"].mean(), 4),
        ],
        "F1": [
            round(df["baseline_f1"].mean(), 4),
            round(df["at_crag_f1"].mean(), 4),
        ],
        "Avg Latency (s)": [
            round(df["baseline_latency"].mean(), 4),
            round(df["at_crag_latency"].mean(), 4),
        ],
    }
    summary_df = pd.DataFrame(summary)
    print("\n" + "=" * 55)
    print("  EVALUATION SUMMARY")
    print("=" * 55)
    print(summary_df.to_string(index=False))
    print("=" * 55)

    # ---- Ablation --------------------------------------------------------
    if run_ablation_flag:
        run_ablation(df)


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

def run_ablation(df: pd.DataFrame) -> None:
    """Print per-complexity-tier metrics for AT-CRAG.

    Shows that SIMPLE queries are faster (less retrieval) while COMPLEX
    queries are more accurate (more retrieval) — the core hypothesis.

    Args:
        df: DataFrame with per-question evaluation results.
    """
    print("\n" + "=" * 65)
    print("  ABLATION: AT-CRAG by Complexity Tier")
    print("=" * 65)

    for tier in ["SIMPLE", "MEDIUM", "COMPLEX"]:
        subset = df[df["at_crag_complexity"] == tier]
        if subset.empty:
            continue
        n = len(subset)
        em = round(subset["at_crag_em"].mean(), 4)
        f1 = round(subset["at_crag_f1"].mean(), 4)
        lat = round(subset["at_crag_latency"].mean(), 4)
        bl_em = round(subset["baseline_em"].mean(), 4)
        bl_f1 = round(subset["baseline_f1"].mean(), 4)
        bl_lat = round(subset["baseline_latency"].mean(), 4)

        print(f"\n  {tier} (n={n})")
        print(f"    {'':15s} {'EM':>8s} {'F1':>8s} {'Latency':>10s}")
        print(f"    {'Baseline':15s} {bl_em:8.4f} {bl_f1:8.4f} {bl_lat:10.4f}")
        print(f"    {'AT-CRAG':15s} {em:8.4f} {f1:8.4f} {lat:10.4f}")

    print("\n" + "=" * 65)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AT-CRAG Evaluation")
    parser.add_argument("--num", type=int, default=500,
                        help="Number of TriviaQA questions (default: 500)")
    parser.add_argument("--ablation", action="store_true",
                        help="Include ablation breakdown by complexity tier")
    args = parser.parse_args()

    run_evaluation(num=args.num, run_ablation_flag=args.ablation)
