"""
classifier.py — Train and infer query complexity (SIMPLE / MEDIUM / COMPLEX).

This is the **core novel component** of AT-CRAG.  A lightweight
scikit-learn classifier is trained on heuristically labelled TriviaQA
questions and used at inference time to decide the retrieval budget
*before* the CRAG loop begins.

Complexity levels:
    0 — SIMPLE   (single-hop, short query)
    1 — MEDIUM   (moderate depth needed)
    2 — COMPLEX  (multi-hop, comparative, or negated)
"""

import os
import json
import random
from typing import List, Tuple

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import load_dataset
from tqdm import tqdm

from at_crag.complexity.feature_extractor import extract_features

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

# Persistence defaults
_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")


# ---------------------------------------------------------------------------
# Heuristic labelling
# ---------------------------------------------------------------------------

def _heuristic_label(query: str) -> int:
    """Assign a complexity label based on surface-level query features.

    Labelling rules (deliberately simple — the classifier will learn to
    generalise beyond these heuristics):
        * SIMPLE (0) — ≤ 8 tokens AND no conjunctions AND ≤ 1 entity
        * COMPLEX (2) — > 1 conjunction OR comparative keyword OR > 2 entities
                         OR negation present
        * MEDIUM (1) — everything else

    Args:
        query: Raw question string.

    Returns:
        Integer complexity label in {0, 1, 2}.
    """
    tokens = query.split()
    tokens_lower = [t.lower().strip("?.,!;:'\"") for t in tokens]
    length = len(tokens)

    num_entities = sum(1 for t in tokens[1:] if t and t[0].isupper())
    conjunctions = sum(1 for t in tokens_lower if t in {"and", "or", "but", "both", "either"})
    has_negation = any(t in {"not", "never", "without", "except"} for t in tokens_lower)
    has_comparative = any(
        t in {"compare", "difference", "versus", "vs", "than"}
        for t in tokens_lower
    )

    # SIMPLE
    if length <= 8 and conjunctions == 0 and num_entities <= 1:
        return 0
    # COMPLEX
    if conjunctions > 1 or has_comparative or num_entities > 2 or has_negation:
        return 2
    # MEDIUM
    return 1


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_labelled_dataset(
    num_samples: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Create features + labels from TriviaQA validation questions.

    Args:
        num_samples: Number of questions to label.

    Returns:
        ``(X, y, raw_queries)`` where X is ``(N, 15)`` and y is ``(N,)``.
    """
    print(f"[classifier] Building labelled dataset ({num_samples:,} samples) …")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation", streaming=True)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    queries: List[str] = []

    for i, row in enumerate(tqdm(ds, total=num_samples, desc="Labelling")):
        if i >= num_samples:
            break
        q = row["question"]
        feat = extract_features(q)
        label = _heuristic_label(q)
        X_list.append(feat)
        y_list.append(label)
        queries.append(q)

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"[classifier] Label distribution: "
          f"SIMPLE={np.sum(y==0)}, MEDIUM={np.sum(y==1)}, COMPLEX={np.sum(y==2)}")
    return X, y, queries


# ---------------------------------------------------------------------------
# Classifier wrapper
# ---------------------------------------------------------------------------

class ComplexityClassifier:
    """Wraps a scikit-learn model for three-class query complexity.

    Attributes:
        model:     The trained estimator (LogisticRegression or RandomForest).
        model_dir: Path where the model artefact is persisted.
    """

    LABELS = {0: "SIMPLE", 1: "MEDIUM", 2: "COMPLEX"}

    def __init__(self, model_dir: str | None = None, use_rf: bool = False):
        """Initialise (does NOT load/train automatically).

        Args:
            model_dir: Folder for model persistence.
            use_rf:    If ``True``, use RandomForestClassifier; else
                       LogisticRegression.
        """
        self.model_dir = model_dir or _DEFAULT_MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)

        if use_rf:
            self.model = RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=42, n_jobs=-1,
            )
        else:
            self.model = LogisticRegression(
                max_iter=1000, random_state=42, multi_class="multinomial",
            )
        self._is_fitted = False

    # ---- training --------------------------------------------------------

    def train(self, num_samples: int = 5000) -> dict:
        """Build labelled dataset, train classifier, evaluate, and persist.

        Args:
            num_samples: Number of TriviaQA questions to use.

        Returns:
            Dict with train/test accuracy and classification report.
        """
        X, y, _ = build_labelled_dataset(num_samples)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42,
        )

        print("[classifier] Training …")
        self.model.fit(X_train, y_train)
        self._is_fitted = True

        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        y_pred = self.model.predict(X_test)
        report = classification_report(
            y_test, y_pred, target_names=list(self.LABELS.values()),
        )

        self.save()

        result = {
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "report": report,
        }
        print(f"[classifier] Train acc={train_acc:.4f}  Test acc={test_acc:.4f}")
        print(report)
        return result

    # ---- inference -------------------------------------------------------

    def predict_complexity(self, query: str) -> int:
        """Predict the complexity level of a single query.

        Args:
            query: Raw question string.

        Returns:
            Integer complexity label: 0 (SIMPLE), 1 (MEDIUM), 2 (COMPLEX).
        """
        if not self._is_fitted:
            self.load()
        feat = extract_features(query).reshape(1, -1)
        return int(self.model.predict(feat)[0])

    def predict_complexity_label(self, query: str) -> str:
        """Human-readable version of :meth:`predict_complexity`.

        Args:
            query: Raw question string.

        Returns:
            One of ``"SIMPLE"``, ``"MEDIUM"``, ``"COMPLEX"``.
        """
        return self.LABELS[self.predict_complexity(query)]

    # ---- persistence -----------------------------------------------------

    def save(self) -> None:
        """Persist the trained model to ``model_dir``."""
        path = os.path.join(self.model_dir, "complexity_classifier.joblib")
        joblib.dump(self.model, path)
        print(f"[classifier] Saved model to {path}")

    def load(self) -> None:
        """Load a previously saved model from ``model_dir``."""
        path = os.path.join(self.model_dir, "complexity_classifier.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No trained model at {path}. Run .train() first."
            )
        self.model = joblib.load(path)
        self._is_fitted = True
        print(f"[classifier] Loaded model from {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    clf = ComplexityClassifier(use_rf=False)
    clf.train(num_samples=5000)

    test_queries = [
        "Who invented the telephone?",
        "What is the difference between mitosis and meiosis?",
        "When was the Battle of Hastings and who won it?",
    ]
    for q in test_queries:
        level = clf.predict_complexity(q)
        print(f"  [{clf.LABELS[level]}] {q}")
