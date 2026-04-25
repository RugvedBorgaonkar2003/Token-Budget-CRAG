"""
budget_controller.py — Map query complexity to a retrieval budget.

The budget table is the mechanism that lets AT-CRAG spend fewer tokens
on easy questions and more tokens on hard ones.
"""

from typing import Dict


# ---------------------------------------------------------------------------
# Budget table  (tune via ablation — see eval/run_eval.py)
# ---------------------------------------------------------------------------
#   complexity  →  K (documents to retrieve)  ×  N (corrective iterations)
_BUDGET_TABLE: Dict[int, Dict[str, int]] = {
    0: {"K": 3,  "N": 1},   # SIMPLE  — shallow retrieval, 1 pass
    1: {"K": 5,  "N": 2},   # MEDIUM  — standard retrieval, 2 passes
    2: {"K": 10, "N": 3},   # COMPLEX — deep retrieval, 3 passes
}

_LABEL_NAMES = {0: "SIMPLE", 1: "MEDIUM", 2: "COMPLEX"}


def assign_budget(complexity_level: int) -> Dict[str, int]:
    """Return the retrieval budget for a given complexity level.

    Args:
        complexity_level: Integer in {0, 1, 2} produced by the
            :class:`~at_crag.complexity.classifier.ComplexityClassifier`.

    Returns:
        Dict with keys ``"K"`` (number of documents) and ``"N"``
        (maximum corrective iterations).

    Raises:
        ValueError: If *complexity_level* is not in {0, 1, 2}.
    """
    if complexity_level not in _BUDGET_TABLE:
        raise ValueError(
            f"Unknown complexity level {complexity_level!r}. "
            f"Expected one of {list(_BUDGET_TABLE.keys())}."
        )
    budget = _BUDGET_TABLE[complexity_level]
    return dict(budget)  # return a *copy* so callers can mutate safely


def get_budget_table() -> Dict[int, Dict[str, int]]:
    """Return the full budget table (read-only snapshot).

    Returns:
        Copy of the internal budget mapping.
    """
    return {k: dict(v) for k, v in _BUDGET_TABLE.items()}


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for level in (0, 1, 2):
        b = assign_budget(level)
        print(f"  complexity {level} ({_LABEL_NAMES[level]:>7s})  →  K={b['K']:>2d}, N={b['N']}")
