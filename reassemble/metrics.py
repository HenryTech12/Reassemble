"""Accuracy metrics against ground truth.

* **Neighbour accuracy**: fraction of true adjacent pairs (a directly left of b) that the
  reconstruction also has adjacent, in the same direction.  Robust to one early mistake.
* **Direct accuracy**: fraction of pieces at exactly the right index.  Harsh: a single
  piece inserted at the wrong place shifts everything after it.
* **Content** variants ignore blank strips (page margins), whose mutual order is
  genuinely undecidable and visually irrelevant.
"""

from __future__ import annotations


def true_pairs(order: list[str], missing: list[str]) -> set[tuple[str, str]]:
    miss = set(missing)
    pairs = set()
    for a, b in zip(order, order[1:]):
        if a not in miss and b not in miss:
            pairs.add((a, b))
    return pairs


def pred_pairs(order: list[str | None]) -> set[tuple[str, str]]:
    return {(a, b) for a, b in zip(order, order[1:]) if a is not None and b is not None}


def neighbour_accuracy(pred: list[str | None], truth_order: list[str], missing=(),
                       ignore: set[str] | None = None) -> float:
    if ignore:
        pred = [p for p in pred if p not in ignore]
        truth_order = [p for p in truth_order if p not in ignore]
    tp = true_pairs(truth_order, list(missing))
    if not tp:
        return 1.0
    return len(tp & pred_pairs(pred)) / len(tp)


def direct_accuracy(pred: list[str | None], truth_order: list[str], missing=(),
                    ignore: set[str] | None = None) -> float:
    miss = set(missing) | (ignore or set())
    t = [p for p in truth_order if p not in miss]
    q = [p for p in pred if p is not None and p not in miss]
    if not t:
        return 1.0
    return sum(1 for a, b in zip(t, q) if a == b) / len(t)


def gap_detection(pred: list[str | None], truth_order: list[str], missing) -> dict:
    """Precision/recall of predicted gap markers versus real missing-piece positions."""
    miss = set(missing)
    present = [p for p in truth_order if p not in miss]
    real = set()
    for a, b in zip(present, present[1:]):
        ia, ib = truth_order.index(a), truth_order.index(b)
        if ib - ia > 1:
            real.add((a, b))
    found = set()
    for i, p in enumerate(pred):
        if p is None and 0 < i < len(pred) - 1 and pred[i - 1] and pred[i + 1]:
            found.add((pred[i - 1], pred[i + 1]))
    tp = len(real & found)
    return {"real": len(real), "predicted": len(found), "correct": tp,
            "precision": tp / len(found) if found else (1.0 if not real else 0.0),
            "recall": tp / len(real) if real else 1.0}


def evaluate_strips(pred: list[str | None], truth: dict, blank: set[str] | None = None) -> dict:
    order, missing = truth["order"], truth.get("missing", [])
    out = {
        "neighbour": neighbour_accuracy(pred, order, missing),
        "direct": direct_accuracy(pred, order, missing),
    }
    if blank:
        out["neighbour_content"] = neighbour_accuracy(pred, order, missing, ignore=blank)
        out["direct_content"] = direct_accuracy(pred, order, missing, ignore=blank)
    if missing:
        out["gaps"] = gap_detection(pred, order, missing)
    return out
