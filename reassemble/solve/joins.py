"""Missing-piece (gap) detection for solved strip orders."""

from __future__ import annotations

import numpy as np


def detect_gaps(rel_cost: np.ndarray, ratio: np.ndarray, informative: np.ndarray,
                min_rel: float = 1.5, min_ratio: float = 1.2) -> np.ndarray:
    """Flag joins that are probably a missing strip rather than a real seam.

    When a strip is missing, its two neighbours get forced together.  Such a join shows
    two symptoms at once:

    * its cost is high compared with the typical join of this page (``rel_cost`` =
      cost / median join cost), and
    * neither piece prefers the other: the ratio-normalised cost (``ratio`` = cost /
      second-best alternative, see ``normalize_costs``) is above 1.

    Requiring both keeps false alarms low; correct-but-noisy seams usually only show one
    symptom.  Joins between blank strips (``informative`` False) are never flagged —
    there is no evidence either way.  Smooth photo regions are the known weak spot: a
    gradient extrapolates across a missing strip almost seamlessly.
    """
    rel_cost, ratio, informative = map(np.asarray, (rel_cost, ratio, informative))
    return (rel_cost > min_rel) & (ratio > min_ratio) & informative


# --------------------------------------------------------------------------- learned gap model

FEATURES = ["rel_cost", "ratio", "confidence", "rank_out", "rank_in", "ink_left", "ink_right",
            "rel_cost_vs_neighbours"]


def join_features(order: list[int], D: np.ndarray, N: np.ndarray, confs: np.ndarray,
                  ink_right: np.ndarray, ink_left: np.ndarray, med: float) -> np.ndarray:
    """One feature row per join of ``order`` (see ``FEATURES``).

    The intuition behind each: a missing strip makes the forced join *expensive*
    (rel_cost), *unpreferred* by both pieces (ratio, ranks), *not forced* by the rest of
    the solution (confidence), happen between *inky* edges (a blank seam can't reveal a
    gap), and stand out against the joins *around it* (rel_cost_vs_neighbours).
    """
    rows = []
    costs = np.array([D[a, b] for a, b in zip(order, order[1:])])
    for k, (a, b) in enumerate(zip(order, order[1:])):
        row_a = np.delete(D[a], a)
        col_b = np.delete(D[:, b], b)
        nb = costs[max(0, k - 3):k + 4]
        rows.append([costs[k] / max(med, 1e-9), N[a, b], confs[k],
                     float((row_a < D[a, b]).sum()), float((col_b < D[a, b]).sum()),
                     ink_right[a], ink_left[b], costs[k] / (np.median(nb) + 1e-9)])
    return np.array(rows, dtype=np.float64).reshape(-1, len(FEATURES))


def gap_model_path(scorer: str):
    from pathlib import Path
    return Path(__file__).resolve().parents[2] / "models" / f"gap_clf_{scorer}.pkl"


_GAP_MODELS: dict = {}


def load_gap_model(scorer: str):
    import pickle
    if scorer not in _GAP_MODELS:
        p = gap_model_path(scorer)
        _GAP_MODELS[scorer] = pickle.loads(p.read_bytes()) if p.exists() else None
    return _GAP_MODELS[scorer]


def train_gap_model(scorer: str = "handcrafted", n_pages: int = 160, seed: int = 0,
                    threshold: float = 0.5, verbose: bool = True) -> dict:
    """Fit a gradient-boosting classifier "is this join really a missing strip?".

    Training data: generator pages (clean and noisy) with 0-3 random strips removed,
    solved *without* gap detection; each join of the solution is labelled by whether
    strips are missing between its two pieces in the ground truth.
    """
    import pickle
    import random

    from sklearn.ensemble import HistGradientBoostingClassifier

    from ..pipeline import SolveOptions, prepare, solve_prepared
    from ..synth.pages import render_page
    from ..synth.shredders import Realism, strip_shred

    rng = random.Random(seed)
    X, y = [], []
    kinds = ["memo", "letter", "invoice", "photo", "handwriting"]
    for i in range(n_pages):
        kind = rng.choice(kinds)
        s = 10_000 + seed * 1000 + i
        page, _ = render_page(kind, s)
        drop = rng.choice([0, 1, 2, 2, 3])
        r = Realism.noisy(drop) if rng.random() < 0.7 else Realism(drop=drop)
        pieces, truth = strip_shred(page, rng.randint(10, 60), r, seed=s)
        prep = prepare(pieces)
        rec = solve_prepared(prep, SolveOptions(scorer=scorer, detect_gaps=False))
        feats = rec.extra.get("join_features")
        if feats is None:
            continue
        order, miss = truth["order"], set(truth["missing"])
        for j, f in zip(rec.joins, feats):
            ia, ib = order.index(j["left"]), order.index(j["right"])
            X.append(f)
            y.append(int(ib > ia + 1 and all(order[t] in miss for t in range(ia + 1, ib))))
        if verbose and (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n_pages} pages, {sum(y)} gap joins of {len(y)}")
    X, y = np.array(X), np.array(y)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight="balanced",
                                         random_state=seed).fit(X, y)
    model = {"clf": clf, "threshold": threshold, "features": FEATURES}
    gap_model_path(scorer).write_bytes(pickle.dumps(model))
    _GAP_MODELS.pop(scorer, None)
    return {"joins": len(y), "gaps": int(y.sum())}
