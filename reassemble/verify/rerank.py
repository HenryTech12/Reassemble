"""Rerank alternative strip orders with OCR readability."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..model import Reconstruction
from ..render.compose import render_strips
from .ocr import ocr_words, readability


def candidate_orders(M: np.ndarray, best_tour: list[int], n: int, beam_alts=(), k: int = 8) -> list[list[int]]:
    """Plausible alternatives to the best solution.

    Alternatives that differ from the best order *at its least certain joins* are the
    ones worth checking, so for each join we forbid it and re-optimise (the same
    computation as the min-marginal confidence).  The cheapest such alternatives plus the
    top beam-search sequences make the candidate list.
    """
    from ..solve.strips import join_min_marginals, tour_cost, tour_to_order

    deltas, tours = join_min_marginals(M, best_tour, return_tours=True)
    cands = [(tour_cost(M, best_tour), tour_to_order(list(best_tour), n))]
    for d, t in sorted(zip(deltas, tours), key=lambda x: x[0]):
        cands.append((tour_cost(M, best_tour) + d, tour_to_order(list(t), n)))
    for c, o in beam_alts:
        cands.append((c, list(o)))
    seen, out = set(), []
    for c, o in sorted(cands, key=lambda x: x[0]):
        key = tuple(o)
        if key not in seen:
            seen.add(key)
            out.append((c, o))
    return out[:k]


def rerank_candidates(prep, cands: list[tuple[float, list[int]]], langs: str = "en",
                      alpha: float = 0.03, scale: float | None = None,
                      lm_weight: float = 0.0) -> tuple[list[int], dict]:
    """Pick the candidate maximising
    ``readability + lm_weight * LM log-prob/char - alpha * extra_cost / scale``.

    The language-model term (``verify/lm.py``, English only) rewards text that reads
    like English across seams, not just tokens that happen to be dictionary words.

    ``extra_cost`` is how much worse than the best the candidate's seam cost is, in units
    of a typical join cost: OCR can overrule the pixel solver, but only when the text is
    clearly more readable.
    """
    by_id = {p.id: p for p in prep.pieces}

    def score(order):
        rec = Reconstruction(order=[prep.ids[i] for i in order])
        img = render_strips(by_id, rec, background=(255, 255, 255, 255))
        words = ocr_words(img)
        r = readability(words, langs)
        r["lm"] = _lm_score(words) if lm_weight else 0.0
        return r, words

    with ThreadPoolExecutor(max_workers=4) as ex:
        scored = list(ex.map(score, [o for _, o in cands]))
    reads = [r for r, _ in scored]
    costs = np.array([c for c, _ in cands])
    sc = scale or max(1e-9, float(np.median(np.diff(sorted(costs)))) if len(costs) > 2 else 1.0)
    total = (np.array([r["score"] for r in reads]) + lm_weight * np.array([r["lm"] for r in reads])
             - alpha * (costs - costs.min()) / sc)
    k = int(np.argmax(total))
    info = {"chosen": k, "candidates": [
        {"cost": float(c), "readability": r, "total": float(t), "order": [prep.ids[i] for i in o]}
        for c, r, t, (_, o) in zip(costs, reads, total, cands)]}
    info["words"] = scored[k][1]
    return list(cands[k][1]), info


def _lm_score(words: list[dict]) -> float:
    from .lm import default_lm
    lm = default_lm()
    if lm is None:
        return 0.0
    return lm.logprob_per_char(" ".join(w["text"] for w in words))
