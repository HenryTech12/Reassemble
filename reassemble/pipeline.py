"""High-level entry points that wire features -> matcher -> solver -> Reconstruction."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .features.edges import EdgeFeatures, extract, extract_back, sharpen_amount, sharpness
from .features.normalize import normalize_heights, normalize_piece
from .match.handcrafted import ScorerConfig, edge_inkiness, handcrafted_cost, normalize_costs
from .model import Piece, Reconstruction
from .solve.joins import detect_gaps
from .solve.strips import build_atsp, join_min_marginals, solve_strips

Progress = Callable[[dict], None]


@dataclass
class SolveOptions:
    methods: tuple[str, ...] = ("greedy", "beam", "atsp")
    beam_width: int = 200
    top_k: int = 10
    scorer: str = "handcrafted"           # handcrafted | learned | combined
    use_back: bool = True
    detect_gaps: bool = True
    boundary_weight: float = 2.0
    locked: list[tuple[str, str]] = field(default_factory=list)
    forbidden: list[tuple[str, str]] = field(default_factory=list)
    ocr_rerank: bool = False
    normalize: bool = False               # solve on ratio-normalised costs instead of raw
    park_blanks: bool = True
    scorer_cfg: ScorerConfig = field(default_factory=ScorerConfig)


@dataclass
class Prepared:
    pieces: list[Piece]
    feats: list[EdgeFeatures]
    back_feats: list[EdgeFeatures] | None
    ids: list[str]

    @property
    def blank(self) -> set[str]:
        return {f.id for f in self.feats if f.ink_fraction < 0.002}


def prepare(pieces: list[Piece]) -> Prepared:
    """Deskew/crop every piece, equalise heights and extract features."""
    norm = normalize_heights([normalize_piece(p) for p in pieces])
    amount = sharpen_amount(sharpness(norm))
    feats = [extract(p, sharpen_by=amount) for p in norm]
    backs = [extract_back(p, sharpen_by=amount) for p in norm]
    back_feats = backs if all(b is not None for b in backs) else None
    return Prepared(norm, feats, back_feats, [p.id for p in norm])


def cost_matrices(prep: Prepared, opts: SolveOptions) -> dict[str, np.ndarray]:
    back = prep.back_feats if opts.use_back else None
    comps = handcrafted_cost(prep.feats, opts.scorer_cfg, back)
    D = comps["total"]
    if opts.scorer in ("learned", "combined"):
        from .match.learned import learned_cost  # lazy: torch is optional
        L = learned_cost(prep.pieces)
        D = L if opts.scorer == "learned" else 0.5 * (D / np.median(D[D < 1e5]) + L / np.median(L[L < 1e5]))
        np.fill_diagonal(D, 1e6)
    comps["raw"] = D
    comps["norm"] = normalize_costs(D)
    return comps


def solve_prepared(prep: Prepared, opts: SolveOptions | None = None,
                   progress: Progress | None = None) -> Reconstruction:
    opts = opts or SolveOptions()
    t0 = time.perf_counter()
    emit = progress or (lambda e: None)
    emit({"stage": "features", "n": len(prep.ids)})
    comps = cost_matrices(prep, opts)
    D, N = comps["raw"], comps["norm"]
    emit({"stage": "scored"})
    lin, rin = edge_inkiness(prep.feats)
    idx = {pid: i for i, pid in enumerate(prep.ids)}
    locked = [(idx[a], idx[b]) for a, b in opts.locked if a in idx and b in idx]
    forbidden = [(idx[a], idx[b]) for a, b in opts.forbidden if a in idx and b in idx]
    S = N if opts.normalize else D
    bscale = float(np.median(S[S < 1e5])) * opts.boundary_weight
    # Blank strips (pure margin) carry no information, so we take them out of the
    # optimisation and park them at the page edges afterwards.
    in_lock = {i for pair in locked for i in pair}
    blank_idx = [i for i, f in enumerate(prep.feats)
                 if f.ink_fraction < 0.002 and i not in in_lock] if opts.park_blanks else []
    content = [i for i in range(len(prep.ids)) if i not in set(blank_idx)]
    if len(content) < 2:
        content, blank_idx = list(range(len(prep.ids))), []
    sub = {g: k for k, g in enumerate(content)}
    res = solve_strips(S[np.ix_(content, content)], bscale * lin[content], bscale * rin[content],
                       [(sub[a], sub[b]) for a, b in locked if a in sub and b in sub],
                       [(sub[a], sub[b]) for a, b in forbidden if a in sub and b in sub],
                       opts.methods, opts.beam_width, opts.top_k)

    def full(order_sub: list[int]) -> list[int]:
        return park_blanks([content[i] for i in order_sub], blank_idx, prep.feats)

    for r in {id(v): v for v in res.values()}.values():
        r.order = full(r.order)
        r.alternatives = [(c, full(o)) for c, o in r.alternatives]
    best = res["best"]
    chosen_order = best.order
    extra: dict = {"methods": {k: {"cost": v.cost, "runtime_s": v.runtime_s,
                                   "order": [prep.ids[i] for i in v.order]}
                               for k, v in res.items() if k != "best"}}
    if opts.ocr_rerank and "beam" in res:
        from .verify.rerank import rerank_candidates
        cands = [(c, o) for c, o in res["beam"].alternatives]
        if res["best"].order not in [o for _, o in cands]:
            cands.insert(0, (best.cost, best.order))
        emit({"stage": "ocr_rerank", "candidates": len(cands)})
        chosen_order, rr = rerank_candidates(prep, cands)
        extra["ocr_rerank"] = rr
    emit({"stage": "solved", "order": [prep.ids[i] for i in chosen_order]})
    lin_b, rin_b = bscale * lin, bscale * rin
    rec = build_reconstruction(prep, chosen_order, D, opts, locked, start=lin_b, end=rin_b)
    rec.method = best.method + ("+ocr" if opts.ocr_rerank else "")
    rec.runtime_s = time.perf_counter() - t0
    rec.extra.update(extra)
    return rec


def park_blanks(order: list[int], blanks: list[int], feats: list[EdgeFeatures]) -> list[int]:
    """Put blank strips at the two ends so the left and right margins come out balanced.

    Printed pages usually have similar left and right margins.  We measure how much blank
    paper is already at each end of the solved content and add blank strips, widest first,
    to whichever side currently has the narrower margin.
    """
    if not blanks:
        return list(order)
    left_m = feats[order[0]].extra.get("blank_left", 0) if order else 0
    right_m = feats[order[-1]].extra.get("blank_right", 0) if order else 0
    left, right = [], []
    for b in sorted(blanks, key=lambda i: -feats[i].width):
        if left_m <= right_m:
            left.insert(0, b)
            left_m += feats[b].width
        else:
            right.append(b)
            right_m += feats[b].width
    return left + list(order) + right


def build_reconstruction(prep: Prepared, order: list[int], D: np.ndarray, opts: SolveOptions,
                         locked_idx=(), start=None, end=None) -> Reconstruction:
    """Attach per-join confidences and gap markers to a solved order.

    Confidence is ``1 - exp(-delta / scale)`` where ``delta`` is the join's min-marginal
    (cost increase of the best solution when the join is forbidden) and ``scale`` is a
    fraction of the typical join cost.
    """
    n = len(order)
    costs = np.array([D[a, b] for a, b in zip(order, order[1:])])
    M = build_atsp(D, start, end)
    tour = [len(D)] + list(order)
    mm = np.array(join_min_marginals(M, tour))[1:n]  # edges between consecutive pieces
    content_costs = [c for (a, b), c in zip(zip(order, order[1:]), costs)
                     if prep.feats[a].ink_fraction >= 0.002 or prep.feats[b].ink_fraction >= 0.002]
    scale = 0.08 * float(np.median(content_costs)) if content_costs else 1.0
    confs = 1.0 - np.exp(-mm / max(scale, 1e-9))
    locked_set = set(locked_idx)
    blank = prep.blank
    N = normalize_costs(D)
    informative = np.array([prep.ids[a] not in blank and prep.ids[b] not in blank
                            for a, b in zip(order, order[1:])], bool)
    med = float(np.median(content_costs)) if content_costs else 1.0
    gaps = (detect_gaps(costs / max(med, 1e-9), [N[a, b] for a, b in zip(order, order[1:])],
                        informative) if opts.detect_gaps and len(costs) >= 4
            else np.zeros(len(costs), bool))
    out_order: list[str | None] = [prep.ids[order[0]]] if order else []
    joins, gap_list = [], []
    med_w = float(np.median([f.width for f in prep.feats])) if prep.feats else 0
    for k, (a, b) in enumerate(zip(order, order[1:])):
        is_locked = (a, b) in locked_set
        if gaps[k] and not is_locked:
            out_order.append(None)
            gap_list.append({"after": prep.ids[a], "before": prep.ids[b],
                             "est_width": med_w, "cost": float(costs[k])})
        joins.append({"left": prep.ids[a], "right": prep.ids[b], "cost": float(costs[k]),
                      "confidence": 1.0 if is_locked else float(confs[k]),
                      "gap": bool(gaps[k] and not is_locked), "locked": is_locked,
                      "blank": prep.ids[a] in blank and prep.ids[b] in blank})
        out_order.append(prep.ids[b])
    score = float(np.mean(costs)) if len(costs) else 0.0
    return Reconstruction(order=out_order, joins=joins, gaps=gap_list, score=score,
                          extra={"blank": sorted(blank)})


def solve_pieces(pieces: list[Piece], opts: SolveOptions | None = None,
                 progress: Progress | None = None) -> Reconstruction:
    return solve_prepared(prepare(pieces), opts, progress)
