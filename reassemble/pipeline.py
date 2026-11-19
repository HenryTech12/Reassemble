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
    scorer: str = "auto"                  # auto | handcrafted | learned | combined
    use_back: bool = True
    detect_gaps: bool = True
    boundary_weight: float = 2.0
    locked: list[tuple[str, str]] = field(default_factory=list)
    forbidden: list[tuple[str, str]] = field(default_factory=list)
    ocr_rerank: bool = False
    normalize: bool = False               # solve on ratio-normalised costs instead of raw
    park_blanks: bool = True
    ocr_candidates: int = 8
    ocr_alpha: float = 0.05
    langs: str = "en"
    scorer_cfg: ScorerConfig = field(default_factory=ScorerConfig)


@dataclass
class Prepared:
    pieces: list[Piece]
    feats: list[EdgeFeatures]
    back_feats: list[EdgeFeatures] | None
    ids: list[str]

    def is_blank(self, i: int) -> bool:
        """No ink on the front (and on the back, for double-sided pieces)."""
        front = self.feats[i].ink_fraction < 0.002
        if self.back_feats is None:
            return front
        return front and self.back_feats[i].ink_fraction < 0.002

    @property
    def blank(self) -> set[str]:
        return {self.ids[i] for i in range(len(self.ids)) if self.is_blank(i)}


def prepare(pieces: list[Piece]) -> Prepared:
    """Deskew/crop every piece, equalise heights and extract features."""
    norm = normalize_heights([normalize_piece(p) for p in pieces])
    amount = sharpen_amount(sharpness(norm))
    feats = [extract(p, sharpen_by=amount) for p in norm]
    backs = [extract_back(p, sharpen_by=amount) for p in norm]
    back_feats = backs if all(b is not None for b in backs) else None
    return Prepared(norm, feats, back_feats, [p.id for p in norm])


def resolve_scorer(name: str) -> str:
    """'auto' means the learned CNN when PyTorch and trained weights are available.

    The environment variable ``REASSEMBLE_SCORER`` overrides 'auto' (the test suite uses it
    to stay fast)."""
    import os
    if name != "auto":
        return name
    if os.environ.get("REASSEMBLE_SCORER"):
        return os.environ["REASSEMBLE_SCORER"]
    try:
        import torch  # noqa: F401

        from .match.learned import DEFAULT_WEIGHTS
        return "learned" if DEFAULT_WEIGHTS.exists() else "handcrafted"
    except Exception:
        return "handcrafted"


def cost_matrices(prep: Prepared, opts: SolveOptions) -> dict[str, np.ndarray]:
    back = prep.back_feats if opts.use_back else None
    comps = handcrafted_cost(prep.feats, opts.scorer_cfg, back)
    D = comps["total"]
    scorer = resolve_scorer(opts.scorer)
    comps["scorer"] = scorer
    if scorer in ("learned", "combined"):
        from .match.learned import learned_cost  # lazy: torch is optional
        L = learned_cost(prep.pieces)
        if back is not None:  # back side: same transposition trick as the handcrafted cost
            L = 0.5 * (L + learned_cost([Piece(id=p.id, image=p.back) for p in prep.pieces]).T)
        D = L if scorer == "learned" else 0.5 * (D / np.median(D[D < 1e5]) + L / np.median(L[L < 1e5]))
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
    blank_idx = [i for i in range(len(prep.ids))
                 if prep.is_blank(i) and i not in in_lock] if opts.park_blanks else []
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
    if opts.ocr_rerank:
        from .verify.rerank import candidate_orders, rerank_candidates
        # alternatives are generated on the content-only problem, so that swapping two
        # interchangeable blank strips doesn't count as an "alternative"
        lk = [(sub[a], sub[b]) for a, b in locked if a in sub and b in sub]
        Ms = build_atsp(S[np.ix_(content, content)], bscale * lin[content],
                        bscale * rin[content], lk)
        ns = len(content)
        best_sub = [sub[i] for i in best.order if i in sub]
        cands = candidate_orders(Ms, [ns] + best_sub, ns, k=opts.ocr_candidates)
        cands = [(c, full(o)) for c, o in cands]
        emit({"stage": "ocr_rerank", "candidates": len(cands)})
        chosen_order, rr = rerank_candidates(prep, cands, opts.langs, opts.ocr_alpha,
                                             scale=float(np.median(S[S < 1e5])))
        extra["ocr_rerank"] = rr
    emit({"stage": "solved", "order": [prep.ids[i] for i in chosen_order]})
    lin_b, rin_b = bscale * lin, bscale * rin
    rec = build_reconstruction(prep, chosen_order, D, opts, locked, start=lin_b, end=rin_b)
    rec.method = best.method + ("+ocr" if opts.ocr_rerank else "")
    rec.extra["scorer"] = comps["scorer"]
    if opts.ocr_rerank and "ocr_rerank" in extra:
        flag_ocr_suspects(prep, rec, extra["ocr_rerank"].pop("words"), opts.langs)
    rec.runtime_s = time.perf_counter() - t0
    rec.extra.update(extra)
    return rec


def flag_ocr_suspects(prep: Prepared, rec: Reconstruction, words: list[dict], langs: str) -> None:
    """Mark joins crossed by several non-dictionary words (``ocr_suspect`` count)."""
    from .render.compose import strip_layout
    from .verify.ocr import suspicious_joins

    layout = strip_layout({p.id: p for p in prep.pieces}, rec.order)
    x_of = {l["id"]: l["x"] for l in layout if l["id"]}
    seams = {f"{j['left']}|{j['right']}": x_of[j["right"]] for j in rec.joins if j["right"] in x_of}
    sus = suspicious_joins(words, seams, langs)
    for j in rec.joins:
        j["ocr_suspect"] = sus.get(f"{j['left']}|{j['right']}", 0)


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
                     if not (prep.is_blank(a) and prep.is_blank(b))]
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


def subset(prep: Prepared, idx: list[int]) -> Prepared:
    return Prepared([prep.pieces[i] for i in idx], [prep.feats[i] for i in idx],
                    [prep.back_feats[i] for i in idx] if prep.back_feats else None,
                    [prep.ids[i] for i in idx])


def solve_mixed(prep: Prepared, opts: SolveOptions | None = None, k: int | None = None,
                progress: Progress | None = None, labels: list[int] | None = None) -> Reconstruction:
    """Cluster pieces into documents, then solve each document separately.

    ``labels`` lets a caller (the web editor) override the clustering, e.g. after a user
    moved a piece to another document.
    """
    from .cluster.documents import cluster_documents

    t0 = time.perf_counter()
    emit = progress or (lambda e: None)
    if labels is None:
        lab, info = cluster_documents(prep.feats, k=k)
    else:
        lab, info = np.asarray(labels), {"k": len(set(labels)), "method": "user"}
    emit({"stage": "clustered", "k": int(info["k"])})
    docs = []
    for d in sorted(set(lab.tolist())):
        idx = [i for i in range(len(lab)) if lab[i] == d]
        rec = solve_prepared(subset(prep, idx), opts)
        doc = rec.to_dict()
        doc["doc"] = int(d)
        doc["pieces"] = [prep.ids[i] for i in idx]
        docs.append(doc)
        emit({"stage": "document_solved", "doc": int(d), "order": rec.order})
    docs.sort(key=lambda x: -len(x["pieces"]))
    main = Reconstruction.from_dict(docs[0])
    main.documents = docs
    main.method = (docs[0].get("method") or "") + "+cluster"
    main.runtime_s = time.perf_counter() - t0
    main.extra["clustering"] = {k_: v for k_, v in info.items() if k_ != "silhouette"}
    main.extra["labels"] = {prep.ids[i]: int(lab[i]) for i in range(len(lab))}
    return main


def detect_shape(pieces: list[Piece]) -> str:
    """Guess the shred type from piece geometry.

    Tall thin pieces are strips; pieces whose paper fills much less than their bounding
    box have irregular (torn) outlines; anything else is a cross-cut rectangle.
    """
    aspect = np.median([p.height / max(p.width, 1) for p in pieces])
    fill = np.median([p.mask.mean() for p in pieces])
    if aspect > 3:
        return "strip"
    if fill < 0.975:
        return "torn"
    return "grid"


def solve_grid_pieces(pieces: list[Piece], rows: int | None = None, cols: int | None = None,
                      scorer: str = "handcrafted") -> Reconstruction:
    from .solve.grid import solve_grid

    res, LR, TB = solve_grid(pieces, rows, cols, scorer=scorer)
    ids = [p.id for p in pieces]
    grid = [[ids[x] if x is not None else None for x in row] for row in res.grid]
    joins = []
    for r, row in enumerate(res.grid):
        for c, a in enumerate(row):
            if a is None:
                continue
            if c + 1 < len(row) and row[c + 1] is not None:
                b = row[c + 1]
                joins.append({"left": ids[a], "right": ids[b], "side": "lr", "cost": float(LR[a, b]),
                              "confidence": float(np.clip(1.5 - LR[a, b], 0, 1))})
            if r + 1 < len(res.grid) and res.grid[r + 1][c] is not None:
                b = res.grid[r + 1][c]
                joins.append({"left": ids[a], "right": ids[b], "side": "tb", "cost": float(TB[a, b]),
                              "confidence": float(np.clip(1.5 - TB[a, b], 0, 1))})
    order = [x for row in grid for x in row]
    return Reconstruction(order=order, grid=grid, joins=joins, score=res.cost, method="grid",
                          runtime_s=res.runtime_s)


def solve_torn_pieces(pieces: list[Piece]) -> Reconstruction:
    from .solve.torn import assemble

    res = assemble(pieces)
    ids = [p.id for p in pieces]
    joins = [{"left": ids[m.a], "right": ids[m.b], "side": m.direction, "cost": float(m.cost),
              "confidence": float(np.clip(1.0 - m.cost / 3.0, 0, 1)), "dx": m.dx, "dy": m.dy}
             for m in res.accepted]
    positions = {ids[i]: [int(x), int(y)] for i, (x, y) in res.positions.items()}
    return Reconstruction(order=[ids[i] for i in sorted(res.positions)], joins=joins, method="torn",
                          runtime_s=res.runtime_s, extra={"positions": positions,
                                                          "groups": [[ids[i] for i in g] for g in res.groups]})
