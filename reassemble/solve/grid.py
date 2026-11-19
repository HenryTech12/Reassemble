"""Cross-cut (2D jigsaw) solver for rectangular pieces of known orientation.

Every piece now has four neighbours, so we need two cost matrices:

* ``LR[a, b]``: b placed right of a (the strip matcher, unchanged);
* ``TB[a, b]``: b placed below a.  We get it by **transposing** every piece (swapping
  x and y), which turns "bottom edge vs top edge" into "right edge vs left edge", so the
  very same matcher can be reused.

Placement follows the greedy idea of Pomeranz et al. (2011) and Paikin & Tal (2015):
start from the most trustworthy piece and repeatedly fill the empty slot / piece pair we
are most confident about, where confidence prefers slots surrounded by several already
placed pieces and *best-buddy* agreements.  A local-search refinement then swaps pieces
while the total seam cost drops.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..features.edges import extract
from ..match.handcrafted import ScorerConfig, pixel_cost_matrix
from ..model import Piece

DIRS = {(0, 1): "lr", (0, -1): "rl", (1, 0): "tb", (-1, 0): "bt"}


def transpose_piece(p: Piece) -> Piece:
    return Piece(id=p.id, image=np.ascontiguousarray(p.image.transpose(1, 0, 2)))


def border_costs(pieces: list[Piece]) -> dict[str, np.ndarray]:
    """Ink on each outer edge (0 = blank).  Pieces on the page border normally have a
    blank outer edge (the margin), so ink there is penalised."""
    from ..match.handcrafted import edge_inkiness
    f = [extract(p) for p in pieces]
    ft = [extract(transpose_piece(p)) for p in pieces]
    left, right = edge_inkiness(f)
    top, bottom = edge_inkiness(ft)
    return {"left": left, "right": right, "top": top, "bottom": bottom}


def line_phase_cost(pieces: list[Piece]) -> np.ndarray:
    """TB cue from text-line periodicity.

    Text lines repeat every ``s`` pixels down the page.  If B really sits below A, then
    the last ``s`` rows of A and the first ``s`` rows of B are one period apart, so their
    "inside a text line" profiles must agree — even when the cut itself falls in the
    blank space between two lines, where the pixel matcher sees nothing.
    Pairs without text lines on both sides get the neutral cost 0.5.
    """
    from scipy.ndimage import gaussian_filter1d
    feats = [extract(p) for p in pieces]
    sp = [f.line_spacing for f in feats if f.line_spacing > 0]
    n = len(pieces)
    out = np.full((n, n), 0.5)
    if not sp:
        return out
    s = int(round(np.median(sp)))
    bands = []
    for f in feats:
        p = f.profile
        on = (p > max(p.max() * 0.1, 1e-6)).astype(np.float32) if p.max() > 0 else np.zeros_like(p)
        bands.append(gaussian_filter1d(on, 1.0))
    for a in range(n):
        ba = bands[a]
        if len(ba) < s or ba.max() < 0.5:
            continue
        tail = ba[-s:]
        for b in range(n):
            bb = bands[b]
            if a == b or len(bb) < s or bb.max() < 0.5:
                continue
            out[a, b] = float(np.abs(tail - bb[:s]).mean())
    return out


def grid_costs(pieces: list[Piece], cfg: ScorerConfig | None = None, scorer: str = "handcrafted",
               w_phase: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(LR, TB) dissimilarity matrices, each divided by its median."""
    cfg = cfg or ScorerConfig(shift=0)
    tps = [transpose_piece(p) for p in pieces]
    if scorer == "learned":
        from ..match.learned import learned_cost
        LR, TB = learned_cost(pieces), learned_cost(tps)
    else:
        LR = pixel_cost_matrix([extract(p) for p in pieces], cfg=cfg)
        TB = pixel_cost_matrix([extract(p) for p in tps], cfg=cfg)
    # Ratio normalisation (cost / second-best alternative, both sides): in 2D a raw cost
    # makes blank-against-blank seams almost free, so wrong layouts that park all the
    # inky edges on the outer border look cheaper than the truth.  Ratios ask "how much
    # better than the alternatives is this?", which is ~1 for uninformative blank seams.
    from ..match.handcrafted import normalize_costs
    if w_phase:
        ph = line_phase_cost(pieces)
        TB = TB / (np.median(TB[np.isfinite(TB)]) + 1e-9) + w_phase * ph / (np.median(ph) + 1e-9)
    LR, TB = normalize_costs(np.where(np.isfinite(LR), LR, 1e6)), normalize_costs(np.where(np.isfinite(TB), TB, 1e6))
    for M in (LR, TB):
        np.fill_diagonal(M, np.inf)
    return LR, TB


def guess_grid_shape(n: int, piece_w: float, piece_h: float, page_aspect: float = 1100 / 850) -> tuple[int, int]:
    """Choose rows x cols = n whose overall aspect ratio is closest to a portrait page."""
    best, shape = np.inf, (1, n)
    for r in range(1, n + 1):
        if n % r:
            continue
        c = n // r
        err = abs(np.log((r * piece_h) / (c * piece_w)) - np.log(page_aspect))
        if err < best:
            best, shape = err, (r, c)
    return shape


@dataclass
class GridResult:
    grid: list[list[int | None]]
    cost: float
    runtime_s: float


def _cost_with(LR, TB, grid, r, c, p):
    """Mean seam cost of placing piece p at (r, c) given its placed neighbours."""
    R, C = grid.shape
    costs = []
    if c > 0 and grid[r, c - 1] >= 0:
        costs.append(LR[grid[r, c - 1], p])
    if c + 1 < C and grid[r, c + 1] >= 0:
        costs.append(LR[p, grid[r, c + 1]])
    if r > 0 and grid[r - 1, c] >= 0:
        costs.append(TB[grid[r - 1, c], p])
    if r + 1 < R and grid[r + 1, c] >= 0:
        costs.append(TB[p, grid[r + 1, c]])
    return costs


def total_cost(LR, TB, grid, border: dict | None = None, w_border: float = 1.0) -> float:
    """Sum of all internal seam costs (+ optional penalty for ink on the outer border)."""
    g = np.asarray(grid)
    full = (g >= 0).all()
    if full:
        s = LR[g[:, :-1], g[:, 1:]].sum() + TB[g[:-1], g[1:]].sum()
    else:
        s = 0.0
        R, C = g.shape
        for r in range(R):
            for c in range(C):
                a = g[r, c]
                if a < 0:
                    continue
                if c + 1 < C and g[r, c + 1] >= 0:
                    s += LR[a, g[r, c + 1]]
                if r + 1 < R and g[r + 1, c] >= 0:
                    s += TB[a, g[r + 1, c]]
    if border is not None and full:
        s += w_border * (border["left"][g[:, 0]].sum() + border["right"][g[:, -1]].sum()
                         + border["top"][g[0]].sum() + border["bottom"][g[-1]].sum())
    return float(s)


def greedy_place(LR: np.ndarray, TB: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Greedy placement on a *floating* grid that is clipped to rows x cols.

    We place on a canvas twice the size, start in the middle, and only allow placements
    that keep the occupied bounding box within rows x cols — so the solver discovers
    where the borders are instead of having to guess the first corner piece.
    """
    n = len(LR)
    # best-buddy counts tell us which piece is the most reliable seed
    bb = np.zeros(n)
    for M in (LR, TB):
        Mf = np.where(np.isfinite(M), M, np.inf)
        bs, bp = Mf.argmin(1), Mf.argmin(0)
        bb += (bp[bs] == np.arange(n))
        bb += (bs[bp] == np.arange(n))
    R2, C2 = 2 * rows + 1, 2 * cols + 1
    grid = -np.ones((R2, C2), int)
    seed = int(np.argmax(bb))
    grid[rows, cols] = seed
    used = {seed}
    # second-best margins for "confidence" of a slot/piece choice
    while len(used) < n:
        occ = np.argwhere(grid >= 0)
        rmin, cmin = occ.min(0)
        rmax, cmax = occ.max(0)
        best = (np.inf, None, None)
        cand_slots = set()
        for r, c in occ:
            for dr, dc in DIRS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < R2 and 0 <= cc < C2 and grid[rr, cc] < 0:
                    nr = max(rmax, rr) - min(rmin, rr) + 1
                    nc = max(cmax, cc) - min(cmin, cc) + 1
                    if nr <= rows and nc <= cols:
                        cand_slots.add((rr, cc))
        unused = [p for p in range(n) if p not in used]
        for rr, cc in cand_slots:
            scores = []
            for p in unused:
                cs = _cost_with(LR, TB, grid, rr, cc, p)
                scores.append(np.mean(cs))
            scores = np.array(scores)
            order = np.argsort(scores)
            k = len(_cost_with(LR, TB, grid, rr, cc, unused[0]))
            first = scores[order[0]]
            second = scores[order[1]] if len(order) > 1 else first * 2 + 1
            # lower is better: own cost relative to the runner-up, rewarded for more
            # placed neighbours (more evidence)
            key = (first / (second + 1e-9)) - 0.15 * (k - 1)
            if key < best[0]:
                best = (key, (rr, cc), unused[order[0]])
        if best[1] is None:  # no slot available (shouldn't happen), fall back
            break
        (rr, cc), p = best[1], best[2]
        grid[rr, cc] = p
        used.add(p)
    occ = np.argwhere(grid >= 0)
    rmin, cmin = occ.min(0)
    out = -np.ones((rows, cols), int)
    for r, c in occ:
        out[r - rmin, c - cmin] = grid[r, c]
    return out


def refine_swaps(LR, TB, grid: np.ndarray, max_rounds: int = 5, border=None) -> np.ndarray:
    """Local search: swap any two cells, or cyclically shift the whole layout by one
    row/column, whenever it lowers the total cost (seams + border prior)."""
    g = grid.copy()
    cells = [tuple(x) for x in np.argwhere(np.ones_like(g, bool))]
    cur = total_cost(LR, TB, g, border)
    for axis in (0, 1):  # whole-layout cyclic shifts fix a correct but offset assembly
        for k in range(1, g.shape[axis]):
            cand = np.roll(g, k, axis=axis)
            c = total_cost(LR, TB, cand, border)
            if c < cur - 1e-9:
                g, cur = cand, c
    for _ in range(max_rounds):
        improved = False
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                a, b = cells[i], cells[j]
                g[a], g[b] = g[b], g[a]
                c = total_cost(LR, TB, g, border)
                if c < cur - 1e-9:
                    cur, improved = c, True
                else:
                    g[a], g[b] = g[b], g[a]
        if not improved:
            break
    return g


def rows_then_stack(LR: np.ndarray, TB: np.ndarray, rows: int, cols: int, border: dict) -> np.ndarray:
    """Two-stage cross-cut solver.

    1. **Rows**: an ATSP over all pieces with ``rows`` copies of the page-border node.  A
       tour must pass through every border copy once, so it splits into ``rows`` chains —
       the strip solver from Phase 1, generalised to several strips at once.  Row starts
       and ends are cheap where the piece's outer edge is blank (the page margin).
    2. **Stacking**: rows are then ordered top-to-bottom with another ATSP whose cost is
       the *sum* of TB costs down every column.  Individual horizontal cuts are often
       ambiguous (they fall between text lines), but a whole row almost always cuts through
       some text, so the summed cost is reliable.
    """
    from ..solve.strips import BIG, or_opt_2opt, tour_cost

    n = len(LR)
    m = n + rows
    M = np.full((m, m), BIG)
    finite = np.where(np.isfinite(LR), LR, BIG)
    M[:n, :n] = finite
    scale = float(np.median(finite[finite < BIG]))
    for k in range(rows):
        M[n + k, :n] = scale * border["left"]
        M[:n, n + k] = scale * border["right"]
    np.fill_diagonal(M, BIG)
    # seed tour: greedy nearest neighbour, then local search
    start = n
    tour, used = [start], {start}
    while len(tour) < m:
        last = tour[-1]
        cand = [j for j in np.argsort(M[last]) if j not in used]
        # force a border node every `cols` pieces in the seed, to start balanced
        if (len(tour) % (cols + 1)) == 0:
            cand = [j for j in cand if j >= n] or cand
        else:
            cand = [j for j in cand if j < n] or cand
        tour.append(int(cand[0]))
        used.add(int(cand[0]))
    tour = or_opt_2opt(M, tour)
    # split into chains at border nodes
    k = next(i for i, x in enumerate(tour) if x >= n)
    tour = tour[k:] + tour[:k]
    chains, cur = [], []
    for x in tour[1:] + [tour[0]]:
        if x >= n:
            chains.append(cur)
            cur = []
        else:
            cur.append(x)
    # rebalance chain lengths to exactly `cols`: cut long chains at their worst join
    pool = [c for c in chains if c]
    pieces_flat = [p for c in pool for p in c]
    rowsets: list[list[int]] = []
    for c in pool:
        while len(c) > cols:
            costs = [LR[a, b] for a, b in zip(c, c[1:])]
            # cut so that one part has a multiple of cols if possible
            cut = max(range(len(costs)), key=lambda i: costs[i] + (0 if (i + 1) % cols == 0 else -1e3))
            rowsets.append(c[:cut + 1])
            c = c[cut + 1:]
        rowsets.append(c)
    # merge short fragments greedily (cheapest LR join) until we have `rows` rows
    rowsets.sort(key=len, reverse=True)
    while len(rowsets) > rows:
        frag = rowsets.pop()  # shortest
        best, where = np.inf, None
        for i, r in enumerate(rowsets):
            if len(r) + len(frag) <= cols:
                for left, right, pos in ((r, frag, "after"), (frag, r, "before")):
                    c = LR[left[-1], right[0]]
                    if c < best:
                        best, where = c, (i, pos)
        if where is None:
            i = min(range(len(rowsets)), key=lambda i: len(rowsets[i]))
            where = (i, "after")
        i, pos = where
        rowsets[i] = rowsets[i] + frag if pos == "after" else frag + rowsets[i]
    while len(rowsets) < rows:
        rowsets.append([])
    # stack rows: ATSP over rows with summed TB cost
    R = len(rowsets)
    S = np.full((R + 1, R + 1), BIG)
    tbs = float(np.median(TB[np.isfinite(TB)]))
    for i, a in enumerate(rowsets):
        for j, b in enumerate(rowsets):
            if i != j:
                k_ = min(len(a), len(b))
                S[i, j] = sum(TB[a[c], b[c]] for c in range(k_)) + tbs * abs(len(a) - len(b))
        S[R, i] = tbs * border["top"][a].sum() if a else 0
        S[i, R] = tbs * border["bottom"][a].sum() if a else 0
    order = list(range(R))
    best_t, best_c = None, np.inf
    import itertools
    if R <= 7:
        for perm in itertools.permutations(order):
            c = tour_cost(S, [R] + list(perm))
            if c < best_c:
                best_t, best_c = list(perm), c
    else:
        t = or_opt_2opt(S, [R] + order)
        k = t.index(R)
        best_t = t[k + 1:] + t[:k]
    g = -np.ones((rows, cols), int)
    leftovers = []
    for r, ri in enumerate(best_t):
        row = rowsets[ri]
        for c, p in enumerate(row[:cols]):
            g[r, c] = p
        leftovers += row[cols:]
    for p in leftovers:
        empty = np.argwhere(g < 0)
        if len(empty):
            g[tuple(empty[0])] = p
    return g


def solve_grid(pieces: list[Piece], rows: int | None = None, cols: int | None = None,
               refine: bool = True, scorer: str = "handcrafted") -> tuple[GridResult, np.ndarray, np.ndarray]:
    t0 = time.perf_counter()
    n = len(pieces)
    if rows is None or cols is None:
        rows, cols = guess_grid_shape(n, np.median([p.width for p in pieces]),
                                      np.median([p.height for p in pieces]))
    LR, TB = grid_costs(pieces, scorer=scorer)
    border = border_costs(pieces)
    cands = [greedy_place(LR, TB, rows, cols), rows_then_stack(LR, TB, rows, cols, border)]
    if refine and n <= 300:
        cands = [refine_swaps(LR, TB, g, border=border) for g in cands]
    g = min(cands, key=lambda g: total_cost(LR, TB, g, border) if (g >= 0).all() else np.inf)
    grid = [[int(x) if x >= 0 else None for x in row] for row in g]
    return GridResult(grid, total_cost(LR, TB, g, border), time.perf_counter() - t0), LR, TB


def evaluate_grid(pred: list[list[str | None]], truth_grid: list[list[str]], blank: set[str] | None = None) -> dict:
    """Neighbour accuracy (LR+TB pairs) and direct accuracy (same absolute cell)."""
    blank = blank or set()

    def pairs(g):
        out = set()
        for r, row in enumerate(g):
            for c, a in enumerate(row):
                if a is None:
                    continue
                if c + 1 < len(row) and row[c + 1] is not None:
                    out.add(("lr", a, row[c + 1]))
                if r + 1 < len(g) and c < len(g[r + 1]) and g[r + 1][c] is not None:
                    out.add(("tb", a, g[r + 1][c]))
        return out

    tp, pp = pairs(truth_grid), pairs(pred)
    content = {x for x in tp if x[1] not in blank or x[2] not in blank}
    pos = {a: (r, c) for r, row in enumerate(truth_grid) for c, a in enumerate(row)}
    cells = [(pos[a], (r, c)) for r, row in enumerate(pred) for c, a in enumerate(row) if a is not None]
    direct = np.mean([t == p for t, p in cells]) if cells else 0.0
    return {"neighbour": len(tp & pp) / len(tp),
            "neighbour_content": len(content & pp) / max(1, len(content)),
            "direct": float(direct)}
