"""Torn-paper reconstruction by contour shape complementarity.

A tear is a shared curve: wherever piece A's right boundary bulges out, piece B's left
boundary has the matching dent.  So if we place B at offset (dx, dy) relative to A,
``x_A(y) + 1 - x_B(y - dy)`` should be the *same number* (= dx) for every overlapping row.
The spread of that difference is our shape cost; its median is the offset.

We search all vertical offsets dy (pieces of a torn page are not aligned), add a light
content-continuity term (colours a few pixels inside each boundary, past the white fibre
rim), and assemble pieces Kruskal-style: accept the best remaining match whose placement
doesn't overlap pieces already in the group.  Top/bottom matching reuses the same code on
transposed pieces.

Assumes pieces are upright (no rotation); see README limitations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..model import Piece
from .grid import transpose_piece


@dataclass
class Match:
    a: int
    b: int
    dx: int
    dy: int
    cost: float
    direction: str


def boundary_profile(piece: Piece, side: str) -> tuple[np.ndarray, np.ndarray]:
    """x of the outermost paper pixel per row (NaN where the row has no paper)."""
    m = piece.mask
    has = m.any(1)
    if side == "right":
        x = piece.width - 1 - np.argmax(m[:, ::-1], axis=1)
    else:
        x = np.argmax(m, axis=1)
    x = x.astype(np.float64)
    # Rows whose boundary sits exactly on the crop's extreme column are page-edge rows
    # (the original sheet's straight border), not tear: they carry no shape information.
    extreme = piece.width - 1 if side == "right" else 0
    has = has & (x != extreme)
    x[~has] = np.nan
    return x, has


def _inside_colour(piece: Piece, xs: np.ndarray, side: str, depth: int = 4) -> np.ndarray:
    rows = np.arange(piece.height)
    step = -depth if side == "right" else depth
    cols = np.clip(np.nan_to_num(xs, nan=0).astype(int) + step, 0, piece.width - 1)
    return piece.rgb[rows, cols].astype(np.float32)


def pair_match(A: Piece, B: Piece, min_overlap: float = 0.4) -> tuple[float, int, int]:
    """Best (cost, dx, dy) for B to the right of A."""
    xa, ha = boundary_profile(A, "right")
    xb, hb = boundary_profile(B, "left")
    ca = _inside_colour(A, xa, "right")
    cb = _inside_colour(B, xb, "left")
    # a side that is (almost) all page edge has nothing to match
    if ha.mean() < 0.3 or hb.mean() < 0.3:
        return np.inf, 0, 0
    HA, HB = A.height, B.height
    need = min_overlap * min(ha.sum(), hb.sum())
    best = (np.inf, 0, 0)
    for dy in range(-HB + 1, HA):
        ya0, ya1 = max(0, dy), min(HA, HB + dy)
        if ya1 - ya0 < need:
            continue
        sa = slice(ya0, ya1)
        sb = slice(ya0 - dy, ya1 - dy)
        v = ha[sa] & hb[sb]
        if v.sum() < need:
            continue
        diff = (xa[sa] + 1 - xb[sb])[v]
        dx = np.median(diff)
        res = np.abs(diff - dx)
        # trimmed mean: ignore the 20 % worst rows (corners where another tear starts)
        k = max(1, int(0.8 * len(res)))
        shape = float(np.sort(res)[:k].mean())
        col = float(np.abs(ca[sa][v] - cb[sb][v]).sum(-1).mean()) / 255.0
        cost = shape + 0.5 * col
        if cost < best[0]:
            best = (cost, int(round(dx)), int(dy))
    return best


def all_matches(pieces: list[Piece]) -> list[Match]:
    tps = [transpose_piece(p) for p in pieces]
    out = []
    n = len(pieces)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c, dx, dy = pair_match(pieces[i], pieces[j])
            if np.isfinite(c):
                out.append(Match(i, j, dx, dy, c, "lr"))
            c, dx, dy = pair_match(tps[i], tps[j])
            if np.isfinite(c):
                out.append(Match(i, j, dy, dx, c, "tb"))  # swap back from transposed coords
    return out


def _overlap(pieces, pos_a: dict, pos_b: dict) -> float:
    """Largest fraction of a piece's area that overlaps another piece across two groups."""
    worst = 0.0
    for i, (xi, yi) in pos_a.items():
        for j, (xj, yj) in pos_b.items():
            Pi, Pj = pieces[i], pieces[j]
            x0, y0 = max(xi, xj), max(yi, yj)
            x1, y1 = min(xi + Pi.width, xj + Pj.width), min(yi + Pi.height, yj + Pj.height)
            if x1 <= x0 or y1 <= y0:
                continue
            mi = Pi.mask[y0 - yi:y1 - yi, x0 - xi:x1 - xi]
            mj = Pj.mask[y0 - yj:y1 - yj, x0 - xj:x1 - xj]
            inter = (mi & mj).sum()
            worst = max(worst, inter / min(Pi.mask.sum(), Pj.mask.sum()))
    return worst


@dataclass
class TornResult:
    positions: dict[int, tuple[int, int]]
    groups: list[list[int]]
    accepted: list[Match]
    runtime_s: float


def assemble(pieces: list[Piece], max_overlap: float = 0.02) -> TornResult:
    t0 = time.perf_counter()
    matches = all_matches(pieces)
    # normalise each match by how it compares with the alternatives for the same edge
    by_edge: dict[tuple, list[float]] = {}
    for m in matches:
        by_edge.setdefault((m.a, m.direction, "out"), []).append(m.cost)
        by_edge.setdefault((m.b, m.direction, "in"), []).append(m.cost)

    def ratio(m):
        def second(key):
            cs = sorted(by_edge[key])
            return cs[1] if len(cs) > 1 else cs[0] * 2 + 1
        return m.cost / (0.5 * (second((m.a, m.direction, "out")) + second((m.b, m.direction, "in"))) + 1e-6)

    matches.sort(key=ratio)
    group = {i: i for i in range(len(pieces))}
    pos = {i: {i: (0, 0)} for i in range(len(pieces))}  # group id -> {piece: (x, y)}
    accepted = []
    for m in matches:
        ga, gb = group[m.a], group[m.b]
        if ga == gb:
            continue
        ax, ay = pos[ga][m.a]
        bx, by = pos[gb][m.b]
        # translate group b so that b sits at a + (dx, dy)
        tx, ty = ax + m.dx - bx, ay + m.dy - by
        moved = {k: (x + tx, y + ty) for k, (x, y) in pos[gb].items()}
        if _overlap(pieces, pos[ga], moved) > max_overlap:
            continue
        pos[ga].update(moved)
        for k in moved:
            group[k] = ga
        del pos[gb]
        accepted.append(m)
    big = max(pos.values(), key=len)
    x0 = min(x for x, _ in big.values())
    y0 = min(y for _, y in big.values())
    positions = {k: (x - x0, y - y0) for k, (x, y) in big.items()}
    return TornResult(positions, [sorted(g) for g in pos.values()], accepted, time.perf_counter() - t0)


def evaluate_torn(pieces: list[Piece], res: TornResult, truth: dict, tol: int = 4) -> dict:
    """Fraction of true adjacent pairs placed at the correct relative offset."""
    ids = [p.id for p in pieces]
    idx = {p: i for i, p in enumerate(ids)}
    off = truth["offsets"]
    grid = truth["grid"]
    pairs = []
    for r, row in enumerate(grid):
        for c, a in enumerate(row):
            if c + 1 < len(row):
                pairs.append((a, row[c + 1]))
            if r + 1 < len(grid):
                pairs.append((a, grid[r + 1][c]))
    ok = 0
    for a, b in pairs:
        ia, ib = idx[a], idx[b]
        if ia in res.positions and ib in res.positions:
            pr = np.subtract(res.positions[ib], res.positions[ia])
            tr = np.subtract(off[b], off[a])
            ok += int(np.abs(pr - tr).max() <= tol)
    return {"neighbour": ok / len(pairs), "placed": len(res.positions) / len(pieces)}


def render_torn(pieces: list[Piece], res: TornResult, background=(24, 26, 30, 255)):
    from PIL import Image

    W = max(x + pieces[i].width for i, (x, y) in res.positions.items())
    H = max(y + pieces[i].height for i, (x, y) in res.positions.items())
    canvas = Image.new("RGBA", (W, H), background)
    for i, (x, y) in res.positions.items():
        canvas.alpha_composite(Image.fromarray(pieces[i].image), (x, y))
    return canvas
