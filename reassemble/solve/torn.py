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


def _straight(x: np.ndarray, has: np.ndarray, tol: float = 0.6) -> bool:
    """True if the boundary is (nearly) a straight line: the original sheet's edge.

    Measured on synthetic pages: a sheet edge, even rotated, has a median residual of
    0.2-0.4 px from a fitted line; tear segments are >= 0.8 px.
    """
    ys = np.flatnonzero(has)
    if len(ys) < 20:
        return True
    sl, ic = np.polyfit(ys, x[ys], 1)
    return float(np.median(np.abs(x[ys] - (ic + sl * ys)))) < tol


def pair_match(A: Piece, B: Piece, min_overlap: float = 0.4, fit_tilt: bool = False):
    """Best (cost, dx, dy) for B to the right of A.

    With ``fit_tilt`` the offset between the two boundaries is modelled as a *line*
    ``dx(y) = a + b*y`` instead of a constant: if B is rotated by a small angle relative
    to A, its boundary drifts linearly along the tear.  The slope ``b`` (≈ the relative
    rotation in radians) is returned as a 4th value.
    """
    xa, ha = boundary_profile(A, "right")
    xb, hb = boundary_profile(B, "left")
    ca = _inside_colour(A, xa, "right")
    cb = _inside_colour(B, xb, "left")
    # a side that is (almost) all page edge has nothing to match: either it lies on the
    # crop boundary (upright pieces) or it is a straight line (rotated pieces)
    if ha.mean() < 0.3 or hb.mean() < 0.3 or _straight(xa, ha) or _straight(xb, hb):
        return (np.inf, 0, 0, 0.0) if fit_tilt else (np.inf, 0, 0)
    HA, HB = A.height, B.height
    need = min_overlap * min(ha.sum(), hb.sum())
    best = (np.inf, 0, 0, 0.0)
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
        k = max(1, int(0.8 * len(diff)))
        if fit_tilt:
            ys = np.arange(ya0, ya1)[v].astype(np.float64)
            slope, icpt = np.polyfit(ys, diff, 1)
            res = np.abs(diff - (icpt + slope * ys))
            keep = np.argsort(res)[:k]  # refit without the worst rows (tear corners)
            slope, icpt = np.polyfit(ys[keep], diff[keep], 1)
            res = np.abs(diff - (icpt + slope * ys))
            dx = icpt + slope * ys.mean()
        else:
            slope = 0.0
            dx = np.median(diff)
            res = np.abs(diff - dx)
        # trimmed mean: ignore the 20 % worst rows (corners where another tear starts)
        shape = float(np.sort(res)[:k].mean())
        col = float(np.abs(ca[sa][v] - cb[sb][v]).sum(-1).mean()) / 255.0
        cost = shape + 0.5 * col
        if cost < best[0]:
            best = (cost, int(round(dx)), int(dy), float(slope))
    return best if fit_tilt else best[:3]


def all_matches(pieces: list[Piece], fit_tilt: bool = False) -> list[Match]:
    tps = [transpose_piece(p) for p in pieces]
    out = []
    n = len(pieces)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c, dx, dy = pair_match(pieces[i], pieces[j], fit_tilt=fit_tilt)[:3]
            if np.isfinite(c):
                out.append(Match(i, j, dx, dy, c, "lr"))
            c, dx, dy = pair_match(tps[i], tps[j], fit_tilt=fit_tilt)[:3]
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


def assemble(pieces: list[Piece], max_overlap: float = 0.02, fit_tilt: bool = False) -> TornResult:
    t0 = time.perf_counter()
    matches = all_matches(pieces, fit_tilt)
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


def relative_angles(pieces: list[Piece]) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    """Absolute rotation (degrees) of every piece relative to a reference piece.

    Each candidate match gives the *relative* rotation of the two pieces (the tilt of the
    fitted boundary line).  We take the most confident matches Kruskal-style to form a
    spanning tree and add up angles along it from the root.
    """
    tps = [transpose_piece(p) for p in pieces]
    n = len(pieces)
    edges = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c, _, _, sl = pair_match(pieces[i], pieces[j], fit_tilt=True)
            if np.isfinite(c):
                edges.append((c, i, j, -np.degrees(sl)))   # LR: B's angle = A's - slope
            c, _, _, sl = pair_match(tps[i], tps[j], fit_tilt=True)
            if np.isfinite(c):
                edges.append((c, i, j, np.degrees(sl)))    # TB: transposition flips the sign
    # ratio-normalise per piece pair direction like `assemble` does, then Kruskal
    best_out: dict[int, list[float]] = {}
    for c, i, j, _ in edges:
        best_out.setdefault(i, []).append(c)
        best_out.setdefault(j, []).append(c)
    second = {k: sorted(v)[1] if len(v) > 1 else v[0] * 2 + 1 for k, v in best_out.items()}
    edges.sort(key=lambda e: e[0] / (0.5 * (second[e[1]] + second[e[2]]) + 1e-6))
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = []
    for c, i, j, ang in edges:
        if find(i) != find(j):
            parent[find(i)] = find(j)
            tree.append((i, j, ang, c))
    # Least squares over every confident relation, not just the tree: errors average out
    # instead of accumulating along a chain.  Each edge says angle[j] - angle[i] = ang,
    # weighted by how much better it is than the alternatives.
    conf_edges = [(i, j, ang, 1.0 / (0.05 + c / (0.5 * (second[i] + second[j]) + 1e-6)) ** 2)
                  for c, i, j, ang in edges if c / (0.5 * (second[i] + second[j]) + 1e-6) < 0.9]
    tree_set = {(i, j) for i, j, _, _ in tree}
    conf_edges += [(i, j, ang, 4.0) for c, i, j, ang in edges if (i, j) in tree_set]
    A = np.zeros((len(conf_edges) + 1, n))
    b = np.zeros(len(conf_edges) + 1)
    for r, (i, j, ang, w) in enumerate(conf_edges):
        sw = np.sqrt(w)
        A[r, j], A[r, i], b[r] = sw, -sw, sw * ang
    A[-1, :] = 1.0  # gauge: mean angle 0
    angle = np.linalg.lstsq(A, b, rcond=None)[0]
    return angle - np.median(angle), tree


def text_angle(piece: Piece, max_angle: float = 5.0, step: float = 0.05) -> tuple[float, float]:
    """Rotation of a piece estimated from its text lines (classic projection-profile deskew).

    Text lines are horizontal on the original page.  Rotating the ink pixels by the right
    angle makes the horizontal projection profile *peaky* (lines and gaps separate
    cleanly); a wrong angle smears it.  We pick the angle maximising the profile's sum of
    squares.  Returns (angle in degrees, strength); strength ~1 means no clear text lines.
    """
    rgb = piece.rgb.astype(np.float32).mean(-1)
    m = piece.mask
    if m.sum() < 100:
        return 0.0, 1.0
    paper = np.median(rgb[m])
    ys, xs = np.nonzero(m & (rgb < paper - 60))
    if len(ys) < 200:
        return 0.0, 1.0
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    scores = []
    angles = np.arange(-max_angle, max_angle + 1e-9, step)
    for a in angles:
        t = np.radians(a)
        yr = ys * np.cos(t) + xs * np.sin(t)
        h = np.bincount(np.round(yr - yr.min()).astype(int)).astype(np.float64)
        scores.append(float((h ** 2).sum()))
    scores = np.array(scores)
    k = int(np.argmax(scores))
    return float(angles[k]), float(scores[k] / (np.median(scores) + 1e-9))


def assemble_rotated(pieces: list[Piece], max_angle: float = 5.0) -> tuple[TornResult, list[Piece], np.ndarray]:
    """Assembly for pieces that were photographed slightly rotated.

    Each piece is deskewed on its own from its text lines (``text_angle``), then the exact
    upright assembler runs.  Pieces without clear text lines (photos, blank paper) fall back
    to the median angle of the others, which is right when the whole pile was shot at one
    skew and harmless otherwise.

    (Chaining relative angles from tear-edge tilts was tried first: errors accumulate
    along the chain and wrong matches pollute the estimate.  See ``relative_angles``.)
    """
    est = [text_angle(p, max_angle) for p in pieces]
    strong = [a for a, s in est if s > 1.1]
    fallback = float(np.median(strong)) if strong else 0.0
    angle = np.array([a if s > 1.1 else fallback for a, s in est])
    fixed = deskew_pieces(pieces, angle)
    return assemble(fixed), fixed, angle


def deskew_pieces(pieces: list[Piece], angles) -> list[Piece]:
    """Rotate each piece by minus its estimated angle (degrees) and re-crop."""
    from ..synth.shredders import rotate_rgba

    fixed = []
    for p, a in zip(pieces, angles):
        if abs(a) < 0.05:
            fixed.append(p)
            continue
        img = rotate_rgba(p.image, -a)
        img[..., 3] = np.where(img[..., 3] > 127, 255, 0).astype(np.uint8)
        ys, xs = np.nonzero(img[..., 3])
        img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()
        img[img[..., 3] == 0, :3] = 0
        fixed.append(Piece(id=p.id, image=img, meta=dict(p.meta, rotation_deg=float(a))))
    return fixed


def evaluate_torn(pieces: list[Piece], res: TornResult, truth: dict, tol: int = 4) -> dict:
    """Fraction of true adjacent pairs placed at the correct relative offset.

    If the truth has piece centroids (``centres``), centroids are compared instead of crop
    origins, which stays meaningful when pieces were rotated before assembly (tolerance
    doubled to absorb the small residual rotation of the whole page).
    """
    if "centres" in truth:
        return _evaluate_centres(pieces, res, truth, 2 * tol)
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


def render_torn(pieces: list[Piece], res: TornResult, background=(24, 26, 30, 255),
                rotations: dict[str, float] | None = None):
    from PIL import Image

    if rotations:
        pieces = deskew_pieces(pieces, [rotations.get(p.id, 0.0) for p in pieces])

    W = max(x + pieces[i].width for i, (x, y) in res.positions.items())
    H = max(y + pieces[i].height for i, (x, y) in res.positions.items())
    canvas = Image.new("RGBA", (W, H), background)
    for i, (x, y) in res.positions.items():
        canvas.alpha_composite(Image.fromarray(pieces[i].image), (x, y))
    return canvas


def _evaluate_centres(pieces: list[Piece], res: TornResult, truth: dict, tol: float) -> dict:
    ids = [p.id for p in pieces]
    grid = truth["grid"]
    pairs = []
    for r, row in enumerate(grid):
        for c, a in enumerate(row):
            if c + 1 < len(row):
                pairs.append((a, row[c + 1]))
            if r + 1 < len(grid):
                pairs.append((a, grid[r + 1][c]))
    centre = {}
    for i, (x, y) in res.positions.items():
        ys, xs = np.nonzero(pieces[i].mask)
        centre[ids[i]] = np.array([x + xs.mean(), y + ys.mean()])
    ok = 0
    for a, b in pairs:
        if a in centre and b in centre:
            pr = centre[b] - centre[a]
            tr = np.subtract(truth["centres"][b], truth["centres"][a])
            ok += int(np.abs(pr - tr).max() <= tol)
    return {"neighbour": ok / len(pairs), "placed": len(res.positions) / len(pieces)}
