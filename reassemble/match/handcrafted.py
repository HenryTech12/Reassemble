"""Handcrafted pairwise edge compatibility for strips.

Notation: ``D[a, b]`` is the cost of putting piece ``b`` immediately to the right of
piece ``a`` (lower = better).  It combines three cues:

1. **Prediction-based pixel cost** (Pomeranz et al. 2011).  Images are locally smooth, so
   the column just across a true seam should be predictable by linear extrapolation from
   the last two columns: ``pred = A[-1] + (A[-1] - A[-2])``.  We compare that prediction
   with B's first column, and symmetrically predict A's last column from B.  This beats a
   plain "difference of boundary columns" because it accounts for gradients (e.g. a
   letter stroke getting thinner as it approaches the cut).
2. **Text-line alignment.**  The ink projection profile near A's right edge should
   correlate with the one near B's left edge.  This is weighted by how regular the text
   lines are, so it fades out for handwriting and photos.
3. **Paper / ink colour similarity** as a weak prior (helps most in mixed bins).

Each cue is divided by its median over all pairs so the weights are unit-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..features.edges import EdgeFeatures

BIG = 1e6


@dataclass
class ScorerConfig:
    shift: int = 2            # vertical misalignment searched (+/- px)
    grad_weight: float = 1.0  # 1 = full linear extrapolation, 0 = plain boundary diff
    power: float = 1.0        # per-row error is |e|_1 ** power
    dead_zone: float = 2.0    # ignore per-row errors below dead_zone * estimated noise
    w_pixel: float = 1.0
    w_baseline: float = 0.15
    w_color: float = 0.05


def _robust(err: np.ndarray, power: float) -> np.ndarray:
    """Per-row penalty ``err ** power``.

    We experimented with saturating penalties (cap the cost of any single row) but they
    throw away the most useful signal: a stroke meeting blank paper is a *large* error and
    should cost more than a slight grey-level mismatch.  Plain L1 (power=1) was the most
    robust choice across clean and noisy scans.
    """
    return err if power == 1 else err ** power


def _noise(col: np.ndarray, valid: np.ndarray) -> float:
    """Estimate the per-row noise (L1 over channels) of an edge column.

    Row-to-row differences of a mostly-blank column are dominated by noise; the median
    absolute difference is insensitive to the few rows where ink starts or stops.
    """
    d = np.abs(np.diff(col, axis=0)).sum(-1)
    v = valid[1:] & valid[:-1]
    if v.sum() < 10:
        return 0.0
    return float(np.median(d[v])) / np.sqrt(2)


def pixel_cost_matrix(feats_a: list[EdgeFeatures], feats_b: list[EdgeFeatures] | None = None,
                      cfg: ScorerConfig | None = None, return_shift: bool = False):
    """Prediction-based seam cost between right edges of ``feats_a`` and left edges of
    ``feats_b`` (defaults to the same list).  Returns an (Na, Nb) matrix."""
    cfg = cfg or ScorerConfig()
    feats_b = feats_a if feats_b is None else feats_b
    H = min(min(f.height for f in feats_a), min(f.height for f in feats_b))
    g = cfg.grad_weight
    # prediction of the column across the seam, from each side
    RA = np.stack([f.right[:H, 0] + g * (f.right[:H, 0] - f.right[:H, 1]) for f in feats_a])
    RA0 = np.stack([f.right[:H, 0] for f in feats_a])
    RV = np.stack([f.right_valid[:H] for f in feats_a])
    LB = np.stack([f.left[:H, 0] + g * (f.left[:H, 0] - f.left[:H, 1]) for f in feats_b])
    LB0 = np.stack([f.left[:H, 0] for f in feats_b])
    LV = np.stack([f.left_valid[:H] for f in feats_b])
    na, nb = len(feats_a), len(feats_b)
    # per-piece noise level of the edge signal (robust: MAD of row-to-row differences)
    na_sig = np.array([_noise(f.right[:H, 0], f.right_valid[:H]) for f in feats_a])
    nb_sig = np.array([_noise(f.left[:H, 0], f.left_valid[:H]) for f in feats_b])
    tau = cfg.dead_zone * (na_sig[:, None] + nb_sig[None, :])  # (na, nb)
    best = np.full((na, nb), np.inf)
    best_dy = np.zeros((na, nb), int)
    s = cfg.shift
    for dy in range(-s, s + 1):
        # shift B vertically by dy (B row y aligns with A row y+dy)
        sl_a = slice(max(0, dy), H + min(0, dy))
        sl_b = slice(max(0, -dy), H + min(0, -dy))
        ra, ra0, rv = RA[:, sl_a], RA0[:, sl_a], RV[:, sl_a]
        lb, lb0, lv = LB[:, sl_b], LB0[:, sl_b], LV[:, sl_b]
        cost = np.empty((na, nb))
        for i in range(na):
            e1 = np.abs(ra[i][None] - lb0).sum(-1)      # predict B from A
            e2 = np.abs(lb - ra0[i][None]).sum(-1)      # predict A from B
            valid = rv[i][None] & lv
            t = tau[i][:, None]
            e1 = np.maximum(e1 - t, 0)
            e2 = np.maximum(e2 - t, 0)
            r = 0.5 * (_robust(e1, cfg.power) + _robust(e2, cfg.power))
            cost[i] = (r * valid).sum(1) / np.maximum(valid.sum(1), 1)
        better = cost < best
        best[better] = cost[better]
        best_dy[better] = dy
    if return_shift:
        return best, best_dy
    return best


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-9 else 0.0


def baseline_cost_matrix(feats: list[EdgeFeatures]) -> np.ndarray:
    """1 - correlation between A's right-third ink profile and B's left-third profile."""
    H = min(f.height for f in feats)
    pr = [gaussian_filter1d(f.profile_right[:H], 1.5) for f in feats]
    pl = [gaussian_filter1d(f.profile_left[:H], 1.5) for f in feats]
    n = len(feats)
    out = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                out[i, j] = 1.0 - _ncc(pr[i], pl[j])
    return out


def color_cost_matrix(feats: list[EdgeFeatures]) -> np.ndarray:
    paper = np.stack([f.paper for f in feats])
    ink = np.stack([f.ink for f in feats])
    dp = np.linalg.norm(paper[:, None] - paper[None], axis=-1)
    di = np.linalg.norm(ink[:, None] - ink[None], axis=-1)
    return dp + 0.3 * di


def _unit(m: np.ndarray) -> np.ndarray:
    off = m[~np.eye(len(m), dtype=bool)] if m.shape[0] == m.shape[1] else m.ravel()
    med = np.median(off[np.isfinite(off)]) if off.size else 1.0
    return m / (med + 1e-9)


def handcrafted_cost(feats: list[EdgeFeatures], cfg: ScorerConfig | None = None,
                     back_feats: list[EdgeFeatures] | None = None) -> dict[str, np.ndarray]:
    """Full combined cost matrix plus its components.

    If ``back_feats`` is given (double-sided pages) the back-side seam cost is added.  The
    back is stored in reading orientation, so "B right of A on the front" means "A.back
    right of B.back": we use the *transpose* of the back pixel-cost matrix.
    """
    cfg = cfg or ScorerConfig()
    n = len(feats)
    pix = pixel_cost_matrix(feats, cfg=cfg)
    if back_feats is not None:
        pix = 0.5 * (pix + pixel_cost_matrix(back_feats, cfg=cfg).T)
    base = baseline_cost_matrix(feats)
    reg = np.array([f.line_regularity for f in feats])
    base_w = np.sqrt(np.outer(reg, reg))  # only trust baselines if both sides have regular text
    col = color_cost_matrix(feats)
    total = (cfg.w_pixel * _unit(pix) + cfg.w_baseline * base_w * _unit(base)
             + cfg.w_color * _unit(col))
    np.fill_diagonal(total, BIG)
    np.fill_diagonal(pix, BIG)
    return {"total": total, "pixel": pix, "baseline": base, "color": col}


def edge_inkiness(feats: list[EdgeFeatures]) -> tuple[np.ndarray, np.ndarray]:
    """How much ink touches each piece's left / right edge (0 = blank margin).

    The first strip of a page normally has a blank left edge (the page margin), so this is
    used as the cost of starting / ending the sequence with that piece.
    """
    def ink(edge, valid, paper):
        g = edge[:, 0].mean(-1)
        dark = (g < paper.mean() - 60) & valid
        # any ink at all on an edge makes it an unlikely page border
        return float(np.clip(dark.mean() / 0.005, 0, 1))
    left = np.array([ink(f.left, f.left_valid, f.paper) for f in feats])
    right = np.array([ink(f.right, f.right_valid, f.paper) for f in feats])
    return left, right


def normalize_costs(D: np.ndarray) -> np.ndarray:
    """Divide each cost by the second-best alternative for both pieces involved.

    Raw costs are not comparable across pieces (a strip full of text has larger seam
    costs than a nearly blank one).  What matters is *how much better* a candidate is
    than the alternatives — the same intuition as Lowe's ratio test or Paikin & Tal's
    confidence.  ``N[a,b] < 1`` means b is clearly a's best match.
    """
    n = len(D)
    Dm = D.copy()
    np.fill_diagonal(Dm, np.inf)
    srt_r = np.sort(Dm, 1)
    srt_c = np.sort(Dm, 0)
    second_r = srt_r[:, 1] if n > 2 else srt_r[:, 0]
    second_c = srt_c[1, :] if n > 2 else srt_c[0, :]
    # A floor keeps blank-vs-blank comparisons (0 / ~0) from producing random ratios.
    off = D[~np.eye(n, dtype=bool)]
    floor = 0.05 * float(np.median(off[off < BIG])) + 1e-9 if off.size else 1e-9
    N = (D + floor) / (0.5 * (second_r[:, None] + second_c[None, :]) + floor)
    np.fill_diagonal(N, BIG)
    return N
