"""Per-piece features used by the matchers.

The central idea: for every row ``y`` of a strip we find where the paper actually ends
(from the alpha mask) and read the ``k`` pixels just inside that boundary.  Reading
relative to the *mask boundary* rather than a fixed image column is what makes frayed,
rotated or torn edges work: the "edge column" bends with the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..model import Piece


@dataclass
class EdgeFeatures:
    """Everything the matchers need from one piece.

    ``right``/``left`` have shape (H, K, 3): K columns read inward from the boundary
    (index 0 = outermost usable column).  ``*_valid`` marks rows that contain paper.
    """

    id: str
    height: int
    width: int
    right: np.ndarray
    left: np.ndarray
    right_valid: np.ndarray
    left_valid: np.ndarray
    right_pos: np.ndarray         # x of the outer boundary per row (for shape matching)
    left_pos: np.ndarray
    paper: np.ndarray             # mean paper colour (3,)
    ink: np.ndarray               # mean ink colour (3,) (= paper if no ink)
    ink_fraction: float
    profile: np.ndarray           # horizontal projection profile of ink (H,)
    profile_right: np.ndarray     # profile of the right third (near the right seam)
    profile_left: np.ndarray
    baselines: np.ndarray         # y positions of detected text baselines
    line_spacing: float           # median baseline distance (0 if no regular lines)
    line_regularity: float        # 0..1, how regular the text lines are
    x_height: float
    extra: dict = field(default_factory=dict)


def _boundary(mask: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-row x of the outermost paper pixel on ``side`` and a validity flag."""
    h, w = mask.shape
    has = mask.any(1)
    if side == "right":
        pos = w - 1 - np.argmax(mask[:, ::-1], axis=1)
    else:
        pos = np.argmax(mask, axis=1)
    return pos, has


def read_edge(rgb: np.ndarray, mask: np.ndarray, side: str, k: int = 3, margin: int = 0,
              smooth: float = 1.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read ``k`` columns inward from the paper boundary on ``side``.

    ``margin`` skips the outermost pixel(s), which are usually half-transparent from
    anti-aliasing or rotation.  A light vertical Gaussian (``smooth``) suppresses scan
    noise, which matters a lot because the prediction step below takes differences.
    """
    h, w = mask.shape
    pos, valid = _boundary(mask, side)
    step = -1 if side == "right" else 1
    rows = np.arange(h)
    cols = np.stack([np.clip(pos + step * (margin + i), 0, w - 1) for i in range(k)], 1)
    vals = rgb[rows[:, None], cols].astype(np.float32)  # (H, k, 3)
    inside = mask[rows[:, None], cols].all(1)
    valid = valid & inside
    if smooth > 0:
        vals = gaussian_filter1d(vals, smooth, axis=0)
    return vals, valid, pos


def sharpen(rgb: np.ndarray, mask: np.ndarray, amount: float = 3.5, sigma: float = 1.0) -> np.ndarray:
    """Unsharp masking restricted to the paper (normalised convolution).

    Scanner/camera blur smears thin strokes so the edge pattern of a true neighbour and
    of a random strip from the same text line look alike.  Boosting high frequencies
    (``x + amount * (x - blur(x))``) partially undoes that.  In our benchmark this single
    step raised noisy-text neighbour accuracy from ~63 % to ~93 %.  The blur used for the
    mask ignores non-paper pixels so the boundary doesn't get a dark halo.
    """
    w = mask.astype(np.float32)
    num = cv2.GaussianBlur(rgb * w[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)[..., None]
    low = num / np.maximum(den, 1e-3)
    return rgb + amount * (rgb - low)


def ink_mask(rgb: np.ndarray, mask: np.ndarray, paper: np.ndarray) -> np.ndarray:
    """Pixels noticeably darker than the paper are 'ink'."""
    gray = rgb.astype(np.float32).mean(-1)
    return mask & (gray < paper.mean() - 60)


def text_lines(profile: np.ndarray, min_gap: int = 8) -> tuple[np.ndarray, float, float]:
    """Find text baselines from a horizontal ink projection profile.

    A text line shows up as a band of rows with ink; its baseline is the bottom of the
    band.  Regular spacing of those bands is the signature of printed text — handwriting
    and photos give irregular or no bands, which we report via ``regularity`` so the
    baseline cue can be down-weighted (Phase 4 requirement).
    """
    on = profile > max(profile.max() * 0.15, 1e-6)
    bases = []
    y, H = 0, len(on)
    while y < H:
        if on[y]:
            s = y
            while y < H and on[y]:
                y += 1
            if y - s >= 3:
                bases.append(y - 1)
        y += 1
    bases = np.array(bases)
    if len(bases) < 3:
        return bases, 0.0, 0.0
    d = np.diff(bases)
    d = d[d >= min_gap]
    if len(d) == 0:
        return bases, 0.0, 0.0
    spacing = float(np.median(d))
    regular = float(np.mean(np.abs(d - spacing) < 0.15 * spacing))
    return bases, spacing, regular


def sharpness(pieces: list[Piece]) -> float:
    """Document-level sharpness in ~[0, 1]: strongest intensity step / ink contrast.

    Crisp scans step from paper to ink within one pixel (ratio ~0.9); a blurry photo
    spreads the step over several pixels (ratio ~0.5).  We use it to decide how much
    deblurring (unsharp masking) the matcher needs: sharpening a crisp scan only adds
    aliasing and noise.
    """
    vals = []
    for p in pieces:
        g = p.rgb.astype(np.float32).mean(-1)
        m = p.image[..., 3] > 200
        if m.sum() < 50:
            continue
        gx = np.abs(np.diff(g, axis=1))[m[:, 1:] & m[:, :-1]]
        gy = np.abs(np.diff(g, axis=0))[m[1:] & m[:-1]]
        paper, ink = np.median(g[m]), np.percentile(g[m], 0.5)
        if paper - ink > 60 and gx.size + gy.size:
            vals.append(np.percentile(np.concatenate([gx, gy]), 99.5) / (paper - ink))
    return float(np.median(vals)) if vals else 1.0


def sharpen_amount(sharp: float) -> float:
    """Map sharpness to an unsharp-mask amount (0 for crisp scans, 3.5 for blurry ones)."""
    return float(np.clip((0.85 - sharp) / 0.3, 0.0, 1.0) * 3.5)


def extract(piece: Piece, k: int = 3, sharpen_by: float = 0.0) -> EdgeFeatures:
    rgb = piece.rgb
    m = piece.mask
    h, w = m.shape
    paper_px = rgb[m]
    if len(paper_px) == 0:
        paper = np.array([255.0, 255.0, 255.0], np.float32)
    else:
        paper = np.median(paper_px, 0).astype(np.float32)
    # White-balance the edges by the paper colour: a strip scanned a little darker or
    # warmer than its neighbour should still match it.
    balanced = rgb.astype(np.float32) * (230.0 / np.maximum(paper, 1.0))
    if sharpen_by > 0:
        balanced = sharpen(balanced, m, amount=sharpen_by)
    right, rv, rpos = read_edge(balanced, m, "right", k)
    left, lv, lpos = read_edge(balanced, m, "left", k)
    ink = ink_mask(rgb, m, paper)
    ink_col = rgb[ink].mean(0).astype(np.float32) if ink.any() else paper.copy()
    prof = ink.sum(1).astype(np.float32)
    third = max(1, w // 3)
    bases, spacing, regular = text_lines(prof)
    # x-height proxy: median height of the ink bands
    on = prof > max(prof.max() * 0.15, 1e-6)
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0])))[::2] if on.any() else []
    xh = float(np.median(runs)) if len(runs) else 0.0
    ink_cols = np.flatnonzero(ink.any(0))
    margins = (int(ink_cols[0]), int(w - 1 - ink_cols[-1])) if len(ink_cols) else (w, w)
    return EdgeFeatures(
        id=piece.id, height=h, width=w, right=right, left=left, right_valid=rv, left_valid=lv,
        right_pos=rpos, left_pos=lpos, paper=paper, ink=ink_col,
        ink_fraction=float(ink.sum() / max(1, m.sum())), profile=prof,
        profile_right=ink[:, w - third:].sum(1).astype(np.float32),
        profile_left=ink[:, :third].sum(1).astype(np.float32),
        baselines=bases, line_spacing=spacing, line_regularity=regular, x_height=xh,
        extra={"blank_left": margins[0], "blank_right": margins[1]},
    )


def extract_back(piece: Piece, k: int = 3, sharpen_by: float = 0.0) -> EdgeFeatures | None:
    if piece.back is None:
        return None
    return extract(Piece(id=piece.id, image=piece.back), k, sharpen_by)
