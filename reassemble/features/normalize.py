"""Bring every piece into a canonical pose before feature extraction.

A photographed strip is never perfectly vertical.  Even 0.4 degrees of rotation shifts the
top of an 1100-px strip by ~8 px relative to its bottom, which would wreck any row-by-row
comparison across a seam.  So we estimate the dominant axis of the paper mask and rotate
the piece upright, then crop to the paper's bounding box.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..model import Piece
from ..synth.shredders import rotate_rgba


def mask_angle(mask: np.ndarray) -> float:
    """Angle (degrees) by which the long axis of the mask deviates from vertical.

    Uses PCA on the coordinates of paper pixels: the first principal component is the
    strip's long axis.  PCA is robust to frayed edges because those are a tiny fraction of
    the pixels.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return 0.0
    pts = np.stack([xs - xs.mean(), ys - ys.mean()], 1).astype(np.float64)
    cov = pts.T @ pts / len(pts)
    evals, evecs = np.linalg.eigh(cov)
    vx, vy = evecs[:, np.argmax(evals)]
    if vy < 0:
        vx, vy = -vx, -vy
    return float(np.degrees(np.arctan2(vx, vy)))


def crop_to_mask(rgba: np.ndarray, thresh: int = 127) -> np.ndarray:
    m = rgba[..., 3] > thresh
    if not m.any():
        return rgba
    ys, xs = np.nonzero(m)
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def deskew(rgba: np.ndarray, max_angle: float = 10.0, min_angle: float = 0.05) -> tuple[np.ndarray, float]:
    """Rotate a piece so its long axis is vertical; returns (image, angle applied).

    Only elongated pieces (strips) have a meaningful long axis, so near-square pieces
    (cross-cut, torn) are left untouched.
    """
    m = rgba[..., 3] > 127
    h, w = m.shape
    if h < 2.5 * w and w < 2.5 * h:
        return crop_to_mask(rgba), 0.0
    ang = mask_angle(m) if h >= w else mask_angle(m.T)
    if abs(ang) < min_angle or abs(ang) > max_angle:
        return crop_to_mask(rgba), 0.0
    # rotate_rgba rotates counter-clockwise for positive angles in image coords
    out = rotate_rgba(rgba, -ang)
    out[..., 3] = np.where(out[..., 3] > 127, 255, 0).astype(np.uint8)
    return crop_to_mask(out), ang


def normalize_piece(piece: Piece) -> Piece:
    """Deskew + crop front (and back) of a piece; returns a new Piece."""
    img, ang = deskew(piece.image)
    back = None
    if piece.back is not None:
        back, _ = deskew(piece.back)
    meta = dict(piece.meta, deskew_deg=ang)
    return Piece(id=piece.id, image=img, back=back, meta=meta)


def normalize_heights(pieces: list[Piece]) -> list[Piece]:
    """Pad pieces at the bottom so all have the same height (transparent padding)."""
    H = max(p.height for p in pieces)
    out = []
    for p in pieces:
        img = p.image
        if p.height < H:
            img = cv2.copyMakeBorder(img, 0, H - p.height, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
        back = p.back
        if back is not None and back.shape[0] < H:
            back = cv2.copyMakeBorder(back, 0, H - back.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
        out.append(Piece(id=p.id, image=img, back=back, meta=p.meta))
    return out
