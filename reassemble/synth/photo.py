"""Simulated phone photo of strips laid on a dark capture mat.

Lets the capture pipeline be tested end-to-end without real paper: strips are scattered on
a dark textured mat with four ArUco corner markers, then the whole scene is photographed
through a random perspective, with uneven lighting, sensor noise and JPEG compression.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from ..model import Piece
from .shredders import rotate_rgba

MAT_W, MAT_H = 2600, 1500       # mat size in "mat pixels" (the rectified coordinate frame)
MARKER = 120                     # marker side
MARKER_IDS = (0, 1, 2, 3)        # TL, TR, BR, BL
MARGIN = 40


def aruco_dict():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def marker_corners_mat(scale: float = 1.0) -> dict[int, np.ndarray]:
    """Mat-frame coordinates of each marker's 4 corners (TL, TR, BR, BL order).

    ``scale`` is the resolution of the mat frame (1.0 = MAT_W x MAT_H pixels)."""
    s = MARKER
    tl = {0: (MARGIN, MARGIN), 1: (MAT_W - MARGIN - s, MARGIN),
          2: (MAT_W - MARGIN - s, MAT_H - MARGIN - s), 3: (MARGIN, MAT_H - MARGIN - s)}
    return {i: np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], np.float32) * scale
            for i, (x, y) in tl.items()}


def draw_mat(rng: np.random.Generator, markers: bool = True, scale: float = 1.0) -> np.ndarray:
    """Dark fabric-like mat with white-bordered ArUco markers."""
    W, H = int(MAT_W * scale), int(MAT_H * scale)
    base = rng.normal(38, 6, (H // 4, W // 4, 1)).astype(np.float32)
    base = cv2.resize(base, (W, H), interpolation=cv2.INTER_CUBIC)[..., None]
    weave = (np.sin(np.arange(W) / 3.0)[None, :] * np.sin(np.arange(H) / 3.0)[:, None]) * 4
    mat = base + weave[..., None] + rng.normal(0, 3, (H, W, 1))
    mat = np.repeat(mat, 3, axis=2) * np.array([0.95, 1.0, 1.1])
    mat = np.clip(mat, 0, 255).astype(np.uint8)
    if markers:
        d = aruco_dict()
        ms = int(MARKER * scale)
        for i, c in marker_corners_mat(scale).items():
            x, y = c[0].astype(int)
            pad = int(14 * scale)
            mat[y - pad:y + ms + pad, x - pad:x + ms + pad] = 255
            m = cv2.aruco.generateImageMarker(d, MARKER_IDS[i], ms)
            mat[y:y + ms, x:x + ms] = m[..., None]
    return mat


def compose_photo(pieces: list[Piece], seed: int = 0, max_rot: float = 1.2, out_w: int = 3400,
                  perspective: float = 0.06, markers: bool = True, lighting: bool = True,
                  noise: float = 3.0, jpeg: int = 92, upscale: float = 2.0) -> tuple[np.ndarray, dict]:
    """Lay pieces in a row on the mat (slightly rotated, unevenly spaced, some nearly
    touching) and 'photograph' it.  Returns (BGR photo, truth) where truth maps each piece
    id to its centre in mat coordinates (at mat-frame resolution ``upscale``).

    ``upscale`` sets the capture resolution relative to the 100-dpi synthetic pages.  A
    12-MP phone photo of a 60 cm mat is ~170 dpi and a flatbed scan 300 dpi, so 2x is a
    realistic default.  (At 1x, letter strokes are ~2 px wide and the 1-px boundary loss
    of any real capture destroys most of the seam evidence.)"""
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    u = upscale
    MW, MH = int(MAT_W * u), int(MAT_H * u)
    mat = draw_mat(rng, markers, u).astype(np.float32)
    order = list(pieces)
    prng.shuffle(order)
    usable_w = MW - 2 * (MARGIN + MARKER + 40) * u
    step = usable_w / max(1, len(order))
    centres = {}
    for k, p in enumerate(order):
        src = p.image
        if u != 1:
            src = cv2.resize(src, (int(round(src.shape[1] * u)), int(round(src.shape[0] * u))),
                             interpolation=cv2.INTER_CUBIC)
        img = rotate_rgba(src, prng.uniform(-max_rot, max_rot))
        h, w = img.shape[:2]
        scale = min(1.0, (MH - 2 * MARGIN * u - 60 * u) / h)
        if scale < 1:
            img = cv2.resize(img, (max(1, int(w * scale)), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        cx = (MARGIN + MARKER + 40) * u + step * (k + 0.5) + prng.uniform(-0.15, 0.15) * step
        cy = MH / 2 + prng.uniform(-25, 25) * u
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
        a = img[..., 3:].astype(np.float32) / 255.0
        region = mat[y0:y0 + h, x0:x0 + w]
        region[:] = region * (1 - a) + img[..., :3].astype(np.float32) * a
        centres[p.id] = [float(cx), float(cy)]
    # lighting: smooth gradient + vignette
    yy, xx = np.mgrid[0:MH, 0:MW].astype(np.float32)
    grad = 1.0 + 0.18 * (xx / MW - 0.5) * rng.choice([-1, 1]) + 0.1 * (yy / MH - 0.5)
    vig = 1.0 - 0.18 * (((xx - MW / 2) / MW) ** 2 + ((yy - MH / 2) / MH) ** 2) * 2
    if lighting:
        mat = mat * (grad * vig)[..., None]
    # perspective
    out_w = int(out_w * u)
    out_h = int(out_w * MAT_H / MAT_W) + 200
    src = np.float32([[0, 0], [MW, 0], [MW, MH], [0, MH]])
    jitter = rng.uniform(-perspective, perspective, (4, 2)) * [out_w, out_h]
    dst = np.float32([[100, 100], [out_w - 100, 100], [out_w - 100, out_h - 100], [100, out_h - 100]]) + jitter
    H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    photo = cv2.warpPerspective(np.clip(mat, 0, 255).astype(np.uint8), H, (out_w, out_h),
                                flags=cv2.INTER_LINEAR, borderValue=(25, 25, 28))
    photo = np.clip(photo.astype(np.float32) + rng.normal(0, noise, photo.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", photo[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, jpeg])
    photo_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    truth = {"kind": "photo", "centres_mat": centres, "homography": H.tolist(),
             "mat_size": [MW, MH], "upscale": u}
    return photo_bgr, truth
