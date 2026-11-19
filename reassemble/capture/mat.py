"""Printable capture mat: dark A4 landscape page with four ArUco corner markers.

Lay the strips upright inside the frame, side by side without overlapping, and photograph
the whole mat from above.  The markers let ``reassemble capture`` undo the perspective
exactly; the dark field gives the paper strong contrast.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..synth.photo import MARKER_IDS, MAT_H, MAT_W, aruco_dict, marker_corners_mat


def render_mat(scale: float = 1.0) -> Image.Image:
    W, H = int(MAT_W * scale), int(MAT_H * scale)
    img = np.full((H, W, 3), 45, np.uint8)
    d = aruco_dict()
    ms = int(120 * scale)
    for i, c in marker_corners_mat(scale).items():
        x, y = c[0].astype(int)
        pad = int(14 * scale)
        img[y - pad:y + ms + pad, x - pad:x + ms + pad] = 255
        img[y:y + ms, x:x + ms] = cv2.aruco.generateImageMarker(d, MARKER_IDS[i], ms)[..., None]
    pil = Image.fromarray(img)
    dr = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(26 * scale))
    except OSError:
        font = ImageFont.load_default()
    msg = ("REASSEMBLE CAPTURE MAT  -  lay strips upright inside the markers, side by side, not touching; "
           "photograph from directly above with all 4 markers visible.")
    dr.text((int(200 * scale), int(MAT_H * scale - 40 * scale)), msg, fill=(160, 160, 160), font=font)
    return pil


def save_mat_pdf(path: str | Path = "assets/capture_mat.pdf") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = render_mat(1.0)
    # 2600 px across a 297 mm A4-landscape width ~ 222 dpi
    img.save(path, "PDF", resolution=222.0)
    return path
