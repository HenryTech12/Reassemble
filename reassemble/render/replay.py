"""Replay animation: pieces fly from a shuffled pile into their solved positions.

Used for the shareable GIF (the web app animates the same thing live in the browser).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image

from ..model import Piece, Reconstruction
from .compose import strip_layout


def _ease(t: float) -> float:
    return 1 - (1 - t) ** 3


def replay_frames(pieces: dict[str, Piece], rec: Reconstruction, scale: float = 0.35,
                  frames_per_piece: int = 3, hold: int = 12, seed: int = 0) -> list[Image.Image]:
    layout = [l for l in strip_layout(pieces, rec.order) if l["id"]]
    if not layout:
        return []
    H = max(p.height for p in pieces.values())
    W = max(l["x"] + l["w"] for l in layout)
    cw, ch = int((W + 40) * scale), int((H * 1.6 + 40) * scale)
    rng = random.Random(seed)
    # pile: random spots in the lower part of the canvas
    start = {l["id"]: (rng.uniform(0, W - 40), H * 0.55 + rng.uniform(0, H * 0.05)) for l in layout}
    small = {pid: Image.fromarray(p.image).resize((max(1, int(p.width * scale)), max(1, int(p.height * scale))))
             for pid, p in pieces.items()}
    rot = {pid: rng.uniform(-25, 25) for pid in pieces}
    frames = []
    n = len(layout)
    total = n * frames_per_piece + 8
    for f in range(total + hold):
        canvas = Image.new("RGB", (cw, ch), (14, 17, 22))
        for k, l in enumerate(layout):
            t0 = k * frames_per_piece
            t = np.clip((f - t0) / 8.0, 0, 1)
            e = _ease(float(t))
            sx, sy = start[l["id"]]
            x = (sx + (l["x"] - sx) * e + 20) * scale
            y = (sy + (0 - sy) * e + 20) * scale
            im = small[l["id"]]
            ang = rot[l["id"]] * (1 - e)
            if abs(ang) > 0.5:
                im = im.rotate(ang, expand=True)
            canvas.paste(im, (int(x), int(y)), im)
        frames.append(canvas)
    return frames


def save_replay_gif(pieces: dict[str, Piece], rec: Reconstruction, path: str | Path, **kw) -> Path:
    frames = replay_frames(pieces, rec, **kw)
    path = Path(path)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=60, loop=0, optimize=True)
    return path
