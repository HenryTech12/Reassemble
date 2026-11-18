"""Turn a Reconstruction back into pictures: assembled page, confidence heatmap."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from ..model import Piece, Reconstruction

GREEN, YELLOW, RED, GREY = (46, 204, 113), (241, 196, 15), (231, 76, 60), (140, 140, 140)


def confidence_colour(conf: float, blank: bool = False, gap: bool = False) -> tuple[int, int, int]:
    if gap:
        return RED
    if blank:
        return GREY
    if conf >= 0.6:
        return GREEN
    if conf >= 0.25:
        return YELLOW
    return RED


def _boundary_median(piece: Piece, side: str) -> float:
    m = piece.mask
    rows = m.any(1)
    if not rows.any():
        return 0.0 if side == "left" else piece.width - 1
    if side == "left":
        pos = np.argmax(m[rows], axis=1)
    else:
        pos = piece.width - 1 - np.argmax(m[rows][:, ::-1], axis=1)
    return float(np.median(pos))


def strip_layout(pieces: dict[str, Piece], order: list[str | None], gap_width: float | None = None) -> list[dict]:
    """x offsets for every piece (and gap placeholder) in a strip reconstruction.

    Ragged or frayed strips overlap their bounding boxes, so instead of butting boxes
    together we align the median right boundary of A with the median left boundary of B.
    """
    widths = [p.width for p in pieces.values()]
    gw = gap_width or (float(np.median(widths)) if widths else 20.0)
    out, x, prev = [], 0.0, None
    for pid in order:
        if pid is None:
            out.append({"id": None, "x": int(round(x)), "y": 0, "w": int(gw), "gap": True})
            x += gw
            prev = None
            continue
        p = pieces[pid]
        if prev is not None:
            x = x - (prev.width - 1 - _boundary_median(prev, "right")) - _boundary_median(p, "left")
        out.append({"id": pid, "x": int(round(x)), "y": 0, "w": p.width, "h": p.height})
        x += p.width
        prev = p
    return out


def render_strips(pieces: dict[str, Piece], rec: Reconstruction, background=(24, 26, 30, 255),
                  heatmap: bool = False, gap_width: float | None = None) -> Image.Image:
    layout = strip_layout(pieces, rec.order, gap_width)
    H = max((p.height for p in pieces.values()), default=10)
    W = max((l["x"] + l["w"] for l in layout), default=10)
    canvas = Image.new("RGBA", (W, H), background)
    for l in layout:
        if l.get("gap"):
            _hatch(canvas, l["x"], l["w"], H)
            continue
        canvas.alpha_composite(Image.fromarray(pieces[l["id"]].image), (l["x"], l["y"]))
    if heatmap:
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        xs = {l["id"]: l for l in layout if l["id"]}
        for j in rec.joins:
            if j["right"] not in xs:
                continue
            x = xs[j["right"]]["x"]
            col = confidence_colour(j["confidence"], j.get("blank", False), j.get("gap", False))
            d.rectangle([x - 4, 0, x + 3, H], fill=col + (110,))
        canvas = Image.alpha_composite(canvas, overlay)
    return canvas


def _hatch(canvas: Image.Image, x: int, w: int, h: int) -> None:
    d = ImageDraw.Draw(canvas)
    d.rectangle([x, 0, x + w - 1, h - 1], fill=(60, 20, 20, 255), outline=(231, 76, 60, 255), width=2)
    for y in range(-w, h, 18):
        d.line([(x, y), (x + w, y + w)], fill=(231, 76, 60, 255), width=2)


def render_grid(pieces: dict[str, Piece], grid: list[list[str | None]], background=(24, 26, 30, 255)) -> Image.Image:
    """Paste cross-cut pieces on a grid (cell size = median piece size)."""
    ws = [p.width for p in pieces.values()]
    hs = [p.height for p in pieces.values()]
    cw, ch = int(np.median(ws)), int(np.median(hs))
    rows, cols = len(grid), max(len(r) for r in grid) if grid else 0
    canvas = Image.new("RGBA", (cols * cw, rows * ch), background)
    for i, row in enumerate(grid):
        for j, pid in enumerate(row):
            if pid is not None and pid in pieces:
                canvas.alpha_composite(Image.fromarray(pieces[pid].image), (j * cw, i * ch))
    return canvas
