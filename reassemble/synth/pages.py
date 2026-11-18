"""Render realistic synthetic test pages with PIL.

Each renderer returns an RGB ``PIL.Image`` plus a small ``style`` dict describing the
paper tone, ink colour and font.  Variation in those styles is what later lets the
clustering stage (Phase 2) tell documents in a mixed bin apart, so we randomise them on
purpose.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import text as T

FONT_DIR = Path(__file__).parent / "fonts"

# Candidate printed fonts (first existing path wins per family).
PRINT_FONTS = {
    "liberation-serif": ["/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"],
    "liberation-sans": ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"],
    "liberation-mono": ["/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"],
    "dejavu-sans": ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "dejavu-serif": ["/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"],
    "free-serif": ["/usr/share/fonts/truetype/freefont/FreeSerif.ttf"],
    "caladea": ["/usr/share/fonts/truetype/crosextra/Caladea-Regular.ttf"],
    "carlito": ["/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf"],
}
BOLD_FONTS = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
HAND_FONTS = {
    "caveat": [str(FONT_DIR / "Caveat.woff")],
    "indie-flower": [str(FONT_DIR / "IndieFlower.woff")],
}

PAGE_SIZE = (850, 1100)  # (width, height): US-letter at 100 dpi


def _first_existing(paths: list[str]) -> str | None:
    for p in paths:
        if Path(p).exists():
            return p
    return None


def available_fonts(kind: str = "print") -> dict[str, str]:
    table = PRINT_FONTS if kind == "print" else HAND_FONTS
    out = {name: _first_existing(paths) for name, paths in table.items()}
    return {k: v for k, v in out.items() if v}


def load_font(path: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if path is None:
        return ImageFont.load_default(size=size)
    return ImageFont.truetype(path, size)


@dataclass
class Style:
    paper: tuple[int, int, int]
    ink: tuple[int, int, int]
    font_name: str
    font_path: str | None
    font_size: int
    line_gap: float

    def to_dict(self) -> dict:
        return {"paper": list(self.paper), "ink": list(self.ink), "font": self.font_name,
                "font_size": self.font_size, "line_gap": self.line_gap}


def random_style(rng: random.Random, hand: bool = False) -> Style:
    """Pick paper tone, ink colour and font.  Ranges are wide enough to make documents in a
    mixed bin distinguishable, but stay inside what real office paper looks like."""
    base = rng.randint(228, 252)
    tint = rng.choice([(0, 0, 0), (4, 2, -6), (-3, -1, 3), (2, 0, -10), (-6, -2, 0)])
    paper = tuple(int(np.clip(base + t, 0, 255)) for t in tint)
    if hand:
        ink = rng.choice([(20, 30, 110), (25, 25, 30), (10, 50, 120), (60, 20, 90)])
        fonts = available_fonts("hand") or available_fonts("print")
        size = rng.randint(30, 38)
    else:
        ink = rng.choice([(15, 15, 15), (30, 30, 35), (10, 10, 40), (40, 35, 30)])
        fonts = available_fonts("print")
        size = rng.randint(17, 23)
    name = rng.choice(sorted(fonts)) if fonts else "default"
    return Style(paper=paper, ink=ink, font_name=name, font_path=fonts.get(name),
                 font_size=size, line_gap=rng.uniform(1.3, 1.7))


def wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    """Greedy word wrap by rendered pixel width."""
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


class _Canvas:
    """Tiny helper that keeps a y-cursor while laying out text."""

    def __init__(self, style: Style, size=PAGE_SIZE, margin: int = 70):
        self.style = style
        self.img = Image.new("RGB", size, style.paper)
        self.draw = ImageDraw.Draw(self.img)
        self.font = load_font(style.font_path, style.font_size)
        self.margin = margin
        self.y = margin
        self.w, self.h = size

    @property
    def line_h(self) -> int:
        return int(self.style.font_size * self.style.line_gap)

    def room(self, lines: int = 1) -> bool:
        return self.y + lines * self.line_h < self.h - self.margin

    def text_line(self, s: str, x: int | None = None, font=None, fill=None) -> None:
        self.draw.text((self.margin if x is None else x, self.y), s,
                       font=font or self.font, fill=fill or self.style.ink)
        self.y += self.line_h

    def paragraph(self, s: str, indent: int = 0) -> None:
        for line in wrap(self.draw, s, self.font, self.w - 2 * self.margin - indent):
            if not self.room():
                return
            self.text_line(line, x=self.margin + indent)
        self.y += self.line_h // 2

    def fill_paragraphs(self, rng: random.Random) -> None:
        while self.room(3):
            self.paragraph(T.paragraph(rng))


def _bold(size: int):
    return load_font(_first_existing(BOLD_FONTS), size)


# --------------------------------------------------------------------------- page types


def render_memo(rng: random.Random, style: Style | None = None) -> tuple[Image.Image, dict]:
    style = style or random_style(rng)
    c = _Canvas(style)
    c.draw.text((c.margin, c.y), "MEMORANDUM", font=_bold(style.font_size + 14), fill=style.ink)
    c.y += int((style.font_size + 14) * 1.8)
    for label, value in [("TO:", T.person(rng)), ("FROM:", T.person(rng)),
                         ("DATE:", T.date(rng)), ("RE:", rng.choice(T.OBJECTS).capitalize())]:
        c.draw.text((c.margin, c.y), label, font=_bold(style.font_size), fill=style.ink)
        c.text_line(value, x=c.margin + 110)
    c.y += 6
    c.draw.line([(c.margin, c.y), (c.w - c.margin, c.y)], fill=style.ink, width=2)
    c.y += c.line_h
    c.fill_paragraphs(rng)
    return c.img, {"type": "memo", **style.to_dict()}


def render_letter(rng: random.Random, style: Style | None = None) -> tuple[Image.Image, dict]:
    style = style or random_style(rng)
    c = _Canvas(style)
    comp = T.company(rng)
    c.draw.text((c.margin, c.y), comp, font=_bold(style.font_size + 8), fill=style.ink)
    c.y += int(style.font_size * 2.2)
    c.text_line(f"{rng.randint(2, 99)} Marina Road, Lagos")
    c.text_line(T.date(rng))
    c.y += c.line_h
    c.text_line(f"Dear {T.person(rng)},")
    c.y += c.line_h // 2
    while c.room(6):
        c.paragraph(T.paragraph(rng))
    if c.room(3):
        c.text_line("Yours sincerely,")
        c.text_line(T.person(rng))
    return c.img, {"type": "letter", **style.to_dict()}


def render_invoice(rng: random.Random, style: Style | None = None) -> tuple[Image.Image, dict]:
    """An invoice with a ruled table: long horizontal/vertical rules are great cues for
    the matcher (and a realistic case, since invoices are what people shred)."""
    style = style or random_style(rng)
    c = _Canvas(style)
    c.draw.text((c.margin, c.y), "INVOICE", font=_bold(style.font_size + 16), fill=style.ink)
    c.draw.text((c.w - c.margin - 260, c.y), f"No. INV-{rng.randint(1000, 9999)}",
                font=c.font, fill=style.ink)
    c.y += int(style.font_size * 3)
    c.text_line(f"Bill to: {T.company(rng)}")
    c.text_line(f"Date: {T.date(rng)}")
    c.y += c.line_h
    cols = [c.margin, c.margin + 330, c.margin + 420, c.margin + 560, c.w - c.margin]
    heads = ["Description", "Qty", "Unit price", "Amount"]
    row_h = int(style.font_size * 1.9)
    top = c.y
    n_rows = rng.randint(8, 14)
    total = 0
    for r in range(n_rows + 1):
        y = top + r * row_h
        if r == 0:
            c.draw.rectangle([cols[0], y, cols[-1], y + row_h], fill=tuple(
                max(0, v - 25) for v in style.paper))
            for i, h in enumerate(heads):
                c.draw.text((cols[i] + 8, y + 6), h, font=_bold(style.font_size - 2), fill=style.ink)
        else:
            item = rng.choice(T.ITEMS)
            qty = rng.randint(1, 40)
            price = rng.randint(5, 900) * 100
            total += qty * price
            for i, v in enumerate([item, str(qty), f"{price:,}", f"{qty * price:,}"]):
                c.draw.text((cols[i] + 8, y + 6), v, font=c.font, fill=style.ink)
    bottom = top + (n_rows + 1) * row_h
    for r in range(n_rows + 2):
        y = top + r * row_h
        c.draw.line([(cols[0], y), (cols[-1], y)], fill=style.ink, width=1)
    for x in cols:
        c.draw.line([(x, top), (x, bottom)], fill=style.ink, width=1)
    c.y = bottom + c.line_h
    c.draw.text((cols[3] - 60, c.y), f"TOTAL: NGN {total:,}", font=_bold(style.font_size),
                fill=style.ink)
    c.y += 2 * c.line_h
    while c.room(3):
        c.paragraph(T.paragraph(rng, 1, 3))
    return c.img, {"type": "invoice", **style.to_dict()}


def _synthetic_photo(rng: random.Random, w: int, h: int) -> Image.Image:
    """A 'photo' made of smooth random colour fields plus a few blurred shapes.  It has the
    statistics that matter for matching (smooth gradients, colour regions, soft edges)
    without needing any copyrighted image."""
    nrng = np.random.default_rng(rng.randint(0, 2**31))
    small = nrng.uniform(0, 255, size=(4, 5, 3)).astype(np.uint8)
    base = Image.fromarray(small).resize((w, h), Image.BICUBIC)
    d = ImageDraw.Draw(base)
    for _ in range(6):
        x0, y0 = rng.randint(0, w), rng.randint(0, h)
        r = rng.randint(20, 90)
        col = tuple(rng.randint(0, 255) for _ in range(3))
        if rng.random() < 0.5:
            d.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=col)
        else:
            d.polygon([(x0, y0 - r), (x0 + r, y0 + r), (x0 - r, y0 + r)], fill=col)
    base = base.filter(ImageFilter.GaussianBlur(3))
    arr = np.array(base).astype(np.float32) + nrng.normal(0, 6, (h, w, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def render_photo_page(rng: random.Random, style: Style | None = None) -> tuple[Image.Image, dict]:
    style = style or random_style(rng)
    c = _Canvas(style)
    # Logo: a coloured circle + square + company name.
    col = tuple(rng.randint(0, 200) for _ in range(3))
    c.draw.ellipse([c.margin, c.y, c.margin + 60, c.y + 60], fill=col)
    c.draw.rectangle([c.margin + 35, c.y + 25, c.margin + 85, c.y + 75], outline=style.ink, width=4)
    c.draw.text((c.margin + 110, c.y + 15), T.company(rng), font=_bold(style.font_size + 10),
                fill=style.ink)
    c.y += 110
    c.paragraph(T.paragraph(rng, 2, 3))
    pw, ph = c.w - 2 * c.margin, rng.randint(300, 420)
    c.img.paste(_synthetic_photo(rng, pw, ph), (c.margin, c.y))
    c.y += ph + 10
    c.text_line("Figure 1: site photograph taken during the inspection.",
                font=load_font(style.font_path, max(12, style.font_size - 4)))
    c.y += c.line_h // 2
    c.fill_paragraphs(rng)
    return c.img, {"type": "photo", **style.to_dict()}


def render_handwriting(rng: random.Random, style: Style | None = None) -> tuple[Image.Image, dict]:
    """Handwriting-style page: script font, per-word baseline jitter, a slight per-line
    slope and size wobble, on ruled notebook paper.  Real handwriting lacks the perfectly
    regular baselines printed text has, which is exactly what stresses the matcher."""
    style = style or random_style(rng, hand=True)
    w, h = PAGE_SIZE
    img = Image.new("RGB", (w, h), style.paper)
    d = ImageDraw.Draw(img)
    rule = int(style.font_size * 1.25)
    rule_col = (170, 190, 225)
    for y in range(110, h - 40, rule):
        d.line([(0, y), (w, y)], fill=rule_col, width=1)
    d.line([(80, 0), (80, h)], fill=(225, 150, 150), width=2)
    y = 110 + rule
    while y < h - 60:
        x = 95 + rng.randint(-4, 8)
        slope = rng.uniform(-0.012, 0.012)
        words = T.sentence(rng).split() + T.sentence(rng).split()
        for word in words:
            size = style.font_size + rng.randint(-2, 2)
            font = load_font(style.font_path, size)
            wlen = d.textlength(word + " ", font=font)
            if x + wlen > w - 50:
                break
            dy = rng.uniform(-2.0, 2.0) + slope * (x - 95)
            d.text((x, y - size + dy + 6), word, font=font, fill=style.ink)
            x += wlen + rng.randint(-2, 4)
        y += rule
    style_d = style.to_dict()
    style_d["line_gap"] = rule / style.font_size
    return img, {"type": "handwriting", **style_d}


RENDERERS = {
    "memo": render_memo,
    "letter": render_letter,
    "invoice": render_invoice,
    "photo": render_photo_page,
    "handwriting": render_handwriting,
}


def render_page(kind: str, seed: int) -> tuple[np.ndarray, dict]:
    """Render page ``kind`` deterministically from ``seed``; returns (HxWx3 uint8, meta)."""
    rng = random.Random(seed)
    img, meta = RENDERERS[kind](rng)
    meta["seed"] = seed
    return np.array(img), meta
