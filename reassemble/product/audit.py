"""Shredder security audit: how much of a shredded document can be recovered?

Feed in the output of *your own* shredder (a case folder of pieces, or a photo on the
capture mat).  We reconstruct it, read the result with OCR and turn the numbers into a
plain-language verdict plus a one-page PDF report.

The verdict follows the spirit of DIN 66399 security levels: the decisive question is
whether an automated tool recovered readable text, and how big the particles are.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..model import Piece, Reconstruction
from ..pipeline import (SolveOptions, detect_shape, prepare, solve_grid_pieces, solve_prepared,
                        solve_torn_pieces)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


@dataclass
class AuditResult:
    shape: str
    n_pieces: int
    piece_width_mm: float
    piece_area_mm2: float
    seconds: float
    mean_confidence: float
    readability: float
    dict_fraction: float
    words_recovered: int
    text_recovered_pct: float | None    # vs ground truth / original, when available
    verdict: str
    level: str
    advice: str
    details: dict = field(default_factory=dict)


def _font(size: int, bold: bool = False):
    try:
        return ImageFont.truetype(FONT_B if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def classify(shape: str, width_mm: float, area_mm2: float, recovered: float,
             seconds: float = 0.0) -> tuple[str, str, str]:
    """(verdict, level, advice).

    ``recovered`` is the fraction of text that came back readable (0-1).
    """
    if recovered >= 0.5:
        verdict = f"UNSAFE: {recovered:.0%} of the text was recovered automatically in {max(seconds, 1):.0f} s."
    elif recovered >= 0.15:
        verdict = f"WEAK: {recovered:.0%} of the text was recovered; fragments of sensitive data are exposed."
    else:
        verdict = "RESISTANT to this tool: little readable text was recovered."
    if shape == "strip":
        level = "strip-cut (~DIN 66399 P-1/P-2)"
        advice = "Strip-cut shredders are unsafe for sensitive documents. Use cross-cut (P-4) or micro-cut (P-5+)."
    elif shape == "grid" and area_mm2 > 160:
        level = "cross-cut, large particles (~P-3)"
        advice = "Acceptable for internal paperwork; use P-4 or better for personal or financial data."
    elif shape == "grid":
        level = "cross-cut / micro-cut (~P-4 or better)"
        advice = "Suitable for most personal and confidential documents."
    else:
        level = "hand-torn"
        advice = "Tearing by hand is not a security measure: large pieces are easy to re-assemble."
    return verdict, level, advice


def _text_overlap(a: list[str], b: list[str]) -> float:
    """Fraction of reference words (with multiplicity) found in the recovered words."""
    from collections import Counter
    ra, rb = Counter(w.lower() for w in a), Counter(w.lower() for w in b)
    tot = sum(ra.values())
    return sum(min(c, rb[w]) for w, c in ra.items()) / tot if tot else 0.0


def audit(pieces: list[Piece], dpi: float = 100.0, original: np.ndarray | None = None,
          opts: SolveOptions | None = None) -> tuple[AuditResult, Image.Image]:
    from ..render.compose import render_grid, render_strips
    from ..verify.ocr import ocr_words, readability, tesseract_available
    import re

    t0 = time.perf_counter()
    shape = detect_shape(pieces)
    if shape == "strip":
        prep = prepare(pieces)
        rec = solve_prepared(prep, opts or SolveOptions())
        img = render_strips({p.id: p for p in prep.pieces}, rec, background=(255, 255, 255, 255))
    elif shape == "grid":
        rec = solve_grid_pieces(pieces)
        img = render_grid({p.id: p for p in pieces}, rec.grid, background=(255, 255, 255, 255))
    else:
        from ..solve.torn import TornResult, render_torn
        rec = solve_torn_pieces(pieces)
        ids = [p.id for p in pieces]
        img = render_torn(pieces, TornResult({ids.index(k): tuple(v) for k, v in rec.extra["positions"].items()}, [], [], 0),
                          background=(255, 255, 255, 255))
    seconds = time.perf_counter() - t0
    mm = 25.4 / dpi
    w_mm = float(np.median([p.width for p in pieces])) * mm
    a_mm2 = float(np.median([p.mask.sum() for p in pieces])) * mm * mm
    conf = float(np.mean([j["confidence"] for j in rec.joins])) if rec.joins else 0.0
    read = {"score": 0.0, "dict_fraction": 0.0, "tokens": 0}
    rec_words: list[str] = []
    pct = None
    if tesseract_available():
        words = ocr_words(img)
        read = readability(words)
        tok = re.compile(r"[A-Za-z]{2,}")
        rec_words = [t for w in words for t in tok.findall(w["text"])]
        if original is not None:
            ref = [t for w in ocr_words(Image.fromarray(original)) for t in tok.findall(w["text"])]
            pct = _text_overlap(ref, rec_words)
    if pct is not None:
        pct = round(pct, 3)
    recovered = pct if pct is not None else read["dict_fraction"] * min(1.0, conf + 0.25)
    verdict, level, advice = classify(shape, w_mm, a_mm2, recovered, seconds)
    res = AuditResult(shape, len(pieces), round(w_mm, 1), round(a_mm2, 1), round(seconds, 1), round(conf, 3),
                      round(read["score"], 3), round(read["dict_fraction"], 3), len(rec_words),
                      pct, verdict, level, advice,
                      {"method": rec.method})
    return res, img


def report_pdf(res: AuditResult, recon: Image.Image, path: str | Path) -> Path:
    """One-page A4 PDF: verdict, numbers, thumbnail of the reconstruction."""
    W, H = 1240, 1754  # A4 at 150 dpi
    page = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(page)
    d.rectangle([0, 0, W, 150], fill=(14, 17, 22))
    d.text((60, 45), "REASSEMBLE  ·  Shredder security audit", font=_font(40, True), fill="white")
    d.text((60, 100), time.strftime("%d %B %Y"), font=_font(22), fill=(150, 160, 175))
    bad = res.verdict.startswith("UNSAFE")
    weak = res.verdict.startswith("WEAK")
    col = (220, 38, 38) if bad else (202, 138, 4) if weak else (22, 163, 74)
    d.rectangle([60, 190, W - 60, 330], outline=col, width=5)
    y = 205
    for line in _wrap(d, res.verdict, _font(30, True), W - 160):
        d.text((85, y), line, font=_font(30, True), fill=col)
        y += 40
    d.text((85, 290), res.level, font=_font(22), fill=(60, 60, 60))
    rows = [("Pieces", res.n_pieces), ("Shred type", res.shape),
            ("Median particle width", f"{res.piece_width_mm} mm"), ("Median particle area", f"{res.piece_area_mm2} mm²"),
            ("Time to reconstruct", f"{res.seconds} s on a laptop CPU"),
            ("Mean join confidence", f"{res.mean_confidence:.0%}"),
            ("OCR readability", f"{res.readability:.0%}"),
            ("Dictionary words recovered", res.words_recovered)]
    if res.text_recovered_pct is not None:
        rows.append(("Text recovered vs original", f"{res.text_recovered_pct:.0%}"))
    y = 370
    for k, v in rows:
        d.text((60, y), k, font=_font(24), fill=(90, 90, 90))
        d.text((560, y), str(v), font=_font(24, True), fill=(20, 20, 20))
        y += 42
    y += 20
    d.text((60, y), "Recommendation", font=_font(26, True), fill=(20, 20, 20))
    y += 40
    for line in _wrap(d, res.advice, _font(24), W - 120):
        d.text((60, y), line, font=_font(24), fill=(40, 40, 40))
        y += 34
    y += 20
    thumb = recon.convert("RGB")
    thumb.thumbnail((W - 120, H - y - 120))
    page.paste(thumb, ((W - thumb.width) // 2, y))
    d.text((60, H - 70), "Reassemble is for testing your own shredders, archives and education. "
                         "Never use it on documents you have no right to access.", font=_font(18), fill=(120, 120, 120))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    page.save(path, "PDF", resolution=150.0)
    return path


def _wrap(d: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        t = f"{cur} {w}".strip()
        if d.textlength(t, font=font) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def audit_dict(res: AuditResult) -> dict:
    return asdict(res)
