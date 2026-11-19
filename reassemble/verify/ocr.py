"""OCR-based readability checks (Tesseract).

The pixel matcher only sees one seam at a time.  OCR adds a *global* signal: if strips
are in the right order, words come out as real dictionary words; if two strips are
swapped, the text around that seam turns into gibberish.  We use it to rerank the
solver's top candidates and to flag suspicious joins.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

WORDLIST_DIR = Path(__file__).parent / "wordlists"
# Tesseract's OpenMP threads thrash badly when several OCR jobs run in parallel; we
# parallelise across images instead, one thread each.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
_TOKEN = re.compile(r"[A-Za-zÀ-ÿẸẹỌọṢṣ]+")


@lru_cache(maxsize=8)
def load_words(langs: str = "en") -> frozenset[str]:
    """Union of the word lists for a comma-separated set of language codes."""
    words: set[str] = set()
    for code in langs.split(","):
        f = WORDLIST_DIR / f"{code.strip()}.txt"
        if f.exists():
            words.update(w.strip().lower() for w in f.read_text(encoding="utf-8").split() if w.strip())
    return frozenset(words)


def tesseract_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def flatten_for_ocr(img: Image.Image) -> Image.Image:
    """Composite onto white so the dark background between pieces isn't read as glyphs."""
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("L")


def ocr_words(img: Image.Image) -> list[dict]:
    """Run Tesseract and return word boxes: {text, conf, x, y, w, h}."""
    import pytesseract

    d = pytesseract.image_to_data(flatten_for_ocr(img), output_type=pytesseract.Output.DICT)
    out = []
    for i, t in enumerate(d["text"]):
        if t and t.strip():
            out.append({"text": t.strip(), "conf": float(d["conf"][i]), "x": d["left"][i],
                        "y": d["top"][i], "w": d["width"][i], "h": d["height"][i]})
    return out


def is_word(token: str, words: frozenset[str]) -> bool:
    t = token.lower()
    return t in words or (t.endswith("s") and t[:-1] in words) or len(t) == 1 and t in "ai"


def readability(words_found: list[dict], langs: str = "en") -> dict:
    """Readability = 0.6 * fraction of alphabetic tokens that are dictionary words
    + 0.4 * mean OCR confidence.  Both go down when strips are misplaced."""
    vocab = load_words(langs)
    toks, confs = [], []
    for w in words_found:
        for t in _TOKEN.findall(w["text"]):
            if len(t) >= 2:
                toks.append(t)
                confs.append(max(0.0, w["conf"]) / 100.0)
    if not toks:
        return {"score": 0.0, "dict_fraction": 0.0, "mean_conf": 0.0, "tokens": 0}
    good = np.mean([is_word(t, vocab) for t in toks])
    conf = float(np.mean(confs))
    return {"score": float(0.6 * good + 0.4 * conf), "dict_fraction": float(good),
            "mean_conf": conf, "tokens": len(toks)}


def suspicious_joins(words_found: list[dict], seams: dict[str, int], langs: str = "en") -> dict[str, int]:
    """Count, per seam (keyed by join id), non-dictionary words that straddle it.

    A word cut by a wrong seam usually comes out as a non-word (``reqiohal``), so a seam
    crossed by several non-words is a good candidate for human review.
    """
    vocab = load_words(langs)
    out = {k: 0 for k in seams}
    for w in words_found:
        toks = [t for t in _TOKEN.findall(w["text"]) if len(t) >= 3]
        if not toks or all(is_word(t, vocab) for t in toks):
            continue
        for k, x in seams.items():
            if w["x"] + 2 < x < w["x"] + w["w"] - 2:
                out[k] += 1
    return out


def extract_text(img: Image.Image, langs: str = "en") -> dict:
    """Full text plus per-word dictionary flags, for the OCR panel in the web app."""
    words_found = ocr_words(img)
    vocab = load_words(langs)
    for w in words_found:
        toks = _TOKEN.findall(w["text"])
        w["known"] = bool(toks) and all(is_word(t, vocab) or len(t) < 2 for t in toks)
    return {"words": words_found, "readability": readability(words_found, langs)}
