"""A tiny local English language model for checking reconstructions.

Why: the dictionary check in ``ocr.py`` only asks "is each token a word?".  When two strips
are swapped the OCR often still produces real words, just in a nonsensical order or split
across the seam ("the re port", "ma nager").  A character-level language model scores how
*English-like the running text* is, which catches those cases.

What: a character 5-gram model with **Witten–Bell interpolation**.  The probability of the
next character mixes what we saw after the full 4-character history with what we saw after
shorter histories, weighting the longer history by how much evidence it has:

    P(c | h) = (C(h, c) + T(h) * P(c | h')) / (C(h) + T(h))

where C counts occurrences, T(h) is the number of distinct characters ever seen after h,
and h' is h without its first character.  It needs no training loop, no GPU and no
network, and it is small enough to ship (``models/char_lm_en.json.gz``).

Training text: English prose extracted from the documentation strings of installed Python
packages (filtered to paragraph-like text).  We deliberately do **not** train on the
synthetic page generator's sentences, so benchmark gains aren't the model memorising the
test documents.  Only English is supported for now.
"""

from __future__ import annotations

import ast
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

ALPHABET = "abcdefghijklmnopqrstuvwxyz .,'"
ORDER = 5
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_MODEL = MODEL_DIR / "char_lm_en.json.gz"

_NON_ALPHA = re.compile(r"[^a-z .,']+")
_SPACES = re.compile(r"\s+")


def normalise(text: str) -> str:
    t = _NON_ALPHA.sub(" ", text.lower().replace("\n", " "))
    return _SPACES.sub(" ", t).strip()


# --------------------------------------------------------------------------- corpus


def _prose_paragraphs(doc: str):
    """Yield paragraphs that look like English prose, not code or parameter lists."""
    for para in re.split(r"\n\s*\n", doc):
        p = " ".join(line.strip() for line in para.splitlines())
        if len(p) < 80 or any(s in p for s in (">>>", "::", "==", "->", "{", "}", "[", "]", "_", "`", "=", "|")):
            continue
        words = p.split()
        alpha = sum(w.strip(".,;:()'\"").isalpha() for w in words)
        if alpha / max(len(words), 1) < 0.85 or not p[0].isupper() or p[-1] not in ".!?":
            continue
        yield p


def build_corpus(roots: list[str | Path], max_chars: int = 6_000_000) -> str:
    """English prose from the docstrings of Python files under ``roots``."""
    out, total = [], 0
    for root in roots:
        for f in sorted(Path(root).rglob("*.py")):
            if "reassemble" in f.parts:
                continue  # never train on our own synthetic-text templates
            try:
                tree = ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
            except (SyntaxError, ValueError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    doc = ast.get_docstring(node)
                    if doc:
                        for p in _prose_paragraphs(doc):
                            n = normalise(p)
                            out.append(n)
                            total += len(n) + 1
                            if total >= max_chars:
                                return " ".join(out)
    return " ".join(out)


# --------------------------------------------------------------------------- model


class CharLM:
    def __init__(self, counts: dict[str, dict[str, int]], order: int = ORDER):
        self.order = order
        self.counts = counts                     # history -> {next char: count}
        self.totals = {h: sum(c.values()) for h, c in counts.items()}
        self.types = {h: len(c) for h, c in counts.items()}

    @classmethod
    def train(cls, text: str, order: int = ORDER, min_count: int = 2) -> "CharLM":
        text = " " + normalise(text) + " "
        counts: dict[str, Counter] = defaultdict(Counter)
        for i in range(1, len(text)):
            c = text[i]
            for k in range(0, order):
                if i - k < 0:
                    break
                counts[text[i - k:i]][c] += 1
        pruned = {}
        for h, cs in counts.items():
            keep = {c: n for c, n in cs.items() if n >= min_count or len(h) <= 1}
            if keep:
                pruned[h] = keep
        return cls(pruned, order)

    def prob(self, history: str, c: str) -> float:
        p = 1.0 / len(ALPHABET)  # uniform base distribution
        for k in range(0, self.order):
            h = history[len(history) - k:] if k else ""
            if k > len(history):
                break
            cs = self.counts.get(h)
            if not cs:
                continue
            tot, typ = self.totals[h], self.types[h]
            p = (cs.get(c, 0) + typ * p) / (tot + typ)
        return p

    def logprob_per_char(self, text: str) -> float:
        """Mean log2 probability per character (higher = more English-like)."""
        t = " " + normalise(text)
        if len(t) < 2:
            return -math.log2(len(ALPHABET))
        lp = 0.0
        for i in range(1, len(t)):
            lp += math.log2(self.prob(t[max(0, i - self.order + 1):i], t[i]))
        return lp / (len(t) - 1)

    def save(self, path: str | Path = DEFAULT_MODEL) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump({"order": self.order, "counts": self.counts}, f, separators=(",", ":"))
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL) -> "CharLM":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["counts"], d["order"])


@lru_cache(maxsize=2)
def default_lm(path: str = str(DEFAULT_MODEL)) -> CharLM | None:
    p = Path(path)
    return CharLM.load(p) if p.exists() else None


def train_default(roots: list[str | Path] | None = None, out: str | Path = DEFAULT_MODEL) -> dict:
    import site
    import sys
    roots = roots or [p for p in site.getsitepackages() + [str(Path(sys.prefix) / "lib")] if Path(p).exists()]
    text = build_corpus(roots)
    lm = CharLM.train(text)
    path = lm.save(out)
    default_lm.cache_clear()
    return {"chars": len(text), "contexts": len(lm.counts), "path": str(path),
            "bytes": path.stat().st_size}
