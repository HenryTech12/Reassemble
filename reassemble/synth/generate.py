"""Generate the sample suite: one folder per test case with pieces + truth.json.

``make demo-synth`` calls :func:`generate_suite`.  Each case folder also contains
``original.png`` (the unshredded page) so you can eyeball reconstructions.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from ..model import save_case
from .pages import render_page
from .shredders import Realism, crosscut_shred, mixed_bin, strip_shred, torn_shred

PAGE_KINDS = ["memo", "letter", "invoice", "photo", "handwriting"]


def _save(folder: Path, pieces, truth, original=None, back_original=None, page_meta=None):
    if page_meta is not None:
        truth["page"] = page_meta
    save_case(folder, pieces, truth)
    if original is not None:
        Image.fromarray(original).save(folder / "original.png")
    if back_original is not None:
        Image.fromarray(back_original).save(folder / "original_back.png")
    return folder


def make_strip_case(out: Path, kind: str, n: int = 30, noisy: bool = False, seed: int = 1,
                    drop: int = 0) -> Path:
    page, meta = render_page(kind, seed)
    r = Realism.noisy(drop) if noisy else Realism(drop=drop)
    pieces, truth = strip_shred(page, n, r, seed=seed)
    return _save(out, pieces, truth, page, page_meta=meta)


def generate_suite(out_dir: str | Path = "samples", seed: int = 1) -> list[Path]:
    """Write every Phase-0 case into ``out_dir`` and return the folders created."""
    out = Path(out_dir)
    made = []
    # 1. clean + realistic strip shreds for every page type
    for i, kind in enumerate(PAGE_KINDS):
        made.append(make_strip_case(out / f"{kind}_30", kind, 30, False, seed + i))
        made.append(make_strip_case(out / f"{kind}_30_noisy", kind, 30, True, seed + i))
    # 2. different strip counts
    made.append(make_strip_case(out / "memo_20", "memo", 20, False, seed + 10))
    made.append(make_strip_case(out / "memo_60", "memo", 60, False, seed + 11))
    # 3. missing pieces
    made.append(make_strip_case(out / "letter_30_missing2", "letter", 30, True, seed + 12, drop=2))
    # 4. cross-cut
    page, meta = render_page("letter", seed + 13)
    for rows, cols in [(6, 8), (10, 12)]:
        pcs, tr = crosscut_shred(page, rows, cols, seed=seed + 13)
        made.append(_save(out / f"crosscut_{rows}x{cols}", pcs, tr, page, page_meta=meta))
    # 5. torn paper
    page, meta = render_page("memo", seed + 14)
    pcs, tr = torn_shred(page, rows=2, cols=4, seed=seed + 14)
    made.append(_save(out / "torn_2x4", pcs, tr, page, page_meta=meta))
    # 6. double-sided
    front, fmeta = render_page("letter", seed + 15)
    back, bmeta = render_page("memo", seed + 16)
    pcs, tr = strip_shred(front, 30, Realism.noisy(), seed=seed + 15, back=back)
    tr["back_page"] = bmeta
    made.append(_save(out / "double_30_noisy", pcs, tr, front, back, page_meta=fmeta))
    # 7. mixed bins
    for k, kinds in [(3, ["memo", "invoice", "letter"]),
                     (5, ["memo", "invoice", "letter", "handwriting", "photo"])]:
        pages, metas = zip(*(render_page(kd, seed + 20 + j) for j, kd in enumerate(kinds)))
        pcs, tr = mixed_bin(list(pages), 30, Realism.noisy(), seed=seed + 20 + k)
        for d, m in zip(tr["documents"], metas):
            d["page"] = m
        folder = _save(out / f"mixed_{k}", pcs, tr)
        for j, pg in enumerate(pages):
            Image.fromarray(pg).save(folder / f"original_{j}.png")
        made.append(folder)
    return made
