"""Benchmark harness: solve many generated cases and report accuracy per method."""

from __future__ import annotations

import time

import numpy as np

from .metrics import evaluate_strips
from .pipeline import SolveOptions, prepare, solve_prepared
from .synth.pages import render_page
from .synth.shredders import Realism, strip_shred


def strip_benchmark(kinds=("memo", "letter", "invoice", "photo", "handwriting"), n: int = 30,
                    noisy: bool = False, seeds=range(100, 103), opts: SolveOptions | None = None,
                    drop: int = 0, verbose: bool = False) -> dict:
    """Return {kind: {metric: mean}} plus an 'all' row; per-method neighbour accuracy too."""
    opts = opts or SolveOptions()
    rows: dict[str, list[dict]] = {}
    for kind in kinds:
        for seed in seeds:
            page, _ = render_page(kind, seed)
            r = Realism.noisy(drop) if noisy else Realism(drop=drop)
            pieces, truth = strip_shred(page, n, r, seed=seed)
            t0 = time.perf_counter()
            prep = prepare(pieces)
            rec = solve_prepared(prep, opts)
            m = evaluate_strips(rec.order, truth, prep.blank)
            m["runtime_s"] = time.perf_counter() - t0
            for meth, info in rec.extra["methods"].items():
                m[f"nb_{meth}"] = evaluate_strips(info["order"], truth)["neighbour"]
            if "gaps" in m:
                g = m.pop("gaps")
                m["gap_recall"], m["gap_precision"] = g["recall"], g["precision"]
            rows.setdefault(kind, []).append(m)
            if verbose:
                print(kind, seed, {k: round(v, 3) for k, v in m.items()})
    out = {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in rows.items()}
    allr = [r for v in rows.values() for r in v]
    out["all"] = {m: float(np.mean([r[m] for r in allr])) for m in allr[0]}
    return out


def print_table(res: dict, cols=None) -> None:
    cols = cols or list(next(iter(res.values())).keys())
    print(f"{'case':<12}" + "".join(f"{c[:14]:>15}" for c in cols))
    for k, v in res.items():
        print(f"{k:<12}" + "".join(f"{v.get(c, float('nan')):>15.3f}" for c in cols))


def double_sided_benchmark(seeds=range(200, 210), n: int = 40) -> dict:
    """Front-only vs front+back matching on noisy double-sided pages."""
    import random as _r
    kinds = ["memo", "letter", "invoice", "handwriting", "photo"]
    out = {False: [], True: []}
    for seed in seeds:
        rng = _r.Random(seed)
        front, _ = render_page(rng.choice(kinds), seed)
        back, _ = render_page(rng.choice(kinds), seed + 1000)
        pieces, truth = strip_shred(front, n, Realism.noisy(), seed=seed, back=back)
        prep = prepare(pieces)
        for ub in (False, True):
            rec = solve_prepared(prep, SolveOptions(use_back=ub))
            out[ub].append(evaluate_strips(rec.order, truth, prep.blank))
    return {("front+back" if k else "front only"): {m: float(np.mean([r[m] for r in v])) for m in v[0]}
            for k, v in out.items()}


def mixed_benchmark(k: int = 3, seeds=range(500, 508), n: int = 30) -> dict:
    """Purity / document count / per-document accuracy on noisy mixed bins of k pages."""
    import random as _r
    from .metrics import evaluate_mixed
    from .pipeline import solve_mixed
    from .synth.shredders import mixed_bin
    kinds = ["memo", "letter", "invoice", "handwriting", "photo"]
    rows = []
    for seed in seeds:
        rng = _r.Random(seed)
        pages = [render_page(kd, seed * 10 + j)[0] for j, kd in enumerate(rng.sample(kinds, k))]
        pieces, truth = mixed_bin(pages, n, Realism.noisy(), seed=seed)
        prep = prepare(pieces)
        rec = solve_mixed(prep, SolveOptions())
        m = evaluate_mixed(rec.documents, truth, prep.blank)
        m["k_correct"] = float(m["n_docs_found"] == m["n_docs_true"])
        rows.append(m)
    keys = ["purity", "k_correct", "neighbour_content_mean"]
    return {f"{k} docs": {m: float(np.mean([r[m] for r in rows])) for m in keys}}


def ocr_benchmark(kinds=("memo", "letter", "invoice", "handwriting"), seeds=range(100, 106)) -> dict:
    """Content-neighbour accuracy on noisy pages without / with OCR reranking."""
    out = {"without OCR": [], "with OCR rerank": []}
    for kind in kinds:
        for seed in seeds:
            page, _ = render_page(kind, seed)
            pieces, truth = strip_shred(page, 30, Realism.noisy(), seed=seed)
            prep = prepare(pieces)
            for label, ocr in (("without OCR", False), ("with OCR rerank", True)):
                rec = solve_prepared(prep, SolveOptions(ocr_rerank=ocr, detect_gaps=False))
                out[label].append(evaluate_strips(rec.order, truth, prep.blank))
    return {k: {m: float(np.mean([r[m] for r in v])) for m in ("neighbour", "neighbour_content", "direct_content")}
            for k, v in out.items()}
