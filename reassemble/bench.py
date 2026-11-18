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
