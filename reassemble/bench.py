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
            for meth, info in rec.extra.get("methods", {}).items():
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


def scorer_benchmark(scorers=("handcrafted", "learned", "combined"), seeds=range(700, 704)) -> dict:
    """Compare edge scorers on several conditions (content-neighbour accuracy).

    Conditions: clean / realistic printed pages, heavy degradation, handwriting only,
    60 strips, and pages whose fonts were *held out* from the CNN's training set.
    """
    from dataclasses import replace

    from .match.learned import VAL_FONTS
    heavy = replace(Realism.noisy(), blur_sigma=1.2, noise_sigma=10.0)
    conds = {
        "clean": (("memo", "letter", "invoice", "photo"), Realism(), 30, None),
        "realistic": (("memo", "letter", "invoice", "photo"), Realism.noisy(), 30, None),
        "heavy noise+blur": (("memo", "letter", "invoice"), heavy, 30, None),
        "handwriting": (("handwriting",), Realism.noisy(), 30, None),
        "60 strips": (("memo", "letter", "invoice"), Realism.noisy(), 60, None),
        "held-out fonts": (("memo", "letter", "invoice", "handwriting"), Realism.noisy(), 30, VAL_FONTS),
    }
    table = {}
    for cname, (kinds, real, n, fonts) in conds.items():
        cases = []
        for kind in kinds:
            s, found = 0, 0
            while found < len(seeds):
                seed = seeds[0] + s
                s += 1
                page, meta = render_page(kind, seed)
                if fonts is not None and meta["font"] not in fonts:
                    continue
                found += 1
                cases.append((page, seed))
        rows = {sc: [] for sc in scorers}
        for page, seed in cases:
            pieces, truth = strip_shred(page, n, real, seed=seed)
            prep = prepare(pieces)
            for sc in scorers:
                rec = solve_prepared(prep, SolveOptions(scorer=sc, detect_gaps=False))
                rows[sc].append(evaluate_strips(rec.order, truth, prep.blank)["neighbour_content"])
        table[cname] = {sc: float(np.mean(v)) for sc, v in rows.items()}
        print(cname, {k: round(v, 3) for k, v in table[cname].items()}, flush=True)
    return table


def crosscut_benchmark(sizes=((3, 2), (4, 3), (6, 5), (8, 6), (10, 8)), seeds=range(40, 43),
                       kinds=("letter", "memo")) -> dict:
    """Cross-cut accuracy by grid size (clean scans)."""
    from .features.edges import extract
    from .solve.grid import evaluate_grid, solve_grid
    from .synth.shredders import crosscut_shred
    out = {}
    for rows, cols in sizes:
        ms = []
        for kind in kinds:
            for seed in seeds:
                page, _ = render_page(kind, seed)
                pieces, truth = crosscut_shred(page, rows, cols, seed=seed)
                blank = {p.id for p in pieces if extract(p).ink_fraction < 0.002}
                res, _, _ = solve_grid(pieces, rows, cols)
                ids = [p.id for p in pieces]
                pred = [[ids[x] if x is not None else None for x in r] for r in res.grid]
                m = evaluate_grid(pred, truth["grid"], blank)
                m["runtime_s"] = res.runtime_s
                ms.append(m)
        piece = f"{850 // cols}x{1100 // rows}px"
        out[f"{rows}x{cols} ({piece})"] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    return out


def torn_benchmark(seeds=range(20, 23), kinds=("memo", "letter", "photo", "handwriting")) -> dict:
    from dataclasses import replace
    from .solve.torn import assemble, evaluate_torn
    from .synth.shredders import torn_shred
    conds = {"clean, 14px tears": (Realism(), 14), "clean, 6px tears": (Realism(), 6),
             "noisy (no rotation)": (replace(Realism.noisy(), rotation_deg=0), 10),
             "noisy + 0.4 deg rotation": (Realism.noisy(), 10)}
    out = {}
    for name, (r, amp) in conds.items():
        for shape in ((2, 4), (3, 4)):
            ms = []
            for kind in kinds:
                for seed in seeds:
                    page, _ = render_page(kind, seed)
                    pcs, t = torn_shred(page, *shape, amplitude=amp, seed=seed, realism=r)
                    res = assemble(pcs)
                    m = evaluate_torn(pcs, res, t)
                    m["runtime_s"] = res.runtime_s
                    ms.append(m)
            out[f"{name} {shape[0]}x{shape[1]}"] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    return out


def capture_benchmark(kinds=("memo", "letter", "invoice", "handwriting", "photo"), seeds=(33, 34), n: int = 24) -> dict:
    """Simulated phone photo -> capture -> solve, content-neighbour accuracy per scorer."""
    from .capture.pipeline import capture
    from .synth.photo import compose_photo
    out = {"handcrafted": [], "learned": [], "pieces_found": []}
    for kind in kinds:
        for seed in seeds:
            page, _ = render_page(kind, seed)
            pieces, truth = strip_shred(page, n, Realism(noise_sigma=2), seed=seed)
            photo, pt = compose_photo(pieces, seed=seed)
            cap, rep = capture(photo)
            cen = pt["centres_mat"]
            mp = {p.id: min(cen, key=lambda k: np.linalg.norm(np.array(cen[k]) - np.array(p.meta["centre"])))
                  for p in cap}
            out["pieces_found"].append(len(set(mp.values())) / n)
            prep = prepare(cap)
            for sc in ("handcrafted", "learned"):
                rec = solve_prepared(prep, SolveOptions(scorer=sc))
                order = [mp[x] if x else None for x in rec.order]
                out[sc].append(evaluate_strips(order, truth, {mp[b] for b in prep.blank})["neighbour_content"])
    return {"simulated photo": {k: float(np.mean(v)) for k, v in out.items()}}
