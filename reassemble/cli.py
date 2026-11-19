"""Command-line interface: ``python -m reassemble.cli <command>`` (or ``reassemble``)."""

from __future__ import annotations

import argparse
import sys


def cmd_synth(args: argparse.Namespace) -> int:
    from .synth.generate import generate_suite, make_strip_case

    if args.kind:
        folder = make_strip_case(__import__("pathlib").Path(args.out), args.kind, args.n,
                                 args.noisy, args.seed, args.drop)
        print(f"wrote {folder}")
    else:
        for f in generate_suite(args.out, args.seed):
            print(f"wrote {f}")
    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from .metrics import evaluate_strips
    from .model import load_case
    from .pipeline import SolveOptions, detect_shape, prepare, solve_mixed, solve_prepared
    from .render.compose import render_strips

    pieces, truth = load_case(args.case)
    if not pieces:
        print(f"no pieces found in {args.case}")
        return 1
    opts = SolveOptions(methods=tuple(args.methods.split(",")), beam_width=args.beam,
                        scorer=args.scorer, ocr_rerank=args.ocr, langs=args.lang)
    shape = args.shape
    if shape == "auto":
        kind = (truth or {}).get("kind")
        shape = {"crosscut": "grid", "torn": "torn"}.get(kind) or detect_shape(pieces)
    if shape in ("grid", "torn"):
        return _solve_2d(args, pieces, truth, shape)
    prep = prepare(pieces)
    prog = (lambda e: print("  ..", e["stage"])) if args.verbose else None
    mixed = args.mixed or (truth or {}).get("kind") == "mixed"
    rec = solve_mixed(prep, opts, k=args.docs, progress=prog) if mixed else solve_prepared(prep, opts, prog)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by_id = {p.id: p for p in prep.pieces}
    from .model import Reconstruction
    if rec.documents:
        for d in rec.documents:
            r = Reconstruction.from_dict(d)
            render_strips(by_id, r).save(out / f"reconstruction_doc{d['doc']}.png")
            render_strips(by_id, r, heatmap=True).save(out / f"heatmap_doc{d['doc']}.png")
        print(f"found {len(rec.documents)} document(s)")
    render_strips(by_id, rec).save(out / "reconstruction.png")
    render_strips(by_id, rec, heatmap=True).save(out / "heatmap.png")
    result = rec.to_dict()
    if truth and truth.get("kind") == "mixed":
        from .metrics import evaluate_mixed
        result["metrics"] = evaluate_mixed(rec.documents, truth, prep.blank)
        print("mixed: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in result["metrics"].items()))
    print(f"solved {len(pieces)} pieces with {rec.method} in {rec.runtime_s:.2f}s, "
          f"{len(rec.gaps)} gap(s) detected")
    if truth and truth.get("kind") == "strip":
        m = evaluate_strips(rec.order, truth, prep.blank)
        result["metrics"] = m
        print("accuracy: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items() if isinstance(v, float)))
        for meth, info in rec.extra["methods"].items():
            print(f"  {meth:<7} neighbour={evaluate_strips(info['order'], truth)['neighbour']:.3f} "
                  f"cost={info['cost']:.2f} time={info['runtime_s']:.2f}s")
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(f"wrote {out}/reconstruction.png, heatmap.png, result.json")
    return 0


def _solve_2d(args, pieces, truth, shape) -> int:
    import json
    from pathlib import Path

    from .pipeline import solve_grid_pieces, solve_torn_pieces

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if shape == "grid":
        from .render.compose import render_grid
        from .solve.grid import evaluate_grid
        rows = truth.get("rows") if truth else args.rows
        cols = truth.get("cols") if truth else args.cols
        rec = solve_grid_pieces(pieces, rows or args.rows, cols or args.cols)
        render_grid({p.id: p for p in pieces}, rec.grid).save(out / "reconstruction.png")
        result = rec.to_dict()
        if truth:
            result["metrics"] = evaluate_grid(rec.grid, truth["grid"])
    else:
        from .solve.torn import TornResult, evaluate_torn, render_torn
        rec = solve_torn_pieces(pieces)
        ids = [p.id for p in pieces]
        tr = TornResult({ids.index(k): tuple(v) for k, v in rec.extra["positions"].items()}, [], [], 0)
        render_torn(pieces, tr).save(out / "reconstruction.png")
        result = rec.to_dict()
        if truth:
            result["metrics"] = evaluate_torn(pieces, tr, truth)
    print(f"solved {len(pieces)} {shape} pieces in {rec.runtime_s:.2f}s")
    if "metrics" in result:
        print("accuracy: " + ", ".join(f"{k}={v:.3f}" for k, v in result["metrics"].items()))
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(f"wrote {out}/reconstruction.png, result.json")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from . import bench as B
    from .bench import print_table, strip_benchmark
    from .pipeline import SolveOptions

    if args.suite == "mixed":
        for k in (2, 3, 5):
            print_table(B.mixed_benchmark(k))
        return 0
    if args.suite == "double":
        print_table(B.double_sided_benchmark())
        return 0
    if args.suite == "crosscut":
        print_table(B.crosscut_benchmark())
        return 0
    if args.suite == "capture":
        print_table(B.capture_benchmark())
        return 0
    if args.suite == "torn":
        print_table(B.torn_benchmark())
        return 0
    if args.suite == "scorers":
        B.scorer_benchmark()
        return 0
    if args.suite == "ocr":
        print_table(B.ocr_benchmark())
        return 0

    seeds = range(args.seed, args.seed + args.seeds)
    for noisy, drop, label in [(False, 0, "clean"), (True, 0, "noisy"), (True, 2, "noisy, 2 missing")]:
        opts = SolveOptions(scorer=args.scorer, ocr_rerank=args.ocr)
        res = strip_benchmark(n=args.n, noisy=noisy, seeds=seeds, opts=opts, drop=drop)
        print(f"\n### {args.n} strips, {label} ({len(seeds)} pages per type, scorer={args.scorer})")
        print_table(res)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from .match.learned import train

    stats = train(args.out, n_pages=args.pages, epochs=args.epochs, seed=args.seed)
    Path(args.out).with_suffix(".json").write_text(json.dumps(stats, indent=1))
    print(f"saved {args.out} ({stats['seconds']:.0f}s)")
    return 0


def cmd_finetune(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from .match.learned import DEFAULT_WEIGHTS, finetune_from_corrections

    corrections = json.loads(Path(args.corrections).read_text())
    if isinstance(corrections, dict):
        corrections = corrections.get("pairs", [])
    stats = finetune_from_corrections(corrections, init=args.init or DEFAULT_WEIGHTS, out=args.out)
    print(f"fine-tuned on {len(corrections)} human pairs in {stats['seconds']:.0f}s -> {args.out}")
    return 0


def cmd_synth_photo(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    import cv2

    from .synth.pages import render_page
    from .synth.photo import compose_photo
    from .synth.shredders import Realism, strip_shred

    page, meta = render_page(args.kind, args.seed)
    pieces, truth = strip_shred(page, args.n, Realism(noise_sigma=2), seed=args.seed)
    photo, pt = compose_photo(pieces, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "photo.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 92])
    truth.update(pt)
    truth["page"] = meta
    (out / "photo_truth.json").write_text(json.dumps(truth))
    print(f"wrote {out}/photo.jpg ({photo.shape[1]}x{photo.shape[0]}) and photo_truth.json")
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    import cv2
    import numpy as np

    from .capture.pipeline import capture
    from .model import save_case

    img = cv2.imread(args.photo, cv2.IMREAD_COLOR)
    if img is None:
        print(f"cannot read {args.photo}")
        return 1
    pieces, rep = capture(img, mat_scale=args.scale, out_scale=args.out_scale)
    truth = None
    tf = Path(args.photo).with_name("photo_truth.json")
    if tf.exists():  # simulated photo: translate ground truth to the captured piece ids
        pt = json.loads(tf.read_text())
        cen = pt["centres_mat"]
        mapping = {}
        for p in pieces:
            c = np.array(p.meta["centre"])
            mapping[min(cen, key=lambda k: np.linalg.norm(np.array(cen[k]) - c))] = p.id
        truth = {"kind": "strip", "order": [mapping.get(k, f"missing_{k}") for k in pt["order"]],
                 "missing": [f"missing_{k}" for k in pt["order"] if k not in mapping]}
    save_case(args.out, pieces, truth)
    (Path(args.out) / "capture.json").write_text(json.dumps(
        {"report": rep.__dict__, "pieces": {p.id: p.meta for p in pieces}}, indent=1))
    print(f"captured {rep.n_pieces} pieces (markers: {rep.markers_found}, rectified: {rep.rectified}, "
          f"watershed splits: {rep.split_by_watershed}) -> {args.out}")
    for n in rep.notes:
        print("  note:", n)
    return 0


def cmd_mat(args: argparse.Namespace) -> int:
    from .capture.mat import save_mat_pdf
    print(f"wrote {save_mat_pdf(args.out)}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from .model import load_case
    from .product.audit import audit, audit_dict, report_pdf

    src = Path(args.input)
    if src.is_file():
        import cv2
        from .capture.pipeline import capture
        pieces, _ = capture(cv2.imread(str(src), cv2.IMREAD_COLOR))
        dpi = args.dpi * 2  # capture works at 2x
    else:
        pieces, _ = load_case(src)
        dpi = args.dpi
    original = None
    if args.original:
        original = np.array(Image.open(args.original).convert("RGB"))
    elif src.is_dir() and (src / "original.png").exists():
        original = np.array(Image.open(src / "original.png").convert("RGB"))
    res, img = audit(pieces, dpi=dpi, original=original)
    out = Path(args.out)
    report_pdf(res, img, out)
    out.with_suffix(".json").write_text(json.dumps(audit_dict(res), indent=1))
    print(res.verdict)
    print(f"  {res.level}; {res.n_pieces} pieces, width {res.piece_width_mm} mm, solved in {res.seconds}s")
    if res.text_recovered_pct is not None:
        print(f"  text recovered vs original: {res.text_recovered_pct:.0%}")
    print(f"  {res.advice}")
    print(f"wrote {out}")
    return 0


def cmd_archive(args: argparse.Namespace) -> int:
    from .product.archive import process_archive
    r = process_archive(args.input, args.out)
    ok = sum(b["status"] == "done" for b in r["bins"])
    print(f"{ok}/{len(r['bins'])} bins reconstructed in {r['seconds']:.0f}s -> {args.out}/index.html, archive.pdf")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reassemble", description="Reassemble shredded documents.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="generate synthetic shredded test cases")
    s.add_argument("--out", default="samples")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--kind", choices=["memo", "letter", "invoice", "photo", "handwriting"],
                   help="generate a single strip case of this page type instead of the suite")
    s.add_argument("-n", type=int, default=30, help="number of strips (with --kind)")
    s.add_argument("--noisy", action="store_true", help="apply realistic degradation")
    s.add_argument("--drop", type=int, default=0, help="number of strips to remove")
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("synth-photo", help="simulate a phone photo of strips on the capture mat")
    s.add_argument("--out", default="samples/photo")
    s.add_argument("--kind", default="letter", choices=["memo", "letter", "invoice", "photo", "handwriting"])
    s.add_argument("-n", type=int, default=24)
    s.add_argument("--seed", type=int, default=33)
    s.set_defaults(func=cmd_synth_photo)

    s = sub.add_parser("capture", help="photo/scan of pieces on a dark background -> piece PNGs")
    s.add_argument("photo")
    s.add_argument("--out", default="pieces")
    s.add_argument("--scale", type=float, default=2.0, help="mat-frame resolution (2 = ~200 dpi)")
    s.add_argument("--out-scale", type=float, default=1.0, help="resize output pieces")
    s.set_defaults(func=cmd_capture)

    s = sub.add_parser("audit", help="shredder security audit with a one-page PDF report")
    s.add_argument("input", help="case folder of pieces, or a photo on the capture mat")
    s.add_argument("--out", default="result/audit.pdf")
    s.add_argument("--dpi", type=float, default=100.0, help="scan resolution of the pieces")
    s.add_argument("--original", help="optional image of the original page, for exact % recovered")
    s.set_defaults(func=cmd_audit)

    s = sub.add_parser("archive", help="batch-reconstruct many bins into PDFs with a progress dashboard")
    s.add_argument("input", help="directory of bins (case folders or photos)")
    s.add_argument("--out", default="result/archive")
    s.set_defaults(func=cmd_archive)

    s = sub.add_parser("mat", help="write the printable capture-mat PDF")
    s.add_argument("--out", default="assets/capture_mat.pdf")
    s.set_defaults(func=cmd_mat)

    s = sub.add_parser("solve", help="reconstruct a folder of strip pieces")
    s.add_argument("case", help="case folder (with pieces/ or PNGs directly)")
    s.add_argument("--out", default="result")
    s.add_argument("--methods", default="greedy,beam,atsp")
    s.add_argument("--beam", type=int, default=200, help="beam width")
    s.add_argument("--scorer", default="auto", choices=["auto", "handcrafted", "learned", "combined"])
    s.add_argument("--ocr", action="store_true", help="rerank top candidates with OCR")
    s.add_argument("--lang", default="en", help="word lists for OCR, e.g. en,pcm,yo")
    s.add_argument("--mixed", action="store_true", help="pile may contain several documents")
    s.add_argument("--docs", type=int, default=None, help="number of documents (default: auto)")
    s.add_argument("--shape", default="auto", choices=["auto", "strip", "grid", "torn"])
    s.add_argument("--rows", type=int, default=None, help="cross-cut grid rows (default: guess)")
    s.add_argument("--cols", type=int, default=None, help="cross-cut grid cols (default: guess)")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(func=cmd_solve)

    s = sub.add_parser("bench", help="benchmark strip reconstruction on fresh synthetic pages")
    s.add_argument("--suite", default="strips", choices=["strips", "mixed", "double", "ocr", "crosscut", "torn", "scorers", "capture"])
    s.add_argument("-n", type=int, default=30)
    s.add_argument("--seeds", type=int, default=5)
    s.add_argument("--seed", type=int, default=100)
    s.add_argument("--scorer", default="auto", choices=["auto", "handcrafted", "learned", "combined"])
    s.add_argument("--ocr", action="store_true")
    s.set_defaults(func=cmd_bench)

    s = sub.add_parser("train", help="train the learned edge matcher on synthetic data")
    s.add_argument("--out", default="models/edge_cnn.pt")
    s.add_argument("--pages", type=int, default=80)
    s.add_argument("--epochs", type=int, default=6)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("finetune", help="fine-tune the matcher on corrections exported from the editor")
    s.add_argument("corrections", help="JSON list of {case, left, right, label}")
    s.add_argument("--init", default=None)
    s.add_argument("--out", default="models/edge_cnn_finetuned.pt")
    s.set_defaults(func=cmd_finetune)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
