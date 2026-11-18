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
    from .pipeline import SolveOptions, prepare, solve_prepared
    from .render.compose import render_strips

    pieces, truth = load_case(args.case)
    if not pieces:
        print(f"no pieces found in {args.case}")
        return 1
    opts = SolveOptions(methods=tuple(args.methods.split(",")), beam_width=args.beam,
                        scorer=args.scorer, ocr_rerank=args.ocr)
    prep = prepare(pieces)
    rec = solve_prepared(prep, opts, progress=(lambda e: print("  ..", e["stage"])) if args.verbose else None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by_id = {p.id: p for p in prep.pieces}
    render_strips(by_id, rec).save(out / "reconstruction.png")
    render_strips(by_id, rec, heatmap=True).save(out / "heatmap.png")
    result = rec.to_dict()
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


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import print_table, strip_benchmark
    from .pipeline import SolveOptions

    seeds = range(args.seed, args.seed + args.seeds)
    for noisy, drop, label in [(False, 0, "clean"), (True, 0, "noisy"), (True, 2, "noisy, 2 missing")]:
        opts = SolveOptions(scorer=args.scorer, ocr_rerank=args.ocr)
        res = strip_benchmark(n=args.n, noisy=noisy, seeds=seeds, opts=opts, drop=drop)
        print(f"\n### {args.n} strips, {label} ({len(seeds)} pages per type, scorer={args.scorer})")
        print_table(res)
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

    s = sub.add_parser("solve", help="reconstruct a folder of strip pieces")
    s.add_argument("case", help="case folder (with pieces/ or PNGs directly)")
    s.add_argument("--out", default="result")
    s.add_argument("--methods", default="greedy,beam,atsp")
    s.add_argument("--beam", type=int, default=200, help="beam width")
    s.add_argument("--scorer", default="handcrafted", choices=["handcrafted", "learned", "combined"])
    s.add_argument("--ocr", action="store_true", help="rerank top candidates with OCR")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(func=cmd_solve)

    s = sub.add_parser("bench", help="benchmark strip reconstruction on fresh synthetic pages")
    s.add_argument("-n", type=int, default=30)
    s.add_argument("--seeds", type=int, default=5)
    s.add_argument("--seed", type=int, default=100)
    s.add_argument("--scorer", default="handcrafted", choices=["handcrafted", "learned", "combined"])
    s.add_argument("--ocr", action="store_true")
    s.set_defaults(func=cmd_bench)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
