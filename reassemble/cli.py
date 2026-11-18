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
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
