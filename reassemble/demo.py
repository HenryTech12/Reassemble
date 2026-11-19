"""``make demo``: generate a sample, solve it, write the results and open them."""

from __future__ import annotations

import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path


def main(out: str = "result/demo") -> int:
    from .model import load_case, save_case
    from .pipeline import SolveOptions, prepare, solve_prepared
    from .metrics import evaluate_strips
    from .render.compose import render_strips
    from .render.replay import save_replay_gif
    from .synth.pages import render_page
    from .synth.shredders import Realism, strip_shred

    out_p = Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    page, _ = render_page("letter", 7)
    pieces, truth = strip_shred(page, 30, Realism.noisy(), seed=7)
    save_case(out_p / "case", pieces, truth)
    print(f"1/3 shredded a letter into 30 noisy strips -> {out_p / 'case'}")
    prep = prepare(load_case(out_p / "case")[0])
    rec = solve_prepared(prep, SolveOptions())
    m = evaluate_strips(rec.order, truth, prep.blank)
    print(f"2/3 solved with the {rec.extra.get('scorer')} scorer in {rec.runtime_s:.1f}s: "
          f"neighbour accuracy {m['neighbour']:.3f} (content {m['neighbour_content']:.3f})")
    by_id = {p.id: p for p in prep.pieces}
    render_strips(by_id, rec).save(out_p / "reconstruction.png")
    render_strips(by_id, rec, heatmap=True).save(out_p / "heatmap.png")
    save_replay_gif(by_id, rec, out_p / "replay.gif", scale=0.3)
    from PIL import Image
    Image.fromarray(page).save(out_p / "original.png")
    html = out_p / "index.html"
    html.write_text(f"""<!doctype html><meta charset=utf-8><title>Reassemble demo</title>
<style>body{{background:#0b0e13;color:#d7dee9;font:14px system-ui;padding:20px}}img{{height:520px;margin:8px;background:#fff}}</style>
<h2>Reassemble demo: neighbour accuracy {m['neighbour']:.3f}, content {m['neighbour_content']:.3f}</h2>
<img src=replay.gif><img src=reconstruction.png><img src=heatmap.png><img src=original.png>""")
    print(f"3/3 wrote {html}")
    opener = shutil.which("xdg-open") or shutil.which("open")
    if opener and sys.stdout.isatty():
        subprocess.Popen([opener, str(html)])
    else:
        webbrowser.open(html.resolve().as_uri()) if sys.stdout.isatty() else None
    return 0


if __name__ == "__main__":
    sys.exit(main())
