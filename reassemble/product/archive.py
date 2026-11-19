"""Archive mode: batch-reconstruct many bins and export the pages as PDF.

Input is a directory where every entry is one *bin*: either a case folder of piece PNGs or
a single photo/scan of pieces on the capture mat.  Each bin may contain several
documents (mixed bins are separated automatically).  Output:

* ``<out>/pages/<bin>_doc<k>.png`` for every recovered page,
* ``<out>/<bin>.pdf`` per bin and ``<out>/archive.pdf`` with everything,
* ``<out>/index.html``: a self-refreshing progress dashboard, rewritten after every bin,
  plus ``status.json`` with the same data.
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _bins(in_dir: Path) -> list[Path]:
    out = []
    for p in sorted(in_dir.iterdir()):
        if p.is_dir() and (any(p.glob("*.png")) or (p / "pieces").is_dir()):
            out.append(p)
        elif p.suffix.lower() in IMAGE_EXT:
            out.append(p)
    return out


def _load_bin(path: Path):
    from ..model import load_case
    if path.is_dir():
        return load_case(path)[0]
    import cv2
    from ..capture.pipeline import capture
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return capture(img)[0]


def _dashboard(out: Path, rows: list[dict], started: float, total: int) -> None:
    done = sum(r["status"] in ("done", "error") for r in rows)
    pct = 100 * done / max(total, 1)
    body = "".join(
        f"<tr><td>{html.escape(r['bin'])}</td><td class='{r['status']}'>{r['status']}</td>"
        f"<td>{r.get('pieces', '')}</td><td>{r.get('documents', '')}</td>"
        f"<td>{'' if r.get('confidence') is None else f'{r['confidence']:.0%}'}</td>"
        f"<td>{'' if r.get('seconds') is None else f'{r['seconds']:.1f}s'}</td>"
        f"<td>{''.join(f'<a href=\"pages/{html.escape(p)}\"><img src=\"pages/{html.escape(p)}\"></a>' for p in r.get('pages', []))}</td>"
        f"<td class='err'>{html.escape(r.get('error') or '')}</td></tr>"
        for r in rows)
    refresh = "<meta http-equiv='refresh' content='2'>" if done < total else ""
    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>Reassemble archive</title>{refresh}
<meta name='viewport' content='width=device-width, initial-scale=1'>
<style>
:root{{--bg:#0b0e13;--panel:#121721;--line:#232b38;--text:#d7dee9;--dim:#8592a6;--ok:#22c55e;--bad:#ef4444;--acc:#38bdf8}}
body{{background:var(--bg);color:var(--text);font:14px/1.4 system-ui,sans-serif;margin:0;padding:24px}}
h1{{font:600 18px ui-monospace,monospace;margin:0 0 4px}} .dim{{color:var(--dim)}}
.bar{{height:10px;background:var(--panel);border:1px solid var(--line);border-radius:6px;overflow:hidden;margin:12px 0 20px;max-width:640px}}
.bar>div{{height:100%;background:var(--acc);width:{pct:.1f}%}}
table{{border-collapse:collapse;width:100%}} td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:middle}}
th{{color:var(--dim);font-weight:500;font-size:12px;text-transform:uppercase;letter-spacing:.06em}}
img{{height:64px;margin-right:4px;border:1px solid var(--line);background:#fff}}
.done{{color:var(--ok)}} .error,.err{{color:var(--bad)}} .running{{color:var(--acc)}}
</style></head><body>
<h1>REASSEMBLE · archive</h1>
<div class='dim'>{done}/{total} bins · {time.time() - started:.0f}s elapsed · <a style='color:var(--acc)' href='archive.pdf'>archive.pdf</a></div>
<div class='bar'><div></div></div>
<table><tr><th>bin</th><th>status</th><th>pieces</th><th>docs</th><th>confidence</th><th>time</th><th>pages</th><th></th></tr>{body}</table>
</body></html>"""
    (out / "index.html").write_text(page)
    (out / "status.json").write_text(json.dumps({"done": done, "total": total, "bins": rows}, indent=1))


def process_archive(in_dir: str | Path, out_dir: str | Path, verbose: bool = True) -> dict:
    from ..model import Reconstruction
    from ..pipeline import SolveOptions, detect_shape, prepare, solve_mixed, solve_grid_pieces
    from ..render.compose import render_grid, render_strips

    in_dir, out = Path(in_dir), Path(out_dir)
    (out / "pages").mkdir(parents=True, exist_ok=True)
    bins = _bins(in_dir)
    rows = [{"bin": b.name, "status": "queued"} for b in bins]
    started = time.time()
    all_pages: list[Image.Image] = []
    _dashboard(out, rows, started, len(bins))
    for b, row in zip(bins, rows):
        row["status"] = "running"
        _dashboard(out, rows, started, len(bins))
        t0 = time.time()
        try:
            pieces = _load_bin(b)
            row["pieces"] = len(pieces)
            pages, confs = [], []
            if detect_shape(pieces) == "grid":
                rec = solve_grid_pieces(pieces)
                pages.append(render_grid({p.id: p for p in pieces}, rec.grid, background=(255, 255, 255, 255)))
                confs += [j["confidence"] for j in rec.joins]
            else:
                prep = prepare(pieces)
                rec = solve_mixed(prep, SolveOptions())
                by_id = {p.id: p for p in prep.pieces}
                for d in rec.documents:
                    r = Reconstruction.from_dict(d)
                    pages.append(render_strips(by_id, r, background=(255, 255, 255, 255)))
                    confs += [j["confidence"] for j in d["joins"]]
            names = []
            for k, pg in enumerate(pages):
                name = f"{b.stem}_doc{k}.png"
                pg.convert("RGB").save(out / "pages" / name)
                names.append(name)
            rgb = [pg.convert("RGB") for pg in pages]
            rgb[0].save(out / f"{b.stem}.pdf", "PDF", save_all=True, append_images=rgb[1:], resolution=100.0)
            all_pages += rgb
            row.update(status="done", documents=len(pages), pages=names,
                       confidence=float(np.mean(confs)) if confs else None, seconds=time.time() - t0)
        except Exception as e:  # keep going: one bad bin must not stop the archive
            row.update(status="error", error=f"{type(e).__name__}: {e}", seconds=time.time() - t0)
        if verbose:
            print(f"[{sum(r['status'] in ('done', 'error') for r in rows)}/{len(bins)}] {b.name}: {row['status']}"
                  + (f", {row.get('documents')} doc(s)" if row["status"] == "done" else f" ({row.get('error')})"))
        _dashboard(out, rows, started, len(bins))
    if all_pages:
        all_pages[0].save(out / "archive.pdf", "PDF", save_all=True, append_images=all_pages[1:], resolution=100.0)
    _dashboard(out, rows, started, len(bins))
    return {"bins": rows, "seconds": time.time() - started}
