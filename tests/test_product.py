import json

import numpy as np

from reassemble.model import save_case
from reassemble.product.archive import process_archive
from reassemble.product.audit import audit, classify, report_pdf
from reassemble.synth.pages import render_page
from reassemble.synth.shredders import crosscut_shred, strip_shred


def test_classify_verdicts():
    assert classify("strip", 6, 1600, 0.9)[0].startswith("UNSAFE")
    assert "strip-cut" in classify("strip", 6, 1600, 0.05)[1]
    assert classify("grid", 4, 40, 0.02)[0].startswith("RESISTANT")
    assert "P-4" in classify("grid", 4, 40, 0.02)[1]


def test_audit_strip_and_pdf(tmp_path):
    page, _ = render_page("letter", 8)
    pieces, _ = strip_shred(page, 20, seed=8)
    res, img = audit(pieces, original=page)
    assert res.shape == "strip" and res.n_pieces == 20
    assert res.piece_width_mm > 5
    if res.text_recovered_pct is not None:  # tesseract available
        assert res.text_recovered_pct > 0.6 and res.verdict.startswith("UNSAFE")
    pdf = report_pdf(res, img, tmp_path / "r.pdf")
    assert pdf.read_bytes()[:4] == b"%PDF"


def test_archive_mode(tmp_path):
    inp = tmp_path / "in"
    page, _ = render_page("memo", 9)
    pcs, t = strip_shred(page, 12, seed=9)
    save_case(inp / "bin_a", pcs, t)
    pcs, t = crosscut_shred(page, 3, 2, seed=9)
    save_case(inp / "bin_b", pcs, t)
    (inp / "junk.png").write_bytes(b"not an image")
    r = process_archive(inp, tmp_path / "out", verbose=False)
    status = {b["bin"]: b["status"] for b in r["bins"]}
    assert status["bin_a"] == "done" and status["bin_b"] == "done" and status["junk.png"] == "error"
    assert (tmp_path / "out" / "archive.pdf").read_bytes()[:4] == b"%PDF"
    assert "bin_a" in (tmp_path / "out" / "index.html").read_text()
    assert json.loads((tmp_path / "out" / "status.json").read_text())["done"] == 3
