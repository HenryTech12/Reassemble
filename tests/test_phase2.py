import numpy as np
import pytest
from PIL import Image, ImageDraw

from reassemble.cluster.documents import cluster_documents, purity
from reassemble.metrics import evaluate_mixed
from reassemble.pipeline import SolveOptions, prepare, solve_mixed, solve_prepared
from reassemble.synth.pages import load_font, render_page
from reassemble.synth.shredders import Realism, mixed_bin, strip_shred
from reassemble.verify.ocr import is_word, load_words, readability, suspicious_joins, tesseract_available

needs_tesseract = pytest.mark.skipif(not tesseract_available(), reason="tesseract not installed")


def test_word_lists_load_and_merge():
    en = load_words("en")
    assert "budget" in en and len(en) > 10000
    both = load_words("en,pcm")
    assert "wahala" in both and "wahala" not in en
    assert is_word("Reports", en)


def test_readability_prefers_words():
    good = [{"text": "The committee approved the budget", "conf": 90}]
    bad = [{"text": "Thx cmmttee aprvd thq bdgxt", "conf": 40}]
    assert readability(good)["score"] > readability(bad)["score"] + 0.3


def test_suspicious_joins_counts_nonwords_across_seam():
    words = [{"text": "regiohal", "conf": 50, "x": 100, "y": 0, "w": 60, "h": 10},
             {"text": "office", "conf": 90, "x": 200, "y": 0, "w": 40, "h": 10}]
    s = suspicious_joins(words, {"a|b": 130, "c|d": 220})
    assert s == {"a|b": 1, "c|d": 0}


@needs_tesseract
def test_ocr_reads_rendered_text():
    from reassemble.verify.ocr import extract_text
    img = Image.new("RGB", (500, 60), "white")
    ImageDraw.Draw(img).text((10, 10), "The committee approved the budget",
                             font=load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24), fill="black")
    r = extract_text(img)
    assert r["readability"]["dict_fraction"] > 0.7


@needs_tesseract
def test_ocr_rerank_runs_and_keeps_good_solution():
    page, _ = render_page("memo", 101)
    pieces, truth = strip_shred(page, 20, seed=1)
    rec = solve_prepared(prepare(pieces), SolveOptions(ocr_rerank=True, ocr_candidates=3))
    assert "ocr_rerank" in rec.extra
    assert all("ocr_suspect" in j for j in rec.joins)


def test_back_side_cost_is_transposed():
    """With a blank front, only the (mirrored) back side can order the strips."""
    blank = np.full((400, 300, 3), 240, np.uint8)
    back, _ = render_page("photo", 3)
    back = back[:400, 100:400]
    pieces, truth = strip_shred(blank, 10, seed=2, back=back)
    prep = prepare(pieces)
    rec = solve_prepared(prep, SolveOptions(use_back=True, park_blanks=False))
    from reassemble.metrics import neighbour_accuracy
    assert neighbour_accuracy(rec.order, truth["order"]) >= 0.8


def test_mixed_bin_separates_three_documents():
    pages = [render_page(k, 900 + i)[0] for i, k in enumerate(["memo", "invoice", "handwriting"])]
    pieces, truth = mixed_bin(pages, 20, Realism.noisy(), seed=3)
    prep = prepare(pieces)
    labels, info = cluster_documents(prep.feats)
    assert info["k"] == 3
    assert purity(labels, [truth["doc_of"][i] for i in prep.ids]) >= 0.85
    rec = solve_mixed(prep, SolveOptions())
    m = evaluate_mixed(rec.documents, truth, prep.blank)
    assert m["purity"] >= 0.85 and len(rec.documents) == 3


def test_single_document_stays_single():
    page, _ = render_page("letter", 5)
    pieces, _ = strip_shred(page, 30, Realism.noisy(), seed=5)
    labels, info = cluster_documents(prepare(pieces).feats)
    assert info["k"] == 1


def test_purity():
    assert purity(np.array([0, 0, 1, 1]), [0, 0, 1, 1]) == 1.0
    assert purity(np.array([0, 0, 0, 0]), [0, 0, 1, 1]) == 0.5
