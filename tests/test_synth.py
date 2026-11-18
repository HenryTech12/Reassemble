import numpy as np
import pytest

from reassemble.model import load_case, save_case
from reassemble.synth.generate import generate_suite
from reassemble.synth.pages import RENDERERS, render_page
from reassemble.synth.shredders import (Realism, crosscut_shred, mixed_bin, strip_shred,
                                        torn_shred)


@pytest.fixture(scope="module")
def page():
    return render_page("memo", 3)[0]


@pytest.mark.parametrize("kind", list(RENDERERS))
def test_pages_render_with_ink(kind):
    img, meta = render_page(kind, 5)
    assert img.shape == (1100, 850, 3)
    assert (img.mean(-1) < 120).mean() > 0.01  # some dark ink on the page
    assert meta["type"] == kind


def test_render_is_deterministic():
    a, _ = render_page("letter", 7)
    b, _ = render_page("letter", 7)
    assert np.array_equal(a, b)


def test_strip_shred_reassembles_exactly(page):
    pieces, truth = strip_shred(page, 25, seed=1)
    assert len(pieces) == 25 and truth["missing"] == []
    by_id = {p.id: p for p in pieces}
    rebuilt = np.concatenate([by_id[i].image[..., :3] for i in truth["order"]], axis=1)
    assert np.array_equal(rebuilt, page)
    # pieces are shuffled and ids don't reveal position
    assert [p.id for p in pieces] != truth["order"]


def test_noisy_strips_have_alpha_fray_and_drop(page):
    pieces, truth = strip_shred(page, 30, Realism.noisy(drop=3), seed=2)
    assert len(pieces) == 27 and len(truth["missing"]) == 3
    assert all(m not in {p.id for p in pieces} for m in truth["missing"])
    assert any((p.image[..., 3] == 0).any() for p in pieces)


def test_double_sided_back_is_mirrored(page):
    back = render_page("letter", 4)[0]
    pieces, truth = strip_shred(page, 10, seed=3, back=back)
    by_id = {p.id: p for p in pieces}
    # back strips in *reverse* true order rebuild the back page
    rebuilt = np.concatenate([by_id[i].back[..., :3] for i in truth["order"][::-1]], axis=1)
    assert np.array_equal(rebuilt, back)


def test_crosscut_grid(page):
    pieces, truth = crosscut_shred(page, 4, 5, seed=1)
    by_id = {p.id: p for p in pieces}
    rows = [np.concatenate([by_id[i].image[..., :3] for i in r], 1) for r in truth["grid"]]
    assert np.array_equal(np.concatenate(rows, 0), page)


def test_torn_pieces_cover_page(page):
    pieces, truth = torn_shred(page, 2, 3, seed=1)
    cover = np.zeros(page.shape[:2], int)
    for p in pieces:
        x, y = truth["offsets"][p.id]
        h, w = p.image.shape[:2]
        cover[y:y + h, x:x + w] += p.image[..., 3] > 0
    assert (cover == 1).all()  # every pixel belongs to exactly one piece


def test_mixed_bin_labels(page):
    other = render_page("invoice", 9)[0]
    pieces, truth = mixed_bin([page, other], 10, seed=1)
    assert len(pieces) == 20
    assert set(truth["doc_of"].values()) == {0, 1}
    assert {p.id for p in pieces} == set(truth["doc_of"])


def test_case_roundtrip(tmp_path, page):
    pieces, truth = strip_shred(page, 5, seed=1, back=page)
    save_case(tmp_path / "c", pieces, truth)
    loaded, t2 = load_case(tmp_path / "c")
    assert t2 == truth
    assert {p.id for p in loaded} == {p.id for p in pieces}
    assert all(p.back is not None for p in loaded)


def test_suite_covers_every_case(tmp_path):
    made = {f.name for f in generate_suite(tmp_path)}
    for name in ["memo_30", "memo_30_noisy", "handwriting_30", "letter_30_missing2",
                 "crosscut_6x8", "torn_2x4", "double_30_noisy", "mixed_3", "mixed_5"]:
        assert name in made
