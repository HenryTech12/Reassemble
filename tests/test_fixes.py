import numpy as np
import pytest

from reassemble.model import Piece
from reassemble.synth.pages import render_page
from reassemble.synth.shredders import rotate_rgba, torn_shred
from reassemble.verify.lm import CharLM, default_lm, normalise


def test_char_lm_trains_and_prefers_seen_text(tmp_path):
    lm = CharLM.train("the quick brown fox jumps over the lazy dog. " * 50, min_count=1)
    assert lm.logprob_per_char("the lazy dog") > lm.logprob_per_char("zqx vvk jjp")
    lm.save(tmp_path / "m.json.gz")
    assert CharLM.load(tmp_path / "m.json.gz").logprob_per_char("the fox") == pytest.approx(lm.logprob_per_char("the fox"))
    assert normalise("Hello,  World!\n42") == "hello, world"


def test_shipped_english_lm_prefers_clean_text():
    lm = default_lm()
    if lm is None:
        pytest.skip("no shipped language model")
    good = "The committee has reviewed the quarterly budget before the end of the month."
    split = "The commit tee has rev iewed the quart erly bud get bef ore the end."
    assert lm.logprob_per_char(good) > lm.logprob_per_char(split) + 1.0


def _rotated(pieces, angles):
    out = []
    for p, a in zip(pieces, angles):
        im = rotate_rgba(p.image, a)
        im[..., 3] = np.where(im[..., 3] > 127, 255, 0)
        ys, xs = np.nonzero(im[..., 3])
        out.append(Piece(p.id, im[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()))
    return out


def test_text_angle_recovers_rotation():
    from reassemble.solve.torn import text_angle
    page, _ = render_page("letter", 5)
    pieces, _ = torn_shred(page, 2, 2, seed=5)
    for a in (-2.0, 1.2):
        est, strength = text_angle(_rotated(pieces[:1], [a])[0])
        assert abs(est - a) < 0.5 and strength > 1.05


def test_rotated_torn_pieces_are_deskewed():
    from reassemble.pipeline import solve_torn_pieces
    page, _ = render_page("memo", 6)
    pieces, truth = torn_shred(page, 2, 2, seed=6)
    rp = _rotated(pieces, [2.5, -2.0, 1.5, -2.5])
    rec = solve_torn_pieces(rp)
    assert rec.extra["rotations"], "rotation should have been detected"
    for p, a in zip(rp, [2.5, -2.0, 1.5, -2.5]):
        assert abs(rec.extra["rotations"][p.id] - a) < 0.6


def test_straight_page_edge_is_not_a_tear():
    from reassemble.solve.torn import _straight
    ys = np.arange(300)
    has = np.ones(300, bool)
    assert _straight(np.round(10 + 0.02 * ys).astype(float), has)
    wiggle = 10 + 6 * np.sin(ys / 25.0)
    assert not _straight(np.round(wiggle).astype(float), has)


def test_gap_features_and_model():
    from reassemble.solve.joins import FEATURES, join_features, load_gap_model
    rng = np.random.default_rng(0)
    D = rng.uniform(0, 1, (6, 6))
    np.fill_diagonal(D, 1e6)
    f = join_features([0, 1, 2, 3, 4, 5], D, D, np.full(5, 0.5), np.zeros(6), np.zeros(6), 0.5)
    assert f.shape == (5, len(FEATURES))
    m = load_gap_model("handcrafted")
    if m is not None:
        p = m["clf"].predict_proba(f)[:, 1]
        assert p.shape == (5,) and ((0 <= p) & (p <= 1)).all()
