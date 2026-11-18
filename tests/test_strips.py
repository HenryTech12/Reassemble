import numpy as np
import pytest

from reassemble.features.edges import extract
from reassemble.features.normalize import deskew, mask_angle
from reassemble.match.handcrafted import handcrafted_cost, normalize_costs
from reassemble.metrics import direct_accuracy, gap_detection, neighbour_accuracy
from reassemble.model import Piece, to_rgba
from reassemble.pipeline import SolveOptions, prepare, solve_prepared
from reassemble.render.compose import render_strips
from reassemble.solve.strips import (beam_search, build_atsp, greedy_chain, or_opt_2opt,
                                     order_to_tour, solve_strips, tour_cost, tour_to_order)
from reassemble.synth.pages import render_page
from reassemble.synth.shredders import Realism, rotate_rgba, strip_shred


@pytest.fixture(scope="module")
def letter():
    return render_page("letter", 42)[0]


def chain_matrix(n, rng):
    """Cost matrix where the identity order is clearly optimal."""
    C = rng.uniform(5, 10, (n, n))
    for i in range(n - 1):
        C[i, i + 1] = rng.uniform(0, 1)
    np.fill_diagonal(C, 1e6)
    return C


@pytest.mark.parametrize("method", ["greedy", "beam", "atsp"])
def test_solvers_recover_planted_chain(method):
    rng = np.random.default_rng(0)
    C = chain_matrix(15, rng)
    res = solve_strips(C, methods=(method,))
    assert res[method].order == list(range(15))


def test_locked_pairs_are_respected():
    rng = np.random.default_rng(1)
    C = chain_matrix(8, rng)
    res = solve_strips(C, locked=[(5, 2)])
    o = res["best"].order
    assert o[o.index(5) + 1] == 2


def test_or_opt_never_worsens():
    rng = np.random.default_rng(2)
    M = rng.uniform(0, 1, (12, 12))
    np.fill_diagonal(M, 1e6)
    t = list(rng.permutation(12))
    t2 = or_opt_2opt(M, t)
    assert sorted(t2) == list(range(12))
    assert tour_cost(M, t2) <= tour_cost(M, t) + 1e-9


def test_tour_order_roundtrip():
    assert tour_to_order(order_to_tour([2, 0, 1], 3), 3) == [2, 0, 1]


def test_metrics():
    truth = ["a", "b", "c", "d"]
    assert neighbour_accuracy(truth, truth) == 1.0
    assert neighbour_accuracy(["b", "c", "d", "a"], truth) == pytest.approx(2 / 3)
    assert direct_accuracy(["b", "c", "d", "a"], truth) == 0.0
    g = gap_detection(["a", None, "c", "d"], ["a", "b", "c", "d"], ["b"])
    assert g["recall"] == 1.0 and g["precision"] == 1.0


def test_deskew_recovers_angle(letter):
    strip = to_rgba(letter[:, 300:330])
    rot = rotate_rgba(strip, 0.5)
    assert mask_angle(rot[..., 3] > 127) == pytest.approx(0.5, abs=0.05)
    out, ang = deskew(rot)
    assert abs(ang - 0.5) < 0.05 and out.shape[1] <= 32


def test_true_neighbour_is_cheap(letter):
    pieces, truth = strip_shred(letter, 20, seed=3)
    prep = prepare(pieces)
    D = handcrafted_cost(prep.feats)["total"]
    idx = {p: i for i, p in enumerate(prep.ids)}
    ranks = [int((D[idx[a]] < D[idx[a], idx[b]]).sum())
             for a, b in zip(truth["order"], truth["order"][1:])
             if a not in prep.blank and b not in prep.blank]
    assert np.mean(np.array(ranks) == 0) > 0.8
    N = normalize_costs(D)
    assert N.shape == D.shape


@pytest.mark.parametrize("noisy", [False, True])
def test_end_to_end_strip_reconstruction(letter, noisy):
    pieces, truth = strip_shred(letter, 30, Realism.noisy() if noisy else Realism(), seed=5)
    prep = prepare(pieces)
    rec = solve_prepared(prep)
    acc = neighbour_accuracy(rec.order, truth["order"], ignore=prep.blank)
    assert acc >= (0.95 if not noisy else 0.85)
    assert len(rec.joins) == len(prep.ids) - 1
    assert all(0 <= j["confidence"] <= 1 for j in rec.joins)
    img = render_strips({p.id: p for p in prep.pieces}, rec, heatmap=True)
    assert img.size[1] == max(p.height for p in prep.pieces)


def test_missing_strips_flagged_or_harmless(letter):
    pieces, truth = strip_shred(letter, 30, Realism(drop=2), seed=7)
    rec = solve_prepared(prepare(pieces))
    assert len(rec.piece_order) == 28
    assert rec.to_dict()["gaps"] == rec.gaps


def test_handles_single_and_two_pieces(letter):
    pieces, _ = strip_shred(letter, 2, seed=1)
    rec = solve_prepared(prepare(pieces[:1]))
    assert rec.piece_order == [pieces[0].id]
    rec = solve_prepared(prepare(pieces))
    assert sorted(rec.piece_order) == sorted(p.id for p in pieces)
