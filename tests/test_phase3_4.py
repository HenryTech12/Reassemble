import numpy as np
import pytest

from reassemble.metrics import neighbour_accuracy
from reassemble.pipeline import (SolveOptions, detect_shape, prepare, resolve_scorer,
                                 solve_grid_pieces, solve_prepared, solve_torn_pieces)
from reassemble.solve.grid import evaluate_grid, greedy_place, guess_grid_shape, rows_then_stack
from reassemble.solve.torn import assemble, evaluate_torn, pair_match
from reassemble.synth.pages import render_page
from reassemble.synth.shredders import Realism, crosscut_shred, strip_shred, torn_shred

torch = pytest.importorskip("torch", reason="learned matcher needs torch")


@pytest.fixture(scope="module")
def memo():
    return render_page("memo", 77)[0]


def test_learned_matcher_shapes_and_training_smoke(tmp_path, memo):
    from reassemble.match.learned import (K, PATCH_H, build_model, edge_columns, learned_cost,
                                          make_dataset, seam_patches, train)
    pieces, _ = strip_shred(memo, 6, seed=1)
    a, b = edge_columns(pieces[0]), edge_columns(pieces[1])
    p, info = seam_patches(a, b)
    assert p.shape[1:] == (3, PATCH_H, 2 * K) and len(info) == len(p)
    assert build_model()(torch.zeros(2, 3, PATCH_H, 2 * K)).shape == (2, 1)
    data = make_dataset(2, None, seed=5)
    assert 0 < data.y.mean() < 1
    stats = train(tmp_path / "m.pt", epochs=1, data=data, val=data, verbose=False)
    assert (tmp_path / "m.pt").exists() and stats["history"]
    D = learned_cost(pieces, weights=tmp_path / "m.pt")
    assert D.shape == (6, 6) and np.all(np.diag(D) >= 1e5)


def test_trained_weights_solve_noisy_page(memo):
    from reassemble.match.learned import DEFAULT_WEIGHTS
    if not DEFAULT_WEIGHTS.exists():
        pytest.skip("no trained weights")
    assert resolve_scorer("learned") == "learned"
    pieces, truth = strip_shred(memo, 30, Realism.noisy(), seed=4)
    prep = prepare(pieces)
    rec = solve_prepared(prep, SolveOptions(scorer="learned"))
    assert neighbour_accuracy(rec.order, truth["order"], ignore=prep.blank) >= 0.9


def test_finetune_from_corrections(tmp_path, memo):
    from reassemble.match.learned import DEFAULT_WEIGHTS, finetune_from_corrections
    from reassemble.model import save_case
    if not DEFAULT_WEIGHTS.exists():
        pytest.skip("no trained weights")
    pieces, truth = strip_shred(memo, 10, seed=2)
    save_case(tmp_path / "case", pieces, truth)
    o = truth["order"]
    corr = [{"case": str(tmp_path / "case"), "left": o[3], "right": o[4], "label": 1},
            {"case": str(tmp_path / "case"), "left": o[3], "right": o[6], "label": 0}]
    import reassemble.match.learned as L
    orig = L.make_dataset
    L.make_dataset = lambda *a, **k: orig(1, None, seed=1)
    try:
        stats = finetune_from_corrections(corr, out=tmp_path / "ft.pt", epochs=1)
    finally:
        L.make_dataset = orig
    assert (tmp_path / "ft.pt").exists() and stats["train_patches"] > 0


def test_grid_solver_planted():
    R, C = 3, 4
    n = R * C
    rng = np.random.default_rng(1)
    G = rng.permutation(n).reshape(R, C)
    LR, TB = rng.uniform(1, 2, (n, n)), rng.uniform(1, 2, (n, n))
    for r in range(R):
        for c in range(C):
            if c + 1 < C:
                LR[G[r, c], G[r, c + 1]] = 0.1
            if r + 1 < R:
                TB[G[r, c], G[r + 1, c]] = 0.1
    np.fill_diagonal(LR, np.inf)
    np.fill_diagonal(TB, np.inf)
    assert (greedy_place(LR, TB, R, C) == G).all()
    border = {k: np.zeros(n) for k in ("left", "right", "top", "bottom")}
    assert (rows_then_stack(LR, TB, R, C, border) == G).all()


def test_grid_shape_guess():
    assert guess_grid_shape(12, 283, 275) == (4, 3)


def test_crosscut_small_grid_solved(memo):
    pieces, truth = crosscut_shred(memo, 3, 2, seed=1)
    assert detect_shape(pieces) == "grid"
    rec = solve_grid_pieces(pieces, 3, 2)
    assert evaluate_grid(rec.grid, truth["grid"])["neighbour"] >= 0.7


def test_torn_pair_offset_and_assembly(memo):
    pieces, truth = torn_shred(memo, 2, 3, seed=3)
    assert detect_shape(pieces) == "torn"
    by = {p.id: p for p in pieces}
    a, b = truth["grid"][0][0], truth["grid"][0][1]
    cost, dx, dy = pair_match(by[a], by[b])
    true = np.subtract(truth["offsets"][b], truth["offsets"][a])
    assert abs(dx - true[0]) <= 2 and abs(dy - true[1]) <= 2
    res = assemble(pieces)
    assert evaluate_torn(pieces, res, truth)["neighbour"] == 1.0
    rec = solve_torn_pieces(pieces)
    assert len(rec.extra["positions"]) == 6


def test_handwriting_downweights_baselines():
    from reassemble.features.edges import extract
    hw, _ = render_page("handwriting", 3)
    pr, _ = render_page("letter", 3)
    f_hw = np.mean([extract(p).line_regularity for p in strip_shred(hw, 10, seed=1)[0]])
    f_pr = np.mean([extract(p).line_regularity for p in strip_shred(pr, 10, seed=1)[0]])
    assert f_pr >= f_hw - 0.05
