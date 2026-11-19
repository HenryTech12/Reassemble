import numpy as np
import pytest

from reassemble.capture.pipeline import CaptureReport, capture, repair_strip_edges, split_touching
from reassemble.metrics import neighbour_accuracy
from reassemble.pipeline import SolveOptions, prepare, solve_prepared
from reassemble.synth.pages import render_page
from reassemble.synth.photo import compose_photo
from reassemble.synth.shredders import Realism, strip_shred


def test_capture_end_to_end_on_simulated_photo():
    page, _ = render_page("memo", 34)
    pieces, truth = strip_shred(page, 16, Realism(noise_sigma=2), seed=34)
    photo, pt = compose_photo(pieces, seed=34)
    cap, rep = capture(photo)
    assert rep.rectified and rep.markers_found >= 3
    assert rep.n_pieces == 16
    # every captured piece maps to a distinct true strip, with the right size
    cen = pt["centres_mat"]
    mp = {p.id: min(cen, key=lambda k: np.linalg.norm(np.array(cen[k]) - np.array(p.meta["centre"]))) for p in cap}
    assert len(set(mp.values())) == 16
    assert all(abs(p.height - 2 * 1100) < 20 for p in cap)
    prep = prepare(cap)
    rec = solve_prepared(prep, SolveOptions(scorer="handcrafted"))
    order = [mp[x] if x else None for x in rec.order]
    acc = neighbour_accuracy(order, truth["order"], ignore={mp[b] for b in prep.blank})
    assert acc >= 0.7


def test_watershed_splits_touching_pieces():
    mask = np.zeros((300, 200), np.uint8)
    mask[20:280, 40:80] = 255
    mask[20:280, 82:122] = 255
    mask[150:153, 80:82] = 255  # a thin bridge joins the two strips
    mask[20:280, 150:190] = 255
    rep = CaptureReport()
    lab = split_touching(mask, rep, 100)
    assert len([k for k in np.unique(lab) if k > 0]) == 3
    assert rep.split_by_watershed == 1


def test_repair_restores_dented_edges():
    rgba = np.zeros((200, 30, 4), np.uint8)
    rgba[..., :3] = 200
    rgba[:, 5:25, 3] = 255
    rgba[50:60, 5:9, 3] = 0       # dent: ink at the edge was thresholded away
    out = repair_strip_edges(rgba)
    assert (out[50:60, 5:9, 3] == 255).all()
    assert (out[:, :5, 3] == 0).all()
