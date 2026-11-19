"""Learned edge matcher: a small CNN that looks across a seam.

Input is a *seam patch*: 32 rows x 16 columns x RGB, made of the last 8 columns of strip A
followed by the first 8 columns of strip B (read relative to the paper boundary, like the
handcrafted features).  The network outputs a logit for "A and B are true neighbours".
This is the early-fusion design used by Paixão et al. for strip-shredded documents: the
convolutions see both sides at once, so they can learn stroke continuity directly instead
of us hand-designing a predictor.

A pair's cost is the mean negative log-probability over its *informative* patches (those
with some ink on either side).  Blank patches carry no evidence and are skipped.

Training data comes for free from the synthetic generator (true neighbours vs hard
negatives from the same page), so the whole thing trains on a CPU in a few minutes.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..features.edges import read_edge
from ..model import Piece

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_WEIGHTS = MODEL_DIR / "edge_cnn.pt"
K = 8          # columns per side
PATCH_H = 32   # rows per patch
STRIDE = 16    # vertical stride between patches


# --------------------------------------------------------------------------- seam tensors


def edge_columns(piece: Piece) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(right, right_valid, left, left_valid) with right flipped to left-to-right order.

    Values are paper-balanced and scaled to roughly [-1, 1].
    """
    rgb = piece.rgb.astype(np.float32)
    m = piece.mask
    paper = np.median(rgb[m], 0) if m.any() else np.full(3, 255.0)
    bal = rgb * (230.0 / np.maximum(paper, 1.0))
    r, rv, _ = read_edge(bal, m, "right", K, 0, 0.0)
    l, lv, _ = read_edge(bal, m, "left", K, 0, 0.0)
    r = r[:, ::-1]  # outermost column last, so A's columns read left->right into the seam
    scale = lambda x: np.clip(x, 0, 255) / 127.5 - 1.0
    return scale(r), rv, scale(l), lv


def seam_patches(a: tuple, b: tuple, H: int | None = None, stride: int = STRIDE) -> tuple[np.ndarray, np.ndarray]:
    """All vertical patches of the seam A|B: returns (P, 3, PATCH_H, 2K) and an
    informative-mask (P,)."""
    ra, rva = a[0], a[1]
    lb, lvb = b[2], b[3]
    H = H or min(len(ra), len(lb))
    seam = np.concatenate([ra[:H], lb[:H]], axis=1)  # (H, 2K, 3)
    valid = rva[:H] & lvb[:H]
    starts = list(range(0, max(1, H - PATCH_H + 1), stride))
    patches = np.stack([seam[s:s + PATCH_H] for s in starts])  # (P, PATCH_H, 2K, 3)
    vmask = np.stack([valid[s:s + PATCH_H].mean() for s in starts])
    ink = (patches.min(axis=(1, 2, 3)) < 0.1)  # some dark pixel (< ~140 grey)
    info = ink & (vmask > 0.8)
    return patches.transpose(0, 3, 1, 2).astype(np.float32), info


# --------------------------------------------------------------------------- model


def build_model():
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),                                  # 16 x 8
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),                                  # 8 x 4
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((2, 1)),
        nn.Flatten(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 1),
    )


_MODEL_CACHE: dict = {}


def load_model(path: Path | str = DEFAULT_WEIGHTS):
    import torch

    path = Path(path)
    if path not in _MODEL_CACHE:
        if not path.exists():
            raise FileNotFoundError(f"no learned matcher weights at {path}; run `reassemble train`")
        m = build_model()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        _MODEL_CACHE[path] = m
    return _MODEL_CACHE[path]


# --------------------------------------------------------------------------- inference


def learned_cost(pieces: list[Piece], weights: Path | str = DEFAULT_WEIGHTS, batch: int = 4096,
                 stride: int = 32) -> np.ndarray:
    """N x N cost matrix: mean -log P(neighbour) over informative seam patches.

    Inference uses non-overlapping patches (stride 32) — half the work of the training
    stride with no measurable accuracy loss — and only the informative ones.
    """
    import torch
    import torch.nn.functional as F

    model = load_model(weights)
    cols = [edge_columns(p) for p in pieces]
    H = min(p.height for p in pieces)
    n = len(pieces)
    D = np.full((n, n), 1e6)
    all_p, owners, infos = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p, info = seam_patches(cols[i], cols[j], H, stride)
            all_p.append(p[info])  # only informative patches go through the network
            infos.append(int(info.sum()))
            owners.append((i, j))
    neutral = float(np.log(2.0))
    if not all_p or sum(infos) == 0:
        D[~np.eye(n, dtype=bool)] = neutral
        return D
    X = torch.from_numpy(np.concatenate(all_p))
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            out.append(-F.logsigmoid(model(X[s:s + batch])).squeeze(1).numpy())
    nll = np.concatenate(out)
    k = 0
    for (i, j), cnt in zip(owners, infos):
        D[i, j] = float(nll[k:k + cnt].mean()) if cnt else neutral
        k += cnt
    return D


# --------------------------------------------------------------------------- training data


TRAIN_FONTS = {"liberation-serif", "liberation-sans", "dejavu-sans", "free-serif", "liberation-mono", "caveat"}
VAL_FONTS = {"caladea", "carlito", "dejavu-serif", "indie-flower"}


@dataclass
class PatchSet:
    X: np.ndarray
    y: np.ndarray


def pairs_to_patches(pieces: list[Piece], pairs: list[tuple[str, str, int]], max_per_pair: int = 12,
                     rng: random.Random | None = None) -> PatchSet:
    """Turn labelled (left_id, right_id, label) pairs into informative seam patches."""
    rng = rng or random.Random(0)
    by_id = {p.id: p for p in pieces}
    cols = {p.id: edge_columns(p) for p in pieces}
    H = min(p.height for p in pieces)
    X, y = [], []
    for a, b, lab in pairs:
        if a not in by_id or b not in by_id:
            continue
        p, info = seam_patches(cols[a], cols[b], H)
        idx = list(np.flatnonzero(info))
        rng.shuffle(idx)
        for k in idx[:max_per_pair]:
            X.append(p[k])
            y.append(lab)
    if not X:
        return PatchSet(np.zeros((0, 3, PATCH_H, 2 * K), np.float32), np.zeros(0, np.float32))
    return PatchSet(np.stack(X), np.array(y, np.float32))


def make_dataset(n_pages: int = 60, fonts: set[str] | None = None, seed: int = 0,
                 n_strips=(20, 45), negatives_per_pos: int = 2) -> PatchSet:
    """Generate seam patches from synthetic pages.

    Negatives are *hard*: other strips of the same page (same font, same line positions),
    including near-misses two or three strips away.  Each page gets random realism, so the
    model sees clean and degraded scans.
    """
    from ..features.normalize import normalize_heights, normalize_piece
    from ..synth.pages import render_page
    from ..synth.shredders import Realism, strip_shred

    rng = random.Random(seed)
    kinds = ["memo", "letter", "invoice", "photo", "handwriting"]
    Xs, ys = [], []
    page_seed = seed * 100_000
    made = 0
    while made < n_pages:
        page_seed += 1
        kind = rng.choice(kinds)
        page, meta = render_page(kind, page_seed)
        if fonts is not None and meta["font"] not in fonts:
            continue
        made += 1
        r = Realism.noisy() if rng.random() < 0.6 else Realism()
        if rng.random() < 0.3:
            r.blur_sigma = rng.uniform(0.3, 1.2)
        pieces, truth = strip_shred(page, rng.randint(*n_strips), r, seed=page_seed)
        pieces = normalize_heights([normalize_piece(p) for p in pieces])
        order = truth["order"]
        pairs = [(a, b, 1) for a, b in zip(order, order[1:])]
        for i, a in enumerate(order[:-1]):
            for _ in range(negatives_per_pos):
                j = i + rng.choice([2, 3, -1, -2]) if rng.random() < 0.4 else rng.randrange(len(order))
                if 0 <= j < len(order) and j != i + 1 and j != i:
                    pairs.append((a, order[j], 0))
        ps = pairs_to_patches(pieces, pairs, rng=rng)
        Xs.append(ps.X)
        ys.append(ps.y)
    return PatchSet(np.concatenate(Xs), np.concatenate(ys))


# --------------------------------------------------------------------------- training


def train(out: Path | str = DEFAULT_WEIGHTS, n_pages: int = 80, epochs: int = 6, seed: int = 0,
          init: Path | str | None = None, data: PatchSet | None = None, val: PatchSet | None = None,
          lr: float = 2e-3, verbose: bool = True) -> dict:
    """Train (or fine-tune from ``init``) and save weights.  Returns training stats."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    t0 = time.time()
    if data is None:
        data = make_dataset(n_pages, TRAIN_FONTS, seed=seed)
    if val is None:
        val = make_dataset(max(8, n_pages // 6), VAL_FONTS, seed=seed + 1)
    if verbose:
        print(f"data: {len(data.y)} train patches ({data.y.mean():.0%} positive), "
              f"{len(val.y)} val patches, built in {time.time() - t0:.0f}s")
    model = build_model()
    if init is not None:
        model.load_state_dict(torch.load(init, map_location="cpu"))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos_w = torch.tensor([(1 - data.y.mean()) / max(data.y.mean(), 1e-6)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    X, y = torch.from_numpy(data.X), torch.from_numpy(data.y)
    Xv, yv = torch.from_numpy(val.X), torch.from_numpy(val.y)
    hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        tot = 0.0
        for s in range(0, len(X), 256):
            idx = perm[s:s + 256]
            xb = X[idx]
            # augmentation: random +/-1 px vertical shift of the B half, brightness
            if np.random.rand() < 0.5:
                xb = xb.clone()
                xb[:, :, :, K:] = torch.roll(xb[:, :, :, K:], int(np.random.choice([-1, 1])), dims=2)
            xb = xb * (1 + 0.05 * torch.randn(len(xb), 1, 1, 1))
            opt.zero_grad()
            loss = loss_fn(model(xb).squeeze(1), y[idx])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xv).squeeze(1)).numpy()
        acc = float(((pv > 0.5) == (val.y > 0.5)).mean())
        hist.append({"epoch": ep + 1, "loss": tot / len(X), "val_acc": acc})
        if verbose:
            print(f"epoch {ep + 1}: loss {tot / len(X):.4f}  val patch acc {acc:.3f}  ({time.time() - t0:.0f}s)")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    _MODEL_CACHE.pop(out, None)
    return {"history": hist, "seconds": time.time() - t0, "train_patches": len(y)}


# --------------------------------------------------------------------------- human corrections


def finetune_from_corrections(corrections: list[dict], init: Path | str = DEFAULT_WEIGHTS,
                              out: Path | str | None = None, epochs: int = 3) -> dict:
    """Fine-tune on pairs exported from the web editor.

    ``corrections`` items: ``{"case": folder, "left": id, "right": id, "label": 1|0}`` —
    label 1 for joins a human locked/confirmed, 0 for joins a human split.  The original
    synthetic data is mixed in (replay) so the model doesn't forget what it knew.
    """
    from ..features.normalize import normalize_heights, normalize_piece
    from ..model import load_case

    by_case: dict[str, list] = {}
    for c in corrections:
        by_case.setdefault(c["case"], []).append((c["left"], c["right"], int(c["label"])))
    Xs, ys = [], []
    for case, pairs in by_case.items():
        pieces, _ = load_case(case)
        pieces = normalize_heights([normalize_piece(p) for p in pieces])
        ps = pairs_to_patches(pieces, pairs, max_per_pair=40)
        Xs += [ps.X] * 4  # upweight human labels
        ys += [ps.y] * 4
    replay = make_dataset(15, TRAIN_FONTS, seed=99)
    data = PatchSet(np.concatenate(Xs + [replay.X]), np.concatenate(ys + [replay.y]))
    return train(out or init, epochs=epochs, init=init, data=data, lr=5e-4)
