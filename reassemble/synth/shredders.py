"""Digital shredders: turn a page image into shuffled pieces + ground truth.

Every shredder returns ``(pieces, truth)`` where ``truth`` is a JSON-serialisable dict.
Piece ids are assigned from a random permutation, so the file name of a piece tells you
nothing about where it belongs.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..model import Piece, to_rgba


@dataclass
class Realism:
    """Physical/scan imperfections applied per piece.

    The defaults are a *clean* shred.  ``Realism.noisy()`` gives the "realistic" preset
    used for the honest-accuracy benchmark.
    """

    rotation_deg: float = 0.0     # max |rotation| per piece
    noise_sigma: float = 0.0      # gaussian scan noise (grey levels)
    blur_sigma: float = 0.0       # optical blur
    brightness: float = 0.0       # max relative brightness change per piece
    width_jitter: float = 0.0     # relative variation of strip width
    fray_px: int = 0              # max depth of irregular edge loss (pixels)
    drop: int = 0                 # number of pieces to remove (missing pieces)

    @classmethod
    def noisy(cls, drop: int = 0) -> "Realism":
        return cls(rotation_deg=0.4, noise_sigma=6.0, blur_sigma=0.7, brightness=0.06,
                   width_jitter=0.08, fray_px=3, drop=drop)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- primitives


def _random_walk(n: int, max_amp: float, rng: np.random.Generator, smooth: float = 2.0) -> np.ndarray:
    """Smoothed random walk in [0, max_amp] used for frayed / torn boundaries."""
    if max_amp <= 0:
        return np.zeros(n)
    walk = np.cumsum(rng.normal(0, 1, n))
    walk = gaussian_filter1d(walk, smooth)
    walk -= walk.min()
    if walk.max() > 0:
        walk = walk / walk.max() * max_amp
    jitter = rng.uniform(0, 1, n)
    return np.clip(walk + jitter, 0, max_amp)


def ragged_cuts(xs: np.ndarray, h: int, depth: int, rng: np.random.Generator) -> np.ndarray:
    """Per-row x position of every cut line: shape (len(xs), h).

    A blade (or a tear) doesn't destroy paper, it *divides* it: if strip A bulges out by
    2 px at some row, strip B is missing exactly those 2 px.  So each internal cut is one
    shared wavy curve and the two neighbours get complementary boundaries.  The page's own
    left and right borders stay straight.
    """
    cuts = np.repeat(xs[:, None].astype(np.float64), h, axis=1)
    if depth > 0:
        for k in range(1, len(xs) - 1):
            cuts[k] += _random_walk(h, depth, rng) - depth / 2
    return np.round(cuts).astype(int)


def degrade(rgba: np.ndarray, r: Realism, rng: np.random.Generator, rotate: bool = True) -> np.ndarray:
    """Apply brightness, blur, noise and rotation to one RGBA piece."""
    img = rgba.astype(np.float32)
    rgb, a = img[..., :3], img[..., 3:]
    if r.brightness:
        rgb *= 1.0 + rng.uniform(-r.brightness, r.brightness)
    if r.blur_sigma:
        # normalised convolution: blur the paper content without smearing in the
        # transparent (non-paper) surroundings
        w = (a[..., 0] > 0).astype(np.float32)
        num = cv2.GaussianBlur(rgb * w[..., None], (0, 0), r.blur_sigma)
        den = cv2.GaussianBlur(w, (0, 0), r.blur_sigma)[..., None]
        rgb = np.where(den > 1e-3, num / np.maximum(den, 1e-3), rgb)
    if r.noise_sigma:
        rgb += rng.normal(0, r.noise_sigma, rgb.shape)
    rgb = np.clip(rgb, 0, 255)
    rgb[a[..., 0] == 0] = 0  # transparent pixels carry no colour
    out = np.concatenate([rgb, a], -1).astype(np.uint8)
    if rotate and r.rotation_deg:
        out = rotate_rgba(out, rng.uniform(-r.rotation_deg, r.rotation_deg))
    return out


def rotate_rgba(rgba: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate an RGBA image about its centre, expanding the canvas so nothing is cut."""
    h, w = rgba.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(np.ceil(h * sin + w * cos)), int(np.ceil(h * cos + w * sin))
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    # premultiply so transparent black doesn't bleed into edge colours
    f = rgba.astype(np.float32)
    f[..., :3] *= f[..., 3:] / 255.0
    out = cv2.warpAffine(f, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    alpha = np.clip(out[..., 3:], 1e-3, 255)
    out[..., :3] = np.where(out[..., 3:] > 0, out[..., :3] * 255.0 / alpha, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _ids(n: int, rng: random.Random, prefix: str = "p") -> list[str]:
    nums = list(range(n))
    rng.shuffle(nums)
    width = max(3, len(str(n)))
    return [f"{prefix}{k:0{width}d}" for k in nums]


def cut_positions(width: int, n: int, jitter: float, rng: np.random.Generator) -> np.ndarray:
    """Return n+1 increasing x positions from 0 to width."""
    w = np.full(n, width / n) * (1 + rng.uniform(-jitter, jitter, n))
    w = w / w.sum() * width
    pos = np.concatenate([[0], np.cumsum(w)])
    return np.round(pos).astype(int)


# --------------------------------------------------------------------------- strip-cut


def strip_shred(page: np.ndarray, n: int = 30, realism: Realism | None = None, seed: int = 0,
                back: np.ndarray | None = None, id_prefix: str = "p") -> tuple[list[Piece], dict]:
    """Cut a page into ``n`` vertical strips.

    If ``back`` is given it is the reverse side *as you would read it*.  A strip at
    physical columns [a, b) of the front carries back-page columns [W-b, W-a): turning the
    sheet over mirrors it left-to-right.  The scanned back strip is stored in reading
    orientation, so matching must compare B.back's right edge with A.back's left edge.
    """
    r = realism or Realism()
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    h, w = page.shape[:2]
    xs = cut_positions(w, n, r.width_jitter, nrng)
    cuts = ragged_cuts(xs, h, r.fray_px, nrng)
    ids = _ids(n, rng, id_prefix)
    cols = np.arange(w)[None, :]
    pieces = []
    for i in range(n):
        lo, hi = cuts[i], cuts[i + 1]
        a, b = int(lo.min()), int(hi.max())
        full = (cols >= lo[:, None]) & (cols < hi[:, None])
        mask = np.where(full[:, a:b], 255, 0).astype(np.uint8)
        front = to_rgba(page[:, a:b])
        front[..., 3] = mask
        p = Piece(id=ids[i], image=degrade(front, r, nrng))
        if back is not None:
            bk = to_rgba(back[:, w - b:w - a])
            bk[..., 3] = mask[:, ::-1]
            p.back = degrade(bk, r, nrng)
        pieces.append(p)
    order = list(ids)
    missing = sorted(rng.sample(order, r.drop)) if r.drop else []
    kept = [p for p in pieces if p.id not in missing]
    rng.shuffle(kept)
    truth = {"kind": "strip", "n": n, "order": order, "missing": missing,
             "cuts": xs.tolist(), "piece_x": {ids[i]: int(cuts[i].min()) for i in range(n)}, "page_size": [h, w], "realism": r.to_dict(),
             "double_sided": back is not None}
    return kept, truth


# --------------------------------------------------------------------------- cross-cut


def crosscut_shred(page: np.ndarray, rows: int, cols: int, realism: Realism | None = None,
                   seed: int = 0) -> tuple[list[Piece], dict]:
    """Cut a page into a ``rows x cols`` grid of rectangles (a cross-cut shredder)."""
    r = realism or Realism()
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    h, w = page.shape[:2]
    ys = cut_positions(h, rows, 0, nrng)
    xs = cut_positions(w, cols, 0, nrng)
    ids = _ids(rows * cols, rng)
    grid, pieces = [], []
    k = 0
    for i in range(rows):
        row = []
        for j in range(cols):
            crop = to_rgba(page[ys[i]:ys[i + 1], xs[j]:xs[j + 1]])
            pieces.append(Piece(id=ids[k], image=degrade(crop, r, nrng, rotate=False)))
            row.append(ids[k])
            k += 1
        grid.append(row)
    rng.shuffle(pieces)
    truth = {"kind": "crosscut", "rows": rows, "cols": cols, "grid": grid, "missing": [],
             "page_size": [h, w], "realism": r.to_dict()}
    return pieces, truth


# --------------------------------------------------------------------------- torn paper


def torn_shred(page: np.ndarray, rows: int = 2, cols: int = 4, amplitude: float = 14.0,
               seed: int = 0, realism: Realism | None = None) -> tuple[list[Piece], dict]:
    """Tear a page into pieces along random-walk boundaries.

    Each vertical tear is a curve x = c_k + walk(y); each horizontal tear y = r_k + walk(x).
    A pixel's piece is (#horizontal tears above it, #vertical tears left of it).  A thin
    lighter "fibre" rim is drawn along torn edges, like the white fluff on real torn paper.
    """
    r = realism or Realism()
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    h, w = page.shape[:2]

    def tear(length: int, base: float) -> np.ndarray:
        walk = np.cumsum(nrng.normal(0, 1, length))
        walk = gaussian_filter1d(walk, 6)
        walk = walk - walk.mean()
        walk = walk / (np.abs(walk).max() + 1e-6) * amplitude
        return base + walk + nrng.normal(0, 0.8, length)

    vtears = [tear(h, w * k / cols) for k in range(1, cols)]
    htears = [tear(w, h * k / rows) for k in range(1, rows)]
    yy, xx = np.mgrid[0:h, 0:w]
    col_idx = np.zeros((h, w), np.int32)
    for t in vtears:
        col_idx += (xx >= t[:, None]).astype(np.int32)
    row_idx = np.zeros((h, w), np.int32)
    for t in htears:
        row_idx += (yy >= t[None, :]).astype(np.int32)
    label = row_idx * cols + col_idx
    ids = _ids(rows * cols, rng)
    pieces, grid, offsets, centres = [], [[None] * cols for _ in range(rows)], {}, {}
    for i in range(rows):
        for j in range(cols):
            k = i * cols + j
            m = label == k
            ys_, xs_ = np.nonzero(m)
            y0, y1, x0, x1 = ys_.min(), ys_.max() + 1, xs_.min(), xs_.max() + 1
            crop = to_rgba(page[y0:y1, x0:x1]).copy()
            pm = m[y0:y1, x0:x1]
            crop[..., 3] = np.where(pm, 255, 0)
            # fibre rim: pixels near the boundary get lighter
            er = cv2.erode(pm.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)
            rim = pm & (er == 0)
            rgb = crop[..., :3].astype(np.float32)
            rgb[rim] = rgb[rim] * 0.4 + 255 * 0.6 * nrng.uniform(0.8, 1.0, (rim.sum(), 1))
            crop[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
            crop[~pm, :3] = 0
            pieces.append(Piece(id=ids[k], image=degrade(crop, r, nrng)))
            grid[i][j] = ids[k]
            offsets[ids[k]] = [int(x0), int(y0)]
            centres[ids[k]] = [float(xs_.mean()), float(ys_.mean())]
    rng.shuffle(pieces)
    truth = {"kind": "torn", "rows": rows, "cols": cols, "grid": grid, "offsets": offsets, "centres": centres,
             "missing": [], "page_size": [h, w], "realism": r.to_dict()}
    return pieces, truth


# --------------------------------------------------------------------------- mixed bins


def mixed_bin(pages: list[np.ndarray], n: int = 30, realism: Realism | None = None,
              seed: int = 0) -> tuple[list[Piece], dict]:
    """Strip-shred several pages and shuffle all the strips into one pile."""
    rng = random.Random(seed)
    all_pieces, docs, doc_of = [], [], {}
    for d, page in enumerate(pages):
        pcs, tr = strip_shred(page, n, realism, seed=seed * 101 + d, id_prefix=f"t{d}_")
        docs.append({"order": tr["order"], "missing": tr["missing"]})
        for p in pcs:
            doc_of[p.id] = d
        all_pieces.extend(pcs)
    # re-label with neutral ids so the prefix doesn't leak the document
    new_ids = _ids(len(all_pieces), rng)
    mapping = {p.id: nid for p, nid in zip(all_pieces, new_ids)}
    # dropped pieces also need names for the truth file
    extra = 0
    for doc in docs:
        for pid in doc["order"]:
            if pid not in mapping:
                mapping[pid] = f"x{extra:03d}"
                extra += 1
    for p in all_pieces:
        p.id = mapping[p.id]
    for doc in docs:
        doc["order"] = [mapping[i] for i in doc["order"]]
        doc["missing"] = [mapping[i] for i in doc["missing"]]
    doc_of = {mapping[k]: v for k, v in doc_of.items()}
    rng.shuffle(all_pieces)
    truth = {"kind": "mixed", "documents": docs, "doc_of": doc_of, "n_per_doc": n,
             "missing": [m for d in docs for m in d["missing"]],
             "realism": (realism or Realism()).to_dict()}
    return all_pieces, truth
