"""Separate a mixed bin of strips into documents before solving.

Strips from the same page share a lot that has nothing to do with seams:

* **text-line layout**: every strip of a page has its text lines at the same heights, so
  their horizontal ink profiles line up (the strongest cue);
* **paper tone and ink colour**: different printers, paper stock and pens;
* **font metrics**: x-height and line spacing.

We turn these into a pairwise distance, run average-linkage agglomerative clustering, and
pick the number of documents with the silhouette score.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from ..features.edges import EdgeFeatures


def _band_profile(f: EdgeFeatures, H: int) -> np.ndarray:
    """Binary 'is this row inside a text line' profile, lightly smoothed."""
    p = f.profile[:H]
    on = (p > max(p.max() * 0.1, 1e-6)).astype(np.float32)
    return gaussian_filter1d(on, 2.0)


def layout_distance(feats: list[EdgeFeatures], max_shift: int = 4) -> np.ndarray:
    """1 - best normalised correlation of text-band profiles (blank strips get NaN)."""
    H = min(f.height for f in feats)
    P = np.stack([_band_profile(f, H) for f in feats])
    n = len(feats)
    has = P.max(1) > 0.05
    P = P - P.mean(1, keepdims=True)
    norms = np.linalg.norm(P, axis=1) + 1e-9
    best = np.full((n, n), -1.0)
    for s in range(-max_shift, max_shift + 1):
        Q = np.roll(P, s, axis=1)
        best = np.maximum(best, (P @ Q.T) / np.outer(norms, norms))
    d = 1.0 - best
    d[~has, :] = np.nan
    d[:, ~has] = np.nan
    np.fill_diagonal(d, 0.0)
    return d


def appearance_vectors(feats: list[EdgeFeatures]) -> np.ndarray:
    rows = []
    for f in feats:
        rows.append(np.concatenate([f.paper / 255.0, f.ink / 255.0 * 0.5,
                                    [f.x_height / 30.0, f.line_spacing / 60.0]]))
    return np.array(rows)


def document_distance(feats: list[EdgeFeatures], w_layout: float = 1.0, w_appear: float = 1.0) -> np.ndarray:
    lay = layout_distance(feats)
    app = appearance_vectors(feats)
    app = (app - app.mean(0)) / (app.std(0) + 1e-6)
    da = np.linalg.norm(app[:, None] - app[None], axis=-1)
    da = da / (np.median(da[da > 0]) + 1e-9)
    lay_filled = np.where(np.isnan(lay), np.nanmedian(lay) if np.isfinite(lay).any() else 1.0, lay)
    lay_filled = lay_filled / (np.median(lay_filled[lay_filled > 0]) + 1e-9)
    # where layout is unknown (blank strips) rely on appearance only
    w_l = np.where(np.isnan(lay), 0.0, w_layout)
    d = (w_l * lay_filled + w_appear * da) / (w_l + w_appear)
    np.fill_diagonal(d, 0.0)
    return (d + d.T) / 2


def cluster_documents(feats: list[EdgeFeatures], k: int | None = None, knn: int = 4,
                      link_thresh: float = 0.55, min_size: int = 3, merge_thresh: float = 0.6) -> tuple[np.ndarray, dict]:
    """Return (labels, info).  ``k=None`` finds the number of documents automatically.

    1. Link every strip to its ``knn`` nearest strips by text-layout distance, keeping only
       links shorter than ``link_thresh`` (strips of one page correlate at ~0.25, strips of
       different pages at ~0.9).
    2. Connected components of that graph with at least ``min_size`` strips are documents.
       The number of documents is simply the number of such components — no need to guess
       k, which is what makes this more robust than silhouette-based selection.
    3. Components whose strips are still similar on median (< ``merge_thresh``) are merged.
    4. Leftover strips (blank margins, nearly empty strips) join the document whose members
       are closest on average, using the combined layout+appearance distance.

    If ``k`` is given, average-linkage agglomerative clustering with that k is used.
    """
    n = len(feats)
    if n < 4:
        return np.zeros(n, int), {"k": 1, "method": "trivial"}
    D = document_distance(feats, w_layout=1.0, w_appear=0.15)
    if k is not None:
        lab = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                      linkage="average").fit_predict(D)
        return lab, {"k": k, "method": "agglomerative"}
    L = layout_distance(feats)
    known = ~np.isnan(L).all(1) | (np.nansum(L, 1) > 0)
    Lf = np.where(np.isnan(L), np.inf, L)
    np.fill_diagonal(Lf, np.inf)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in np.argsort(Lf[i])[:knn]:
            if Lf[i, j] < link_thresh:
                parent[find(i)] = find(int(j))
    # merge components whose strips are, on median, still similar (a page whose left and
    # right halves have different line patterns can form two components)
    while True:
        roots = [find(i) for i in range(n)]
        comps = sorted(set(roots))
        if len(comps) < 2:
            break
        arr = np.array(roots)
        best, pair = merge_thresh, None
        for a_i, a in enumerate(comps):
            for b in comps[a_i + 1:]:
                block = L[np.ix_(arr == a, arr == b)]
                if np.isfinite(block).any():
                    m = float(np.nanmedian(block))
                    if m < best:
                        best, pair = m, (a, b)
        if pair is None:
            break
        parent[find(pair[0])] = find(pair[1])
    roots = [find(i) for i in range(n)]
    sizes = {r: roots.count(r) for r in set(roots)}
    docs = sorted([r for r, c in sizes.items() if c >= min_size], key=lambda r: -sizes[r])
    if len(docs) <= 1:
        return np.zeros(n, int), {"k": 1, "method": "knn-components"}
    lab = np.array([docs.index(r) if r in docs else -1 for r in roots])
    # Leftovers: compare paper *chromaticity* (r,g,b)/(r+g+b), which ignores the brightness
    # differences between scans but keeps the tint of the paper stock; plus layout if any.
    chrom = np.array([f.paper / max(f.paper.sum(), 1e-6) for f in feats])
    doc_chrom = [np.median(chrom[lab == d], 0) for d in range(len(docs))]
    for i in np.flatnonzero(lab < 0):
        dc = np.array([np.linalg.norm(chrom[i] - c) for c in doc_chrom])
        dl = np.array([np.nanmedian(L[i, lab == d]) if np.isfinite(L[i, lab == d]).any() else np.nan
                       for d in range(len(docs))])
        score = dc / (np.median(dc) + 1e-9)
        if np.isfinite(dl).all():
            score = score + 2 * dl / (np.median(dl) + 1e-9)
        lab[i] = int(np.argmin(score))
    return lab, {"k": len(docs), "method": "knn-components",
                 "sizes": [int((lab == d).sum()) for d in range(len(docs))]}


def purity(labels: np.ndarray, truth: list[int]) -> float:
    """Fraction of pieces whose cluster's majority document is their own document."""
    labels = np.asarray(labels)
    truth = np.asarray(truth)
    total = 0
    for c in set(labels):
        members = truth[labels == c]
        total += np.bincount(members).max()
    return total / len(labels)
