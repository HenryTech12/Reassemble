"""Photo / scan of pieces on a dark background -> clean RGBA ``Piece`` objects.

Steps, each chosen for robustness rather than cleverness:

1. **Rectify** (perspective correction).  If at least three of the four ArUco markers of
   the printed capture mat are visible, their corners give an exact homography to the
   mat's frame.  Without markers we skip rectification (a flatbed scan needs none).
2. **Flat-field** the illumination.  A phone photo is brighter in the middle and darker at
   the edges.  We estimate that slowly varying light field from the *background* (a
   heavily blurred version of the image with the pieces masked out) and divide by it.
3. **Segment** paper from background: Otsu threshold on lightness, a little morphology,
   then connected components.  Pieces that touch are split with a distance-transform
   watershed (narrow necks between two strips become the watershed lines).
4. **Deskew** each piece with the PCA long axis of its mask, crop, keep the mask as alpha.
5. **Normalise height** so all strips have the same height (they came from one page).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..model import Piece

ERODE = 2  # px shaved off each piece boundary at 2x capture resolution (mixed paper/background colour; 1 too little, 3 too much in our tests)
from ..synth.photo import MARKER_IDS, MAT_H, MAT_W, aruco_dict, marker_corners_mat
from ..features.normalize import mask_angle
from ..synth.shredders import rotate_rgba


@dataclass
class CaptureReport:
    rectified: bool = False
    markers_found: int = 0
    n_components: int = 0
    n_pieces: int = 0
    split_by_watershed: int = 0
    rejected_small: int = 0
    notes: list[str] = field(default_factory=list)


def find_markers(img_bgr: np.ndarray) -> dict[int, np.ndarray]:
    det = cv2.aruco.ArucoDetector(aruco_dict(), cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    out = {}
    if ids is not None:
        for c, i in zip(corners, ids.ravel()):
            if int(i) in MARKER_IDS:
                out[MARKER_IDS.index(int(i))] = c.reshape(4, 2)
    return out


def rectify(img_bgr: np.ndarray, report: CaptureReport, scale: float = 2.0) -> np.ndarray:
    """Warp the photo into the mat frame using the ArUco markers (needs >= 3 markers).

    ``scale`` is the mat-frame resolution (1.0 = MAT_W x MAT_H px); 2.0 keeps roughly the
    resolution of a phone photo so the 1-px boundary loss stays small relative to strokes.
    """
    found = find_markers(img_bgr)
    report.markers_found = len(found)
    if len(found) < 3:
        report.notes.append("fewer than 3 mat markers found: no perspective correction")
        return img_bgr
    ref = marker_corners_mat(scale)
    src = np.concatenate([found[k] for k in sorted(found)]).astype(np.float32)
    dst = np.concatenate([ref[k] for k in sorted(found)]).astype(np.float32)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    report.rectified = True
    W, Hh = int(MAT_W * scale), int(MAT_H * scale)
    out = cv2.warpPerspective(img_bgr, H, (W, Hh), flags=cv2.INTER_CUBIC, borderValue=(20, 20, 20))
    # paint over the markers (and their white frames) so they are not taken as pieces
    for c in ref.values():
        x0, y0 = c.min(0).astype(int) - int(20 * scale)
        x1, y1 = c.max(0).astype(int) + int(20 * scale)
        out[max(0, y0):y1, max(0, x0):x1] = np.median(out[Hh // 2 - 5:Hh // 2 + 5, 5:15], axis=(0, 1))
    return out


def paper_mask(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = cv2.GaussianBlur(lab[..., 0], (5, 5), 0)
    _, m = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # fill holes (dark ink blobs inside a piece must stay paper)
    filled = m.copy()
    h, w = m.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(filled, ff, (0, 0), 255)
    return m | cv2.bitwise_not(filled)


def flat_field(img_bgr: np.ndarray, mask: np.ndarray, down: int = 16) -> np.ndarray:
    """Divide out uneven lighting, estimated on the paper itself.

    Paper is (roughly) uniformly white, so a heavy blur of the image *restricted to paper
    pixels* (normalised convolution) is an estimate of the light falling on each point.
    The light field is smooth by definition, so we estimate it on a ``down``-times smaller
    image and upsample: same result, ~250x faster than blurring at full resolution.
    """
    h, w = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, (max(1, w // down), max(1, h // down)), interpolation=cv2.INTER_AREA).astype(np.float32)
    m = cv2.resize((mask > 0).astype(np.float32), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    m = (m > 0.9).astype(np.float32)
    s = max(small.shape[:2]) / 12
    num = cv2.GaussianBlur(small * m[..., None], (0, 0), s)
    den = cv2.GaussianBlur(m, (0, 0), s)[..., None]
    light = cv2.resize(num / np.maximum(den, 1e-3), (w, h), interpolation=cv2.INTER_LINEAR)
    f = img_bgr.astype(np.float32)
    target = np.median(f[mask > 0], axis=0) if (mask > 0).any() else 230
    out = f * (target / np.maximum(light, 1.0))
    return np.clip(out, 0, 255).astype(np.uint8)


def _cores_side_by_side(comp: np.ndarray, markers: np.ndarray, nm: int, half_width: float) -> bool:
    """True if distance-transform cores are spread *across* the blob's long axis.

    Two touching strips give two cores next to each other; one strip with a few ink
    dents gives several cores in a line along it.  We project core centroids on the
    blob's short (PCA) axis and require a spread of more than one piece half-width.
    """
    ys, xs = np.nonzero(comp)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    mu = pts.mean(0)
    evals, evecs = np.linalg.eigh(np.cov((pts - mu).T))
    short = evecs[:, np.argmin(evals)]
    proj = []
    for k in range(1, nm):
        cy, cx = np.argwhere(markers == k).mean(0)
        proj.append((np.array([cx, cy]) - mu) @ short)
    return (max(proj) - min(proj)) > half_width


def split_touching(mask: np.ndarray, report: CaptureReport, min_area: int) -> np.ndarray:
    """Label pieces; split touching ones with a distance-transform watershed."""
    n, lab = cv2.connectedComponents(mask)
    report.n_components = n - 1
    areas = np.bincount(lab.ravel())
    med = np.median(areas[1:][areas[1:] >= min_area]) if (areas[1:] >= min_area).any() else 0
    out = np.zeros_like(lab)
    nxt = 1
    for k in range(1, n):
        comp = (lab == k).astype(np.uint8)
        if areas[k] < min_area:
            report.rejected_small += 1
            continue
        if True:
            # Touching pieces form one blob with a narrow neck.  The distance transform
            # peaks in the middle of each piece, so thresholding it at 60 % of its max
            # leaves one "core" per piece; more than one core => split by watershed.
            dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
            _, core = cv2.threshold(dist, 0.6 * dist.max(), 255, cv2.THRESH_BINARY)
            nm, markers = cv2.connectedComponents(core.astype(np.uint8))
            if nm > 2 and not _cores_side_by_side(comp, markers, nm, dist.max()):
                nm = 2  # several cores *along* one strip (ink dents), not touching pieces
            if nm > 2:
                markers = markers + 1
                markers[comp == 0] = 0
                markers[(comp == 1) & (core == 0)] = 0
                ws = cv2.watershed(cv2.cvtColor(comp * 255, cv2.COLOR_GRAY2BGR), markers.astype(np.int32))
                for m_id in range(2, nm + 1):
                    sel = (ws == m_id) & (comp == 1)
                    if sel.sum() >= min_area:
                        out[sel] = nxt
                        nxt += 1
                report.split_by_watershed += 1
                continue
        out[comp == 1] = nxt
        nxt += 1
    return out


def extract_piece(img_bgr: np.ndarray, region: np.ndarray, pid: str) -> Piece:
    """Crop one labelled region, deskew it with its min-area rectangle, return RGBA."""
    ys, xs = np.nonzero(region)
    pad = 6
    y0, y1 = max(0, ys.min() - pad), min(region.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(region.shape[1], xs.max() + pad + 1)
    rgb = img_bgr[y0:y1, x0:x1, ::-1]
    m = region[y0:y1, x0:x1].astype(np.uint8) * 255
    # PCA of the mask: deviation of the long axis from vertical (pieces are assumed to be
    # laid roughly upright, i.e. within +/-45 degrees; a 180-degree flip can't be told apart)
    ang = mask_angle(m > 0)
    # Rotate colour and mask *separately*: colour with full opacity so that pixels just
    # outside the thresholded mask keep their real colour (edge repair below needs them).
    full = np.dstack([rgb, np.full(m.shape, 255, np.uint8)])
    mk = np.dstack([np.repeat(m[..., None], 3, 2), np.full(m.shape, 255, np.uint8)])
    if abs(ang) > 0.05:
        full = rotate_rgba(full, -ang)
        mk = rotate_rgba(mk, -ang)
    rgba = full.copy()
    rgba[..., 3] = np.where((mk[..., 0] > 127) & (mk[..., 3] > 127), 255, 0).astype(np.uint8)
    if rgba.shape[0] > 2.5 * rgba.shape[1]:
        rgba = repair_strip_edges(rgba)
    if ERODE:
        # shave the outermost pixel: a mix of paper and background colour
        a = cv2.erode(rgba[..., 3], np.ones((1, 2 * ERODE + 1), np.uint8))
        rgba[..., 3] = a
    yy, xx = np.nonzero(rgba[..., 3])
    rgba = rgba[yy.min():yy.max() + 1, xx.min():xx.max() + 1].copy()
    rgba[rgba[..., 3] == 0, :3] = 0
    return Piece(id=pid, image=rgba, meta={"source_bbox": [int(x0), int(y0), int(x1), int(y1)],
                                            "rotation_deg": float(ang),
                                            "centre": [float(xs.mean()), float(ys.mean())]})


def repair_strip_edges(rgba: np.ndarray, window: int = 41) -> np.ndarray:
    """Give back the ink that thresholding stole from a strip's edges.

    On a dark background, dark ink touching the cut edge is indistinguishable from the
    background by brightness, so the mask gets a *dent* exactly where the most useful
    pixels for matching are.  A cut edge is locally straight, so we estimate it per row
    with a rolling median of the boundary position and fill any row whose boundary sits
    to that line.  Clear outward bulges (real fraying) are kept.
    The colours of the refilled pixels come from the image itself (they were cropped
    with a margin), so ink at the edge reappears.
    """
    from scipy.ndimage import median_filter

    m = rgba[..., 3] > 127
    h, w = m.shape
    has = m.any(1)
    if has.sum() < 10:
        return rgba
    left = np.where(has, np.argmax(m, 1), np.nan)
    right = np.where(has, w - 1 - np.argmax(m[:, ::-1], 1), np.nan)
    rows = np.arange(h)
    fill = lambda a: np.interp(rows, rows[has], a[has])
    left, right = fill(left), fill(right)
    lf = median_filter(left, size=window, mode="nearest")
    rf = median_filter(right, size=window, mode="nearest")
    # snap to the smoothed edge line; only clear outward bulges (> 2 px, real fraying)
    # survive.  Thresholding noise otherwise leaves a 1-px jagged edge that ruins the
    # row-by-row seam comparison.
    new_l = np.where(left < lf - 2, left, np.round(lf)).astype(int)
    new_r = np.where(right > rf + 2, right, np.round(rf)).astype(int)
    xs = np.arange(w)[None, :]
    newm = (xs >= new_l[:, None]) & (xs <= new_r[:, None])
    top, bot = np.flatnonzero(has)[[0, -1]]
    newm[:top] = False
    newm[bot + 1:] = False
    out = rgba.copy()
    out[..., 3] = np.where(newm | m, 255, 0).astype(np.uint8)
    return out


def normalise_heights(pieces: list[Piece], tol: float = 0.08) -> list[Piece]:
    """Resize pieces whose height is within ``tol`` of the median to exactly the median
    (small scale differences across the photo after rectification)."""
    if not pieces:
        return pieces
    H = int(np.median([p.height for p in pieces]))
    out = []
    for p in pieces:
        if p.height != H and abs(p.height - H) / H < tol:
            w = max(1, int(round(p.width * H / p.height)))
            img = cv2.resize(p.image, (w, H), interpolation=cv2.INTER_AREA)
            img[..., 3] = np.where(img[..., 3] > 127, 255, 0)
            p = Piece(id=p.id, image=img, meta=p.meta)
        out.append(p)
    return out


def capture(img_bgr: np.ndarray, min_area_frac: float = 0.0005, normalise: bool = True,
            mat_scale: float = 2.0, out_scale: float = 1.0) -> tuple[list[Piece], CaptureReport]:
    """Full pipeline; returns pieces ordered left-to-right as they lay on the mat.

    ``out_scale`` resizes the final pieces (e.g. 0.5 to bring a 2x capture back to the
    ~100 dpi the learned matcher was trained on)."""
    rep = CaptureReport()
    img = rectify(img_bgr, rep, mat_scale)
    m0 = paper_mask(img)
    img = flat_field(img, m0)
    mask = paper_mask(img)
    labels = split_touching(mask, rep, int(min_area_frac * mask.size))
    pieces = []
    ids = [k for k in np.unique(labels) if k > 0]
    for k in ids:
        p = extract_piece(img, labels == k, f"c{len(pieces):03d}")
        pieces.append(p)
    pieces.sort(key=lambda p: p.meta["centre"][0])
    for i, p in enumerate(pieces):
        p.id = f"c{i:03d}"
    if out_scale != 1.0:
        for p in pieces:
            h, w = p.image.shape[:2]
            img_s = cv2.resize(p.image, (max(1, int(round(w * out_scale))), max(1, int(round(h * out_scale)))),
                               interpolation=cv2.INTER_AREA)
            img_s[..., 3] = np.where(img_s[..., 3] > 127, 255, 0)
            p.image = img_s
    if normalise:
        pieces = normalise_heights(pieces)
    rep.n_pieces = len(pieces)
    return pieces, rep
