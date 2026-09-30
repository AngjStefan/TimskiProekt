"""
LV measurements from a segmentation mask, and the same measurements from an
EchoNet expert tracing, so prediction and ground truth are measured identically.

Geometry follows the single-plane method of disks (Simpson's rule) used for the
EchoNet-Dynamic labels:
  - long axis L from the apex to the midpoint of the mitral annulus
  - N chords perpendicular to the long axis, evenly spaced
  - V = sum(pi/4 * d_i^2) * L / N

All functions work in the pixel grid of the input. Divide lengths by `s`, areas
by `s**2` and volumes by `s**3` to convert to another grid (e.g. s = 224/112 to
report in native EchoNet pixels).
"""
import cv2
import numpy as np

N_DISKS = 20
BASAL_BAND = 0.25   # basal fraction of the long axis searched for the annulus hinges


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def region_from_contour(contour, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Filled binary region (uint8) for a contour; `shape` defaults to its extent."""
    pts = contour.reshape(-1, 2)
    if shape is None:
        shape = (int(pts[:, 1].max()) + 2, int(pts[:, 0].max()) + 2)
    region = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(region, [contour.reshape(-1, 1, 2).astype(np.int32)], -1, 1, thickness=-1)
    return region


def basal_hinges(pts: np.ndarray, origin: np.ndarray, u: np.ndarray):
    """Mitral annulus hinges: the basal corners of a contour.

    u points from base to apex. Within the basal band, the hinges are the
    points furthest along (base + side) at 45 degrees: argmax(-t - s) and
    argmax(-t + s) in the axis frame. Septal is the image-left one (EchoNet A4C
    shows the LV on the right, septum to its left).
    """
    n = np.array([-u[1], u[0]])
    if n[0] < 0:                      # point toward image-right
        n = -n
    t = (pts - origin) @ u
    s = (pts - origin) @ n
    band = t <= t.min() + BASAL_BAND * (t.max() - t.min())
    bp, bt, bs = pts[band], t[band], s[band]
    return bp[np.argmax(-bt - bs)], bp[np.argmax(-bt + bs)]


def lv_landmarks(contour, region: np.ndarray | None = None) -> dict | None:
    """Apex, mitral annulus hinges (basal septal/lateral) and annulus midpoint.

    The LV major axis comes from PCA of the region pixels; the apex is the end
    pointing up (A4C view: transducer and apex at the top). The hinges are the
    basal corners of the contour (see basal_hinges).
    """
    if contour is None:
        return None
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 5:
        return None
    if region is None:
        region = region_from_contour(contour)

    ys, xs = np.nonzero(region)
    if len(xs) < 10:
        return None
    coords = np.stack([xs, ys], axis=1).astype(np.float64)
    mu = coords.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov((coords - mu).T))
    u = eigvecs[:, np.argmax(eigvals)]
    if u[1] > 0:                      # point toward the top of the image
        u = -u

    apex = pts[np.argmax((pts - mu) @ u)]
    septal, lateral = basal_hinges(pts, mu, u)
    base_mid = (septal + lateral) / 2.0

    return {
        "apex": apex,
        "basal_septal": septal,
        "basal_lateral": lateral,
        "base_mid": base_mid,
    }


def _chord(region: np.ndarray, center: np.ndarray, direction: np.ndarray, step: float = 0.25):
    """Endpoints of the region run through `center` along +/- `direction`."""
    h, w = region.shape
    reach = float(np.hypot(h, w))
    offsets = np.arange(-reach, reach + step, step)
    xy = center[None, :] + offsets[:, None] * direction[None, :]
    xi = np.rint(xy[:, 0]).astype(int)
    yi = np.rint(xy[:, 1]).astype(int)
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    on = np.zeros(len(offsets), dtype=bool)
    on[inside] = region[yi[inside], xi[inside]] > 0
    if not on.any():
        return None

    # contiguous runs of `on`; keep the one containing the center, else the longest
    edges = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0] - 1
    zero = len(offsets) // 2
    hit = [(a, b) for a, b in zip(starts, ends) if a <= zero <= b]
    a, b = hit[0] if hit else max(zip(starts, ends), key=lambda r: r[1] - r[0])
    return xy[a], xy[b]


def method_of_disks(region: np.ndarray, apex, base_mid, n_disks: int = N_DISKS) -> dict | None:
    """Long axis length, disk chords and single-plane volume from a region mask."""
    apex, base_mid = np.asarray(apex, float), np.asarray(base_mid, float)
    axis = apex - base_mid
    L = float(np.linalg.norm(axis))
    if L < 1e-6:
        return None
    a = axis / L
    perp = np.array([-a[1], a[0]])

    chords = []
    for i in range(n_disks):
        c = base_mid + (i + 0.5) / n_disks * axis
        seg = _chord(region, c, perp)
        chords.append(seg if seg is not None else (c, c))
    diam = np.array([np.linalg.norm(p2 - p1) for p1, p2 in chords])
    volume = float(np.sum(np.pi / 4.0 * diam ** 2) * L / n_disks)
    return {"length": L, "chords": chords, "diameters": diam, "volume": volume}


def measure_mask(mask: np.ndarray, contour=None) -> dict | None:
    """Landmarks + area + method-of-disks for one predicted mask (mask pixels)."""
    from src.postprocessing.contours import get_lv_contour
    if contour is None:
        contour, _ = get_lv_contour(mask)
    if contour is None:
        return None
    region = region_from_contour(contour, mask.shape[:2])
    lm = lv_landmarks(contour, region)
    if lm is None:
        return None
    disks = method_of_disks(region, lm["apex"], lm["base_mid"])
    if disks is None:
        return None
    mid = disks["chords"][N_DISKS // 2]
    mid_septal, mid_lateral = sorted(mid, key=lambda p: p[0])
    return {
        **lm,
        "mid_septal": mid_septal,
        "mid_lateral": mid_lateral,
        "area": float(region.sum()),
        "region": region,
        "contour": contour,
        **disks,
    }


def gt_from_tracing(rows: np.ndarray, scale: float = 1.0, shape: tuple[int, int] | None = None) -> dict:
    """Same measurements from an EchoNet VolumeTracings group.

    rows: (K, 4) array of X1, Y1, X2, Y2 in native 112-px coordinates, CSV order.
    Row 0 is the long axis; rows 1.. are the disk chords. `scale` maps native
    coordinates onto the grid the caller works in (e.g. 2.0 for 224 px).
    """
    rows = np.asarray(rows, dtype=np.float64) * scale
    p, q = rows[0, :2], rows[0, 2:]
    apex, base_mid = (p, q) if p[1] < q[1] else (q, p)
    L = float(np.linalg.norm(apex - base_mid))

    ends = rows[1:]
    diam = np.hypot(ends[:, 2] - ends[:, 0], ends[:, 3] - ends[:, 1])
    volume = float(np.sum(np.pi / 4.0 * diam ** 2) * L / len(ends))

    # Chords run apex -> base and the mitral plane is usually oblique to them,
    # so the last chords are clipped short near one corner. The hinges are
    # therefore taken as the basal corners of the traced contour, measured
    # along the expert's own long axis (same rule as for predictions).
    polygon = np.concatenate([ends[:, :2], ends[::-1, 2:]], axis=0)
    a = _unit(apex - base_mid)
    basal_septal, basal_lateral = basal_hinges(polygon, base_mid, a)

    mids = (ends[:, :2] + ends[:, 2:]) / 2.0
    middle = ends[np.argsort((mids - base_mid) @ a)[len(ends) // 2]]
    mid_septal, mid_lateral = sorted([middle[:2], middle[2:]], key=lambda v: v[0])
    out = {
        "apex": apex, "base_mid": base_mid,
        "basal_septal": basal_septal, "basal_lateral": basal_lateral,
        "mid_septal": mid_septal, "mid_lateral": mid_lateral,
        "length": L, "diameters": diam, "volume": volume, "polygon": polygon,
        "chords": [(e[:2], e[2:]) for e in ends],
    }
    if shape is not None:
        region = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(region, [np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)], 1)
        out["region"] = region
        out["area"] = float(region.sum())
    return out


MIN_BEAT_S = 0.35      # shortest plausible cardiac cycle (~170 bpm)
DEFAULT_FPS = 50.0     # EchoNet-Dynamic clips are ~50 fps


def _median_filter(v: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    padded = np.pad(v, half, mode="edge")
    return np.array([np.median(padded[i:i + window]) for i in range(len(v))])


def _peaks(v: np.ndarray, min_dist: int) -> list[int]:
    """Local maxima at least `min_dist` frames apart, strongest first kept."""
    cand = [i for i in range(len(v))
            if v[i] == v[max(0, i - min_dist // 2):i + min_dist // 2 + 1].max()]
    kept: list[int] = []
    for i in sorted(cand, key=lambda i: v[i], reverse=True):
        if all(abs(i - k) >= min_dist for k in kept):
            kept.append(i)
    return sorted(kept)


def ef_from_volume_curve(volumes, fps: float | None = None, window: int = 5) -> dict | None:
    """EF from a per-frame volume curve, beat by beat.

    1. Frames without a segmentation (NaN) are interpolated, and a median filter
       of `window` frames removes single-frame segmentation glitches.
    2. ED frames are peaks of the curve at least MIN_BEAT_S apart and above the
       curve's median; the ES frame of a beat is the minimum before the next ED
       (or before the end of the clip, if at least one beat length remains).
    3. EF is computed per beat, and the median over beats is reported, as in
       EchoNet-Dynamic's beat-to-beat assessment, so one bad beat or glitch
       can't set the result on its own.

    Falls back to the clip's max/min when no complete beat is found.
    `ed_frame`/`es_frame` are the beat whose EF is closest to the median.
    """
    v = np.asarray(volumes, dtype=np.float64)
    ok = np.isfinite(v)
    if ok.sum() < max(window, len(v) // 2):
        return None
    idx = np.arange(len(v))
    v = np.interp(idx, idx[ok], v[ok])
    smooth = _median_filter(v, window)

    min_dist = max(3, int(round(MIN_BEAT_S * (fps or DEFAULT_FPS))))
    level = np.median(smooth)
    eds = [i for i in _peaks(smooth, min_dist) if smooth[i] > level]

    beats = []
    for j, ed in enumerate(eds):
        stop = eds[j + 1] if j + 1 < len(eds) else len(smooth)
        if stop - ed < min_dist:
            continue
        es = ed + int(np.argmin(smooth[ed:stop]))
        if smooth[ed] > 0 and es > ed:
            beats.append((ed, es, 100.0 * (smooth[ed] - smooth[es]) / smooth[ed]))

    if beats:
        ef = float(np.median([b[2] for b in beats]))
        ed, es, _ = min(beats, key=lambda b: abs(b[2] - ef))
        method = "beats"
    else:
        ed, es = int(np.argmax(smooth)), int(np.argmin(smooth))
        if smooth[ed] <= 0:
            return None
        ef = 100.0 * (smooth[ed] - smooth[es]) / smooth[ed]
        method = "minmax"
    return {"ef": ef, "edv": float(smooth[ed]), "esv": float(smooth[es]),
            "ed_frame": int(ed), "es_frame": int(es), "beats": beats,
            "method": method, "smoothed": smooth}


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    denom = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / denom) if denom else float("nan")
