import cv2
import numpy as np

from src.postprocessing.measurements import (
    N_DISKS, lv_landmarks, measure_mask, method_of_disks, region_from_contour,
)


def get_lv_contour(mask: np.ndarray, min_area: int = 50):
    """Extract LV contour from binary mask with geometric constraints."""
    mask_bin = (mask > 0.5).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None

    h, w = mask.shape[:2]
    cx_img, cy_img = w // 2, h // 2
    valid = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        touches_border = (
            np.any(c[:, 0, 1] == 0) or np.any(c[:, 0, 1] == h - 1) or
            np.any(c[:, 0, 0] == 0) or np.any(c[:, 0, 0] == w - 1)
        )
        if touches_border:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        dist_from_center = np.sqrt((cx - cx_img) ** 2 + (cy - cy_img) ** 2)
        if dist_from_center > w * 0.45:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        aspect_ratio = bw / bh if bh > 0 else 0
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            continue

        valid.append((c, area))

    if not valid:
        return None, None

    valid.sort(key=lambda x: x[1], reverse=True)
    return valid[0][0], valid[0][1]


def get_centroid(contour) -> tuple[float, float] | None:
    """Centroid (cx, cy) of the LV contour."""
    if contour is None:
        return None
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return M["m10"] / M["m00"], M["m01"] / M["m00"]


KEYPOINT_NAMES = ("apex", "basal_septal", "basal_lateral", "mid_septal", "mid_lateral")


def get_key_points(contour, measurement: dict | None = None) -> dict:
    """Apex, mitral annulus hinges (basal septal/lateral) and mid-cavity walls.

    Derived from the LV long axis (see src/postprocessing/measurements.py): the
    apex and annulus come from the major axis of the region, and the mid points
    are the ends of the middle disk chord. `centroid` and `base_mid` (annulus
    midpoint) are included for drawing and measurements.
    """
    empty = {name: None for name in (*KEYPOINT_NAMES, "base_mid")}
    empty["centroid"] = get_centroid(contour)
    if contour is None:
        return empty
    if measurement is None:
        region = region_from_contour(contour)
        lm = lv_landmarks(contour, region)
        if lm is None:
            return empty
        disks = method_of_disks(region, lm["apex"], lm["base_mid"])
        if disks is None:
            return empty
        mid_septal, mid_lateral = sorted(disks["chords"][N_DISKS // 2], key=lambda p: p[0])
        measurement = {**lm, "mid_septal": mid_septal, "mid_lateral": mid_lateral}

    out = {name: tuple(float(v) for v in measurement[name]) for name in (*KEYPOINT_NAMES, "base_mid")}
    out["centroid"] = empty["centroid"]
    return out


def _clamp(cx: int, cy: int, h: int, w: int):
    return max(0, min(w - 1, cx)), max(0, min(h - 1, cy))


_POINT_STYLE = {
    "apex": ("A", (255, 60, 60)),
    "basal_septal": ("BS", (70, 140, 255)),
    "basal_lateral": ("BL", (70, 140, 255)),
    "mid_septal": ("MS", (255, 220, 0)),
    "mid_lateral": ("ML", (255, 220, 0)),
}


def draw_contour_and_points(
    image: np.ndarray,
    contour,
    keypoints: dict | None = None,
    color: tuple = (0, 255, 0),
    thickness: int = 1,
    chords: list | None = None,
) -> np.ndarray:
    """Draw contour, long axis, disk chords and landmarks on an (H,W,3) uint8 image."""
    overlay = image.copy()
    h, w = overlay.shape[:2]
    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, color, thickness, lineType=cv2.LINE_AA)

    def ip(pt):
        return _clamp(int(round(pt[0])), int(round(pt[1])), h, w)

    if chords:
        layer = overlay.copy()
        for p1, p2 in chords:
            cv2.line(layer, ip(p1), ip(p2), (0, 200, 255), 1, lineType=cv2.LINE_AA)
        overlay = cv2.addWeighted(layer, 0.55, overlay, 0.45, 0)

    if keypoints and keypoints.get("apex") is not None and keypoints.get("base_mid") is not None:
        cv2.line(overlay, ip(keypoints["apex"]), ip(keypoints["base_mid"]), (255, 255, 255), 1, lineType=cv2.LINE_AA)
        if keypoints.get("basal_septal") is not None and keypoints.get("basal_lateral") is not None:
            cv2.line(overlay, ip(keypoints["basal_septal"]), ip(keypoints["basal_lateral"]),
                     (70, 140, 255), 1, lineType=cv2.LINE_AA)

    for name, (label, lcolor) in _POINT_STYLE.items():
        pt = (keypoints or {}).get(name)
        if pt is None:
            continue
        c = ip(pt)
        cv2.circle(overlay, c, 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, c, 2, lcolor, -1, lineType=cv2.LINE_AA)
        # labels outside the cavity: septal to the left, lateral right, apex above
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
        if name.endswith("septal"):
            org = (c[0] - tw - 5, c[1] + th // 2)
        elif name.endswith("lateral"):
            org = (c[0] + 5, c[1] + th // 2)
        else:
            org = (c[0] - tw // 2, c[1] - 6)
        cv2.putText(overlay, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (0, 0, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(overlay, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    lcolor, 1, lineType=cv2.LINE_AA)

    return overlay


def process_mask(mask: np.ndarray, image: np.ndarray) -> dict:
    """Full pipeline: mask → contour → landmarks + method of disks → overlay image.

    `measurement` holds area (px, from the filled contour), long-axis length,
    disk diameters and volume, all in the mask's pixel grid.
    """
    contour, area = get_lv_contour(mask)
    measurement = measure_mask(mask, contour) if contour is not None else None
    keypoints = get_key_points(contour, measurement)
    overlay = draw_contour_and_points(
        image, contour, keypoints,
        chords=measurement["chords"] if measurement else None,
    )
    return {
        "contour": contour,
        "area_px": area,
        "keypoints": keypoints,
        "measurement": measurement,
        "overlay": overlay,
    }
