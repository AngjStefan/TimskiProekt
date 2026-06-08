import cv2
import numpy as np


def get_lv_contour(mask: np.ndarray, min_area: int = 50):
    """Extract largest LV contour from binary mask."""
    mask_bin = (mask > 0.3).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None
    h, w = mask.shape[:2]
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        touches_border = np.any(c[:, 0, 1] == 0) or np.any(c[:, 0, 1] == h - 1) or \
                         np.any(c[:, 0, 0] == 0) or np.any(c[:, 0, 0] == w - 1)
        if touches_border:
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


def get_key_points(contour) -> dict:
    """Extract key points: apex (bottom-most), basal points (left/right top)."""
    if contour is None:
        return {"apex": None, "basal_left": None, "basal_right": None, "centroid": None}

    centroid = get_centroid(contour)
    pts = contour.squeeze(1)
    if pts.ndim != 2 or len(pts) < 3:
        return {"apex": None, "basal_left": None, "basal_right": None, "centroid": centroid}

    # Apex: bottom-most point (largest y)
    min_y, max_y = pts[:, 1].min(), pts[:, 1].max()
    apex = tuple(pts[pts[:, 1].argmax()])

    # Basal: top-most points (left and right)
    y_range = max_y - min_y
    top_mask = pts[:, 1] < min_y + y_range * 0.2
    top_pts = pts[top_mask]
    if len(top_pts) > 1:
        basal_left = tuple(top_pts[top_pts[:, 0].argmin()])
        basal_right = tuple(top_pts[top_pts[:, 0].argmax()])
    else:
        basal_left = tuple(pts[pts[:, 0].argmin()])
        basal_right = tuple(pts[pts[:, 0].argmax()])

    return {
        "apex": apex,
        "basal_left": basal_left,
        "basal_right": basal_right,
        "centroid": centroid,
    }


def _clamp(cx: int, cy: int, h: int, w: int):
    return max(0, min(w - 1, cx)), max(0, min(h - 1, cy))


def draw_contour_and_points(
    image: np.ndarray,
    contour,
    keypoints: dict | None = None,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw contour and key points on image. Image is (H,W,3) uint8."""
    overlay = image.copy()
    h, w = overlay.shape[:2]
    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, color, thickness)

    if keypoints:
        for name, pt in keypoints.items():
            if pt is None:
                continue
            cx, cy = _clamp(int(pt[0]), int(pt[1]), h, w)
            if name == "apex":
                cv2.circle(overlay, (cx, cy), 6, (255, 255, 255), 2)
                cv2.circle(overlay, (cx, cy), 6, (0, 0, 255), -1)
                cv2.putText(overlay, "A", (cx + 8, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(overlay, "A", (cx + 8, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1)
            elif "basal" in name:
                label = "BL" if "left" in name else "BR"
                cv2.circle(overlay, (cx, cy), 6, (255, 255, 255), 2)
                cv2.circle(overlay, (cx, cy), 6, (255, 0, 0), -1)
                cv2.putText(overlay, label, (cx + 8, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(overlay, label, (cx + 8, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
            elif name == "centroid":
                cv2.circle(overlay, (cx, cy), 6, (255, 255, 255), 2)
                cv2.circle(overlay, (cx, cy), 6, (0, 255, 255), -1)
                cv2.putText(overlay, "C", (cx + 8, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(overlay, "C", (cx + 8, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)
    return overlay


def process_mask(mask: np.ndarray, image: np.ndarray) -> dict:
    """Full pipeline: mask → contour → keypoints → overlay image."""
    contour, area = get_lv_contour(mask)
    keypoints = get_key_points(contour)
    overlay = draw_contour_and_points(image, contour, keypoints)
    return {
        "contour": contour,
        "area_px": area,
        "keypoints": keypoints,
        "overlay": overlay,
    }
