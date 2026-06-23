import cv2
import numpy as np


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


def get_key_points(contour) -> dict:
    """Extract 5 key points: apex, basal septal, basal lateral, mid septal, mid lateral."""
    if contour is None:
        return {
            "apex": None,
            "basal_septal": None,
            "basal_lateral": None,
            "mid_septal": None,
            "mid_lateral": None,
            "centroid": None,
        }

    centroid = get_centroid(contour)
    pts = contour.squeeze(1)
    if pts.ndim != 2 or len(pts) < 3:
        return {
            "apex": None,
            "basal_septal": None,
            "basal_lateral": None,
            "mid_septal": None,
            "mid_lateral": None,
            "centroid": centroid,
        }

    min_y, max_y = pts[:, 1].min(), pts[:, 1].max()
    y_range = max_y - min_y

    apex = tuple(pts[pts[:, 1].argmax()])

    top_mask = pts[:, 1] < min_y + y_range * 0.2
    top_pts = pts[top_mask]
    if len(top_pts) > 1:
        basal_septal = tuple(top_pts[top_pts[:, 0].argmin()])
        basal_lateral = tuple(top_pts[top_pts[:, 0].argmax()])
    else:
        basal_septal = tuple(pts[pts[:, 0].argmin()])
        basal_lateral = tuple(pts[pts[:, 0].argmax()])

    mid_mask = (pts[:, 1] >= min_y + y_range * 0.4) & (pts[:, 1] <= min_y + y_range * 0.6)
    mid_pts = pts[mid_mask]
    if len(mid_pts) > 1:
        mid_septal = tuple(mid_pts[mid_pts[:, 0].argmin()])
        mid_lateral = tuple(mid_pts[mid_pts[:, 0].argmax()])
    else:
        mid_septal = basal_septal
        mid_lateral = basal_lateral

    return {
        "apex": apex,
        "basal_septal": basal_septal,
        "basal_lateral": basal_lateral,
        "mid_septal": mid_septal,
        "mid_lateral": mid_lateral,
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
                label, lcolor = "A", (0, 0, 255)
            elif name == "basal_septal":
                cv2.circle(overlay, (cx, cy), 6, (255, 255, 255), 2)
                cv2.circle(overlay, (cx, cy), 6, (255, 0, 0), -1)
                label, lcolor = "BS", (255, 0, 0)
            elif name == "basal_lateral":
                cv2.circle(overlay, (cx, cy), 6, (255, 255, 255), 2)
                cv2.circle(overlay, (cx, cy), 6, (255, 0, 0), -1)
                label, lcolor = "BL", (255, 0, 0)
            elif name == "mid_septal":
                cv2.circle(overlay, (cx, cy), 6, (255, 255, 255), 2)
                cv2.circle(overlay, (cx, cy), 6, (0, 255, 255), -1)
                label, lcolor = "MS", (0, 255, 255)
            elif name == "mid_lateral":
                cv2.circle(overlay, (cx, cy), 6, (255, 255, 255), 2)
                cv2.circle(overlay, (cx, cy), 6, (0, 255, 255), -1)
                label, lcolor = "ML", (0, 255, 255)
            elif name == "centroid":
                cv2.circle(overlay, (cx, cy), 6, (255, 255, 255), 2)
                cv2.circle(overlay, (cx, cy), 6, (0, 200, 0), -1)
                label, lcolor = "C", (0, 200, 0)
            else:
                continue

            cv2.putText(
                overlay, label, (cx + 8, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )
            cv2.putText(
                overlay, label, (cx + 8, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, lcolor, 1,
            )

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
