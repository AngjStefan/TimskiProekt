import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.postprocessing.contours import draw_contour_and_points


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    overlay = image.copy()
    mask_bool = mask > 0.5
    for c in range(3):
        overlay[..., c] = np.where(
            mask_bool,
            overlay[..., c] * (1 - alpha) + color[c] * alpha,
            overlay[..., c],
        )
    return overlay


def create_comparison_grid(
    frames: list[np.ndarray],
    masks: list[np.ndarray | None],
    contours: list[np.ndarray | None],
    titles: list[str],
    save_path: str | None = None,
) -> np.ndarray | None:
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for i in range(n):
        img = frames[i]
        axes[i].imshow(img, cmap="gray")
        if masks[i] is not None:
            axes[i].imshow(masks[i], cmap="jet", alpha=0.4)
        if contours[i] is not None:
            contour_pts = contours[i].squeeze(1)
            axes[i].plot(contour_pts[:, 0], contour_pts[:, 1], "g-", linewidth=2)
        axes[i].set_title(titles[i])
        axes[i].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return None

    fig.canvas.draw()
    grid_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    grid_img = grid_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return grid_img


def visualize_prediction(
    frame: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray | None = None,
    contour_info: dict | None = None,
) -> plt.Figure:
    n_panels = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    axes[0].imshow(frame, cmap="gray")
    axes[0].set_title("Original Frame")
    axes[0].axis("off")

    axes[1].imshow(pred_mask, cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    overlay = overlay_mask(frame, pred_mask)
    if contour_info and contour_info.get("contour") is not None:
        overlay = draw_contour_and_points(
            overlay, contour_info["contour"], contour_info.get("keypoints")
        )
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    if gt_mask is not None:
        axes[3].imshow(gt_mask, cmap="gray")
        axes[3].set_title("Ground Truth Mask")
        axes[3].axis("off")

    plt.tight_layout()
    return fig
