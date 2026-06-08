import numpy as np


def normalize_frames(
    frames: np.ndarray,
    mean: float | None = None,
    std: float | None = None,
) -> np.ndarray:
    """Z-score normalisation per video.  Input shape (T, H, W)."""
    if mean is None:
        mean = frames.mean()
    if std is None:
        std = frames.std() + 1e-8
    return (frames.astype(np.float32) - mean) / std


def normalize_image(
    image: np.ndarray,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """ImageNet normalisation for 3‑channel image (H,W,3)."""
    img = image.astype(np.float32) / 255.0
    for c in range(3):
        img[..., c] = (img[..., c] - mean[c]) / std[c]
    return img
