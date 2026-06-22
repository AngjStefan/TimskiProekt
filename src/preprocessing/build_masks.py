import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm

from src.config import VOLUME_TRACINGS, FRAME_SIZE_SEG, PROCESSED_DIR


def tracing_to_mask(
    points: list[tuple[float, float]],
    height: int = FRAME_SIZE_SEG,
    width: int = FRAME_SIZE_SEG,
) -> np.ndarray:
    """Convert list of (x, y) contour points to binary mask."""
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def build_all_masks(
    tracings_csv: Path = VOLUME_TRACINGS,
    out_dir: Path | None = None,
    frame_size: int = FRAME_SIZE_SEG,
    video_ids: list[str] | None = None,
) -> None:
    out = Path(out_dir or PROCESSED_DIR / "masks")
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tracings_csv)
    if video_ids is not None:
        video_set = set(video_ids)
        df = df[df["FileName"].str.replace(".avi", "").isin(video_set)]
    grouped = df.groupby(["FileName", "Frame"])

    for (fname, frame), grp in tqdm(grouped, desc="Building masks"):
        orig_h = 112
        orig_w = 112

        scale = frame_size / 112.0

        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)

        pad_y = (frame_size - new_h) // 2
        pad_x = (frame_size - new_w) // 2

        left = np.stack([
            grp["X1"].values * scale + pad_x,
            grp["Y1"].values * scale + pad_y,
        ], axis=1)

        right = np.stack([
            grp["X2"].values[::-1] * scale + pad_x,
            grp["Y2"].values[::-1] * scale + pad_y,
        ], axis=1)

        contour = np.concatenate(
            [left, right],
            axis=0,
        )

        mask = tracing_to_mask(
            contour.tolist(),
            frame_size,
            frame_size,
        )

        out_name = f"{Path(fname).stem}_frame{int(frame)}.npy"

        np.save(
            out / out_name,
            mask,
        )

    print(f"Saved {len(grouped)} masks to {out}")


def load_mask_for_video(
    video_id: str,
    frame: int,
    mask_dir: Path | None = None,
) -> np.ndarray | None:
    md = Path(mask_dir or PROCESSED_DIR / "masks")
    p = md / f"{video_id}_frame{frame}.npy"
    return np.load(str(p)) if p.exists() else None
