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
        xs = grp["X1"].values
        ys = grp["Y1"].values
        scale = frame_size / 112.0
        pts = list(zip(xs * scale, ys * scale))
        mask = tracing_to_mask(pts, frame_size, frame_size)
        out_name = f"{Path(fname).stem}_frame{int(frame)}.npy"
        np.save(str(out / out_name), mask)

    print(f"Saved {len(grouped)} masks to {out}")


def load_mask_for_video(
    video_id: str,
    frame: int,
    mask_dir: Path | None = None,
) -> np.ndarray | None:
    md = Path(mask_dir or PROCESSED_DIR / "masks")
    p = md / f"{video_id}_frame{frame}.npy"
    return np.load(str(p)) if p.exists() else None
