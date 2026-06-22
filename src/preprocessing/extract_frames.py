import zipfile
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import io
import tempfile
import os

from src.config import RAW_VIDEOS, FRAME_SIZE, TARGET_FPS, RAW_DIR


def extract_frames_from_avi(
    avi_bytes,
    target_size=FRAME_SIZE,
    max_frames=None,
):
    fd, tmp_path = tempfile.mkstemp(
        suffix=".avi"
    )

    try:
        os.write(fd, avi_bytes)
        os.close(fd)

        cap = cv2.VideoCapture(tmp_path)

        frames = []

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            gray = (
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY,
                )
                if frame.ndim == 3
                else frame
            )

            gray = cv2.resize(
                gray,
                (
                    target_size,
                    target_size,
                ),
                interpolation=cv2.INTER_AREA,
            )

            frames.append(gray)

            if (
                max_frames is not None
                and len(frames) >= max_frames
            ):
                break

        cap.release()

        return np.stack(
            frames
        ).astype(np.uint8)

    finally:
        os.unlink(tmp_path)


def preprocess_all(
    zip_path: Path,
    dest_dir: Path,
    target_size: int = FRAME_SIZE,
    video_ids: int | None = None,
) -> list[str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(str(zip_path), "r")
    video_entries = sorted(n for n in z.namelist() if n.endswith(".avi"))

    if video_ids:
        allowed = set(video_ids)

        video_entries = [
            x
            for x in video_entries
            if Path(x).stem in allowed
        ]

    processed = []
    for entry in tqdm(video_entries, desc="Extracting frames"):
        fname = Path(entry).name
        out_path = dest_dir / fname.replace(".avi", ".npy")
        raw = z.read(entry)
        frames = extract_frames_from_avi(raw, target_size)
        np.save(str(out_path), frames)
        processed.append(fname.replace(".avi", ""))
    z.close()
    print(f"Extracted {len(processed)} videos to {dest_dir}")
    return processed


def extract_single_video(
    avi_path: str | Path,
    target_size: int = FRAME_SIZE,
    max_frames: int | None = None,
) -> np.ndarray:
    with open(avi_path, "rb") as f:
        return extract_frames_from_avi(f.read(), target_size, max_frames)
