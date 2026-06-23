import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from src.config import (
    PROCESSED_DIR, FRAME_SIZE, FRAME_SIZE_SEG, N_FRAMES,
    SEG_BATCH_SIZE, REGRESSION_BATCH_SIZE, FILE_LIST,
)
from src.preprocessing.normalize import normalize_frames, normalize_image


class EchoVideoDataset(Dataset):
    """Regression dataset: video → EF (& optionally ESV, EDV)."""

    def __init__(
        self,
        split: str = "TRAIN",
        frames_dir: Path | None = None,
        file_list: Path | None = None,
        n_frames: int = N_FRAMES,
        frame_size: int = FRAME_SIZE,
        predict_ef: bool = True,
        subset_size: int | None = None,
    ):
        self.n_frames = n_frames
        self.frame_size = frame_size
        df = pd.read_csv(file_list or FILE_LIST)
        self.df = df[df["Split"] == split].reset_index(drop=True)
        if subset_size is not None:
            self.df = self.df.iloc[:subset_size]
        self.frames_dir = Path(frames_dir or PROCESSED_DIR / "frames")
        self.predict_ef = predict_ef

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        file_id = row["FileName"]

        frames_path = self.frames_dir / f"{file_id}.npy"
        frames = np.load(str(frames_path)).astype(np.float32)

        T, H, W = frames.shape
        frames = normalize_frames(frames)

        if self.n_frames and T > self.n_frames:
            idxs = np.linspace(0, T - 1, self.n_frames, dtype=int)
            frames = frames[idxs]
        elif T < self.n_frames:
            pad = np.tile(frames[-1:], (self.n_frames - T, 1, 1))
            frames = np.concatenate([frames, pad], axis=0)

        if H != self.frame_size:
            resized = []
            for f in frames:
                resized.append(cv2.resize(f, (self.frame_size, self.frame_size)))
            frames = np.stack(resized)

        frames = np.stack([frames, frames, frames], axis=-1)
        video = torch.from_numpy(frames).permute(0, 3, 1, 2).float()

        ef = torch.tensor(row["EF"], dtype=torch.float32) / 100.0
        esv = torch.tensor(row["ESV"], dtype=torch.float32)
        edv = torch.tensor(row["EDV"], dtype=torch.float32)

        if self.predict_ef:
            return video, ef
        return video, ef, esv, edv

    def get_video_ids(self):
        return self.df["FileName"].tolist()

    def get_labels(self):
        return self.df["EF"].values / 100.0


class EchoSegmentationDataset(Dataset):
    """Segmentation dataset: single frame → binary mask."""

    def __init__(
        self,
        split: str = "TRAIN",
        frames_dir: Path | None = None,
        masks_dir: Path | None = None,
        frame_size: int = FRAME_SIZE_SEG,
        file_list: Path | None = None,
        subset_size: int | None = None,
    ):
        self.frame_size = frame_size
        df = pd.read_csv(file_list or FILE_LIST)
        self.df = df[df["Split"] == split].reset_index(drop=True)
        if subset_size is not None:
            self.df = self.df.iloc[:subset_size]
        self.frames_dir = Path(frames_dir or PROCESSED_DIR / "frames")
        self.masks_dir = Path(masks_dir or PROCESSED_DIR / "masks")

        import os
        mask_files = sorted(os.listdir(self.masks_dir))

        allowed = set(
            self.df["FileName"]
            .astype(str)
            .str.replace(".avi", "", regex=False)
        )

        self.samples = []

        for mf in mask_files:

            parts = mf.replace(".npy", "").split("_frame")

            if len(parts) != 2:
                continue

            vid = parts[0]
            fr = int(parts[1])

            if vid in allowed:
                self.samples.append((vid, fr))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        vid, frame_n = self.samples[idx]

        frames = np.load(str(self.frames_dir / f"{vid}.npy"))
        mask = np.load(str(self.masks_dir / f"{vid}_frame{frame_n}.npy"))

        frame = frames[frame_n] if frame_n < len(frames) else frames[-1]
        frame = cv2.resize(frame, (self.frame_size, self.frame_size))
        frame = frame.astype(np.float32)
        frame = np.stack([frame] * 3, axis=-1)
        frame = normalize_image(frame)

        mask = cv2.resize(mask, (self.frame_size, self.frame_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.float32)

        image_t = torch.from_numpy(frame).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return image_t, mask_t


def get_regression_loaders(
    batch_size: int = REGRESSION_BATCH_SIZE,
    n_frames: int = N_FRAMES,
    subset_size: int | None = None,
    file_list: Path | None = None,
):
    train_ds = EchoVideoDataset("TRAIN", n_frames=n_frames,
                                subset_size=subset_size, file_list=file_list)
    val_ds = EchoVideoDataset("VAL", n_frames=n_frames,
                              subset_size=subset_size if subset_size else None,
                              file_list=file_list)
    test_ds = EchoVideoDataset("TEST", n_frames=n_frames,
                               subset_size=subset_size if subset_size else None,
                               file_list=file_list)
    return (
        DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(val_ds, batch_size, shuffle=False, num_workers=2, pin_memory=True),
        DataLoader(test_ds, batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )


def get_segmentation_loaders(
    batch_size: int = SEG_BATCH_SIZE,
    subset_size: int | None = None,
    file_list: Path | None = None,
):
    train_ds = EchoSegmentationDataset("TRAIN",
                                       subset_size=subset_size, file_list=file_list)
    val_ds = EchoSegmentationDataset("VAL",
                                     subset_size=subset_size if subset_size else None,
                                     file_list=file_list)
    test_ds = EchoSegmentationDataset("TEST",
                                      subset_size=subset_size if subset_size else None,
                                      file_list=file_list)
    return (
        DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(val_ds, batch_size, shuffle=False, num_workers=2, pin_memory=True),
        DataLoader(test_ds, batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )
