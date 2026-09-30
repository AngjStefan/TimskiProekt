"""Shared model loading + inference, used by the Streamlit app and evaluation."""
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import MODELS_DIR, FRAME_SIZE, FRAME_SIZE_SEG, N_FRAMES, SEG_THRESHOLD
from src.models.regression import EchoResNet
from src.models.segmentation import UNet
from src.preprocessing.normalize import normalize_frames, normalize_image


def load_unet(device, ckpt: Path = MODELS_DIR / "unet_lv.pth"):
    """Returns (model, loaded). loaded=False means untrained weights."""
    model = UNet().to(device)
    loaded = ckpt.exists()
    if loaded:
        model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.eval(), loaded


def load_regression(device, backbone: str = "resnet18", ckpt: Path | None = None):
    """Returns (model, loaded). loaded=False means untrained EF head."""
    ckpt = ckpt or MODELS_DIR / f"{backbone}_ef.pth"
    model = EchoResNet(backbone=backbone).to(device)
    loaded = ckpt.exists()
    if loaded:
        model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.eval(), loaded


def _largest_component(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)
    return mask


@torch.no_grad()
def segment_frames(
    model,
    frames_gray: np.ndarray,
    device,
    threshold: float = SEG_THRESHOLD,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """U-Net on every frame. frames_gray: (T,H,W) uint8.

    Returns (masks, probs) at FRAME_SIZE_SEG: masks (T,S,S) uint8 keeping the
    largest component, probs (T,S,S) float32 sigmoid output.
    """
    frames = np.stack([
        cv2.resize(f, (FRAME_SIZE_SEG, FRAME_SIZE_SEG)) if f.shape[0] != FRAME_SIZE_SEG else f
        for f in frames_gray
    ])
    probs = []
    for i in range(0, len(frames), batch_size):
        chunk = frames[i:i + batch_size].astype(np.float32)
        x = np.stack([normalize_image(np.stack([f] * 3, axis=-1)) for f in chunk])
        x = torch.from_numpy(x).permute(0, 3, 1, 2).float().to(device)
        probs.append(torch.sigmoid(model(x)).squeeze(1).cpu().numpy())
    probs = np.concatenate(probs).astype(np.float32)
    masks = np.stack([_largest_component((p > threshold).astype(np.uint8)) for p in probs])
    return masks, probs


@torch.no_grad()
def predict_ef(model, frames_gray: np.ndarray, device, n_frames: int = N_FRAMES) -> float:
    """EF (%) from the regression model, sampling n_frames evenly like training."""
    T = len(frames_gray)
    idxs = np.linspace(0, T - 1, n_frames, dtype=int)
    sampled = frames_gray[idxs]
    if sampled.shape[1] != FRAME_SIZE:
        sampled = np.stack([cv2.resize(f, (FRAME_SIZE, FRAME_SIZE)) for f in sampled])
    sampled = normalize_frames(sampled)
    sampled = np.stack([sampled] * 3, axis=-1)
    x = torch.from_numpy(sampled).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
    return float(np.clip(model(x).item(), 0.0, 1.0) * 100.0)
