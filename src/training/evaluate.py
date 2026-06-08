import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.config import MODELS_DIR, REGRESSION_BACKBONE, DEVICE, PROCESSED_DIR, FILE_LIST
from src.models.regression import EchoResNet
from src.models.segmentation import UNet
from src.preprocessing.dataset import EchoVideoDataset, EchoSegmentationDataset


@torch.no_grad()
def evaluate_regression(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds, targets = [], []
    for videos, labels in tqdm(loader, desc="Eval regression"):
        videos = videos.to(device)
        outputs = model(videos).cpu().numpy()
        preds.append(outputs)
        targets.append(labels.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    mae = np.abs(preds - targets).mean()
    rmse = np.sqrt(((preds - targets) ** 2).mean())
    r2 = 1 - ((targets - preds) ** 2).sum() / ((targets - targets.mean()) ** 2).sum()
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}, preds, targets


@torch.no_grad()
def evaluate_segmentation(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    dice_scores, ious = [], []
    for images, masks in tqdm(loader, desc="Eval segmentation"):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        inter = (preds * masks).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
        dice = (2 * inter + 1e-6) / (union + 1e-6)
        iou = inter / (union - inter + 1e-6)

        dice_scores.append(dice.cpu().numpy())
        ious.append(iou.cpu().numpy())

    dice_scores = np.concatenate(dice_scores)
    ious = np.concatenate(ious)
    return {
        "Dice": float(dice_scores.mean()),
        "IoU": float(ious.mean()),
    }, dice_scores, ious


def print_comparison_table(results: dict):
    print("\n" + "=" * 50)
    print(f"{'Model':<20} {'Metric':<10} {'Value':<10}")
    print("-" * 50)
    for model_name, metrics in results.items():
        for metric, value in metrics.items():
            print(f"{model_name:<20} {metric:<10} {value:<10.4f}")
    print("=" * 50)


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    results = {}

    # ---- Regression models ----
    for backbone in ["resnet18", "resnet34"]:
        print(f"\n--- Evaluating {backbone} ---")
        model = EchoResNet(backbone=backbone).to(device)
        ckpt = MODELS_DIR / f"{backbone}_ef.pth"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device))
        else:
            print(f"  Checkpoint {ckpt} not found, using untrained model")
        _, test_loader, _ = get_loaders_for_eval(backbone)
        metrics, _, _ = evaluate_regression(model, test_loader, device)
        results[backbone] = metrics

    # ---- U-Net ----
    print("\n--- Evaluating U-Net ---")
    model = UNet().to(device)
    ckpt = MODELS_DIR / "unet_lv.pth"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        print(f"  Checkpoint {ckpt} not found, using untrained model")
    _, _, test_loader = get_seg_loaders_for_eval()
    metrics, _, _ = evaluate_segmentation(model, test_loader, device)
    results["U-Net"] = metrics

    print_comparison_table(results)
    return results


def get_loaders_for_eval(backbone: str, batch_size: int = 32):
    ds_test = EchoVideoDataset("TEST", n_frames=32)
    loader_test = DataLoader(ds_test, batch_size, shuffle=False, num_workers=2)
    ds_val = EchoVideoDataset("VAL", n_frames=32)
    loader_val = DataLoader(ds_val, batch_size, shuffle=False, num_workers=2)
    ds_train = EchoVideoDataset("TRAIN", n_frames=32)
    loader_train = DataLoader(ds_train, batch_size, shuffle=True, num_workers=2)
    return loader_train, loader_val, loader_test


def get_seg_loaders_for_eval(batch_size: int = 32):
    ds_test = EchoSegmentationDataset("TEST")
    loader_test = DataLoader(ds_test, batch_size, shuffle=False, num_workers=2)
    ds_val = EchoSegmentationDataset("VAL")
    loader_val = DataLoader(ds_val, batch_size, shuffle=False, num_workers=2)
    ds_train = EchoSegmentationDataset("TRAIN")
    loader_train = DataLoader(ds_train, batch_size, shuffle=True, num_workers=2)
    return loader_train, loader_val, loader_test


if __name__ == "__main__":
    main()
