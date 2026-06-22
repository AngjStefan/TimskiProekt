import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import sys

from src.config import (
    MODELS_DIR, SEG_LR, SEG_WEIGHT_DECAY, SEG_EPOCHS,
    SEG_BATCH_SIZE, SEG_PATIENCE, DEVICE, SUBSET_SIZE,
)
from src.models.segmentation import UNet
from src.preprocessing.dataset import get_segmentation_loaders
from src.training.train_regression import EarlyStopping


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    bce: nn.Module,
    dice: DiceLoss,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="Train", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = 0.3 * bce(logits, masks) + 0.7 * dice(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    bce: nn.Module,
    dice: DiceLoss,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = bce(logits, masks) + dice(logits, masks)
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def main(subset_size: int | None = None, file_list: str | None = None):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ss = subset_size if subset_size is not None else SUBSET_SIZE
    fl = Path(file_list) if file_list else None
    train_loader, val_loader, test_loader = get_segmentation_loaders(
        batch_size=SEG_BATCH_SIZE,
        subset_size=ss,
        file_list=fl,
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")

    if len(train_loader.dataset) == 0:
        print("ERROR: No training data found. Run 'python run_pipeline.py' first.")
        sys.exit(1)

    model = UNet(
        n_channels=3,
        n_classes=1,
        pretrained=True,
    ).to(device)
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=SEG_LR, weight_decay=SEG_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7
    )
    stopper = EarlyStopping(patience=SEG_PATIENCE)

    best_val = float("inf")
    for epoch in range(1, SEG_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, bce, dice, device)
        if len(val_loader) > 0:
            val_loss = validate(model, val_loader, bce, dice, device)
        else:
            val_loss = train_loss
        scheduler.step(val_loss)

        print(f"Epoch {epoch:2d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = MODELS_DIR / "unet_lv.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"  -> Saved {ckpt}")

        if stopper(val_loss):
            print("Early stopping")
            break

    # test
    ckpt = MODELS_DIR / "unet_lv.pth"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt))
    if len(test_loader) > 0:
        test_loss = validate(model, test_loader, bce, dice, device)
        print(f"Test loss (BCE+Dice): {test_loss:.6f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--file-list", type=str, default=None)
    args = parser.parse_args()
    main(subset_size=args.subset, file_list=args.file_list)
