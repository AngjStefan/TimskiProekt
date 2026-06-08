import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import sys

from src.config import (
    MODELS_DIR, REGRESSION_LR, REGRESSION_WEIGHT_DECAY,
    REGRESSION_EPOCHS, REGRESSION_BATCH_SIZE, REGRESSION_PATIENCE,
    REGRESSION_BACKBONE, DEVICE, SUBSET_SIZE,
)
from src.models.regression import EchoResNet
from src.preprocessing.dataset import get_regression_loaders


class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for videos, targets in tqdm(loader, desc="Train", leave=False):
        videos, targets = videos.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * videos.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for videos, targets in loader:
        videos, targets = videos.to(device), targets.to(device)
        outputs = model(videos)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * videos.size(0)
    return total_loss / len(loader.dataset)


def main(backbone: str | None = None, subset_size: int | None = None,
         file_list: str | None = None):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    bb = backbone or REGRESSION_BACKBONE
    ss = subset_size if subset_size is not None else SUBSET_SIZE
    fl = Path(file_list) if file_list else None

    train_loader, val_loader, test_loader = get_regression_loaders(
        batch_size=REGRESSION_BATCH_SIZE,
        subset_size=ss,
        file_list=fl,
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")

    if len(train_loader.dataset) == 0:
        print("ERROR: No training data found. Run 'python run_pipeline.py' first.")
        sys.exit(1)

    model = EchoResNet(backbone=bb, pretrained=True).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=REGRESSION_LR, weight_decay=REGRESSION_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    stopper = EarlyStopping(patience=REGRESSION_PATIENCE)

    best_val = float("inf")
    for epoch in range(1, REGRESSION_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        if len(val_loader) > 0:
            val_loss = validate(model, val_loader, criterion, device)
        else:
            val_loss = train_loss
        scheduler.step(val_loss)

        print(f"Epoch {epoch:2d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = MODELS_DIR / f"{bb}_ef.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"  -> Saved {ckpt}")

        if stopper(val_loss):
            print("Early stopping")
            break

    # final test
    ckpt = MODELS_DIR / f"{bb}_ef.pth"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt))
    if len(test_loader) > 0:
        test_loss = validate(model, test_loader, criterion, device)
        print(f"Test MSE: {test_loss:.6f}  (RMSE: {test_loss**0.5:.4f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet34")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--file-list", type=str, default=None)
    args = parser.parse_args()
    main(backbone=args.backbone, subset_size=args.subset,
         file_list=args.file_list)
