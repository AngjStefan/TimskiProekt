import torch
import torch.nn as nn
from torchvision import models


class EchoResNet(nn.Module):
    """ResNet18/34 for EF regression from echo video frames."""

    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
        in_channels: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        if backbone == "resnet18":
            self.cnn = models.resnet18(weights=weights)
        elif backbone == "resnet34":
            self.cnn = models.resnet34(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if in_channels != 3:
            old = self.cnn.conv1
            self.cnn.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            with torch.no_grad():
                self.cnn.conv1.weight[:, : min(3, in_channels)] = old.weight.mean(
                    dim=1, keepdim=True
                )

        feat_dim = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        self.temporal = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        self.ef_head = nn.Sequential(
            nn.Dropout(dropout),

            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(256, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):

        B, T, C, H, W = x.shape

        x = x.view(
            B * T,
            C,
            H,
            W,
        )

        feats = self.cnn(x)

        feats = feats.view(
            B,
            T,
            -1,
        )

        weights = self.temporal(
            feats
        )

        weights = torch.softmax(
            weights,
            dim=1,
        )

        feats = (
                feats * weights
        ).sum(
            dim=1
        )

        return self.ef_head(
            feats
        ).squeeze(-1)


class EchoAreaResNet(nn.Module):
    """Predict ESV and EDV (LV volumes) from video."""

    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
    ):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        if backbone == "resnet18":
            self.cnn = models.resnet18(weights=weights)
        else:
            self.cnn = models.resnet34(weights=weights)

        feat_dim = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        self.volume_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x).view(B, T, -1).mean(dim=1)
        vols = self.volume_head(feats)
        return vols[:, 0], vols[:, 1]  # ESV, EDV
