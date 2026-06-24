import torch
import torch.nn as nn
from torchvision import models


class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_c,
            out_c,
            kernel_size=2,
            stride=2,
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                out_c + skip_c,
                out_c,
                3,
                padding=1,
            ),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_c,
                out_c,
                3,
                padding=1,
            ),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):

        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:

            x = nn.functional.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat(
            [skip, x],
            dim=1,
        )

        return self.conv(x)


class UNet(nn.Module):

    def __init__(
        self,
        n_channels=3,
        n_classes=1,
        pretrained=True,
    ):
        super().__init__()

        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        encoder = models.resnet18(
            weights=weights
        )

        if n_channels != 3:
            encoder.conv1 = nn.Conv2d(
                n_channels,
                64,
                7,
                stride=2,
                padding=3,
                bias=False,
            )

        self.stem = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
        )

        self.pool = encoder.maxpool

        self.e1 = encoder.layer1
        self.e2 = encoder.layer2
        self.e3 = encoder.layer3
        self.e4 = encoder.layer4

        self.center = nn.Sequential(
            nn.Conv2d(
                512,
                512,
                3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.d1 = DecoderBlock(
            512,
            256,
            256,
        )

        self.d2 = DecoderBlock(
            256,
            128,
            128,
        )

        self.d3 = DecoderBlock(
            128,
            64,
            64,
        )

        self.d4 = DecoderBlock(
            64,
            64,
            64,
        )

        self.final_up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        self.out = nn.Conv2d(
            64,
            n_classes,
            1,
        )

    def forward(self, x):

        x0 = self.stem(x)

        x1 = self.pool(x0)

        x2 = self.e1(x1)

        x3 = self.e2(x2)

        x4 = self.e3(x3)

        x5 = self.e4(x4)

        x = self.center(x5)

        x = self.d1(x, x4)

        x = self.d2(x, x3)

        x = self.d3(x, x2)

        x = self.d4(x, x0)

        x = self.final_up(x)

        return self.out(x)