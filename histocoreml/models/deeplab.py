"""DeepLabV3+ implementation for histology segmentation."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from histocoreml.models.encoder_names import to_timm_name


class ASPPConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        x = self.pool(x)
        x = self.conv(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int = 256, dilations: list[int] | None = None
    ) -> None:
        super().__init__()
        # Resolved here rather than in the signature: a list default
        # would be shared by every instance of this class.
        dilations = [6, 12, 18] if dilations is None else dilations

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = ASPPConv(in_channels, out_channels, dilations[0])
        self.conv3 = ASPPConv(in_channels, out_channels, dilations[1])
        self.conv4 = ASPPConv(in_channels, out_channels, dilations[2])
        self.pool = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.pool(x)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.project(x)


class Decoder(nn.Module):
    def __init__(
        self,
        low_level_channels: int,
        num_classes: int,
        aspp_channels: int = 256,
        low_level_reduction: int = 48,
    ) -> None:
        super().__init__()

        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, low_level_reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_reduction),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(
                aspp_channels + low_level_reduction, 256, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, low_level_feat: torch.Tensor) -> torch.Tensor:
        low_level_feat = self.low_level_conv(low_level_feat)
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, low_level_feat], dim=1)
        return self.classifier(x)


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_pretrained: bool = True,
        in_channels: int = 3,
        num_classes: int = 1,
        output_stride: int = 16,
        aspp_dilations: list[int] | None = None,
    ) -> None:
        super().__init__()
        # Resolved here rather than in the signature: a list default
        # would be shared by every instance of this class.
        aspp_dilations = [6, 12, 18] if aspp_dilations is None else aspp_dilations

        self.num_classes = num_classes
        self.output_stride = output_stride

        self.encoder = timm.create_model(
            to_timm_name(encoder_name),
            pretrained=encoder_pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=[1, 4],
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 224, 224)
            feats = self.encoder(dummy)
            low_level_channels = feats[0].shape[1]
            high_level_channels = feats[1].shape[1]

        self.aspp = ASPP(
            in_channels=high_level_channels, out_channels=256, dilations=aspp_dilations
        )

        self.decoder = Decoder(
            low_level_channels=low_level_channels,
            num_classes=num_classes,
            aspp_channels=256,
            low_level_reduction=48,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        low_level_feat, high_level_feat = self.encoder(x)
        x = self.aspp(high_level_feat)
        x = self.decoder(x, low_level_feat)

        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x
