"""Standard U-Net implementation with various backbones."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from histocoreml.models.encoder_names import to_timm_name


class ConvBlock(nn.Module):
    """Double convolution block."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EncoderBlock(nn.Module):
    """Encoder block with downsampling."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        pooled = self.pool(x)
        return x, pooled


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and skip connection."""

    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Standard U-Net architecture."""

    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_pretrained: bool = True,
        in_channels: int = 3,
        num_classes: int = 1,
        features: list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Resolved here rather than in the signature: a list default
        # would be shared by every instance of this class.
        features = [64, 128, 256, 512] if features is None else features

        self.encoder_name = encoder_name
        self.num_classes = num_classes

        if encoder_name == "custom":
            self.encoder = None
            self._build_custom_encoder(in_channels, features, dropout)
            encoder_channels = features
        else:
            self.encoder = timm.create_model(
                to_timm_name(encoder_name),
                pretrained=encoder_pretrained,
                in_chans=in_channels,
                features_only=True,
                out_indices=[1, 2, 3, 4],
            )
            with torch.no_grad():
                dummy = torch.zeros(1, in_channels, 224, 224)
                feats = self.encoder(dummy)
                encoder_channels = [f.shape[1] for f in feats]

        self.bottleneck = ConvBlock(encoder_channels[-1], encoder_channels[-1] * 2, dropout)

        self.decoders = nn.ModuleList()
        decoder_channels = list(reversed(encoder_channels))

        for i in range(len(decoder_channels) - 1):
            self.decoders.append(
                DecoderBlock(
                    in_channels=decoder_channels[i] * 2 if i == 0 else decoder_channels[i],
                    skip_channels=decoder_channels[i + 1],
                    out_channels=decoder_channels[i + 1],
                    dropout=dropout,
                )
            )

        self.segmentation_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)
        self._initialize_weights()

    def _build_custom_encoder(self, in_channels: int, features: list[int], dropout: float) -> None:
        self.encoders = nn.ModuleList()
        for i, feat in enumerate(features):
            in_ch = in_channels if i == 0 else features[i - 1]
            self.encoders.append(EncoderBlock(in_ch, feat, dropout))

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

        if self.encoder is not None:
            skip_connections = self.encoder(x)
            x = skip_connections[-1]
        else:
            skip_connections = []
            for encoder in self.encoders:
                skip, x = encoder(x)
                skip_connections.append(skip)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]
        for i, decoder in enumerate(self.decoders):
            skip = (
                skip_connections[i + 1] if i + 1 < len(skip_connections) else skip_connections[-1]
            )
            x = decoder(x, skip)

        output = self.segmentation_head(x)

        if output.shape[2:] != input_size:
            output = F.interpolate(output, size=input_size, mode="bilinear", align_corners=False)

        return output
