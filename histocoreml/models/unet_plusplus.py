"""UNet++ (Nested U-Net) with EfficientNet and other backbones.

UNet++ introduces nested skip pathways and dense connections between encoder and decoder,
which helps capture fine-grained details better than standard U-Net.

Reference:
    Zhou et al. "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"
    https://arxiv.org/abs/1807.10165
"""

from __future__ import annotations

from typing import cast

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from histocoreml.models.encoder_names import to_timm_name


class ConvBlock(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class NestedBlock(nn.Module):
    """Nested dense block for UNet++.

    Args:
        in_channels:   Channels of the block's own input.
        out_channels:  Channels this block emits.
        skip_channels: Channels of each densely-connected tensor concatenated
            onto the input. These come from *different* decoder levels, so they
            are not all the same width — passing a count and multiplying by
            ``out_channels`` (as this once did) mis-sizes the convolution.
        dropout:       Dropout rate inside the convolution block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: list[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.skip_channels = list(skip_channels)
        total_channels = in_channels + sum(self.skip_channels)
        self.conv = ConvBlock(total_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, nested_upsampled: list[torch.Tensor]) -> torch.Tensor:
        inputs = [x] + nested_upsampled
        x = torch.cat(inputs, dim=1)
        return self.conv(x)


def _feature_channels(encoder: nn.Module, in_channels: int) -> list[int]:
    """Return the channel count of each feature map the encoder emits.

    Reading them off a dummy forward pass keeps this correct for every timm
    encoder. The previous hardcoded tables listed five channels while
    ``out_indices=[1, 2, 3, 4]`` yields four, and silently fell back to another
    model's channels for anything unlisted — which surfaced as shape mismatches
    deep in the decoder rather than a clear error.
    """
    was_training = encoder.training
    encoder.eval()
    try:
        with torch.no_grad():
            features = encoder(torch.zeros(1, in_channels, 64, 64))
    finally:
        encoder.train(was_training)
    return [f.shape[1] for f in features]


class EfficientNetEncoder(nn.Module):
    """EfficientNet encoder with multiple feature extraction points."""

    def __init__(
        self,
        model_name: str = "efficientnet-b4",
        pretrained: bool = True,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.encoder = timm.create_model(
            to_timm_name(model_name),
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=[1, 2, 3, 4],
        )
        self.channels = _feature_channels(self.encoder, in_channels)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.encoder(x)


class ResNetEncoder(nn.Module):
    """ResNet encoder for UNet++."""

    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.encoder = timm.create_model(
            to_timm_name(model_name),
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=[1, 2, 3, 4],
        )
        self.channels = _feature_channels(self.encoder, in_channels)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.encoder(x)


class UNetPlusPlusDecoder(nn.Module):
    """UNet++ nested decoder with dense skip connections."""

    def __init__(
        self,
        encoder_channels: list[int],
        decoder_channels: list[int] | None = None,
        dropout: float = 0.0,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        # Resolved here rather than in the signature: a list default
        # would be shared by every instance of this class.
        decoder_channels = [256, 128, 64, 32] if decoder_channels is None else decoder_channels
        self.deep_supervision = deep_supervision
        self.num_levels = len(encoder_channels)

        # decoder_channels is written widest-first ([256, 128, 64, 32]) while
        # level 0 is the *finest* encoder feature map and should be narrowest.
        # Indexing it directly gave level 0 the widest width, leaving the
        # segmentation head — which expects decoder_channels[-1] — mismatched.
        self.widths = [decoder_channels[self.num_levels - 1 - i] for i in range(self.num_levels)]

        self.nested_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        for i in range(self.num_levels):
            level_blocks = nn.ModuleList()
            level_ups = nn.ModuleList()

            for j in range(self.num_levels - i):
                if j == 0:
                    in_ch = encoder_channels[i]
                else:
                    in_ch = self.widths[i]

                # forward() concatenates one upsampled tensor per deeper level
                # k that already holds a j-th output, i.e. i < k < num_levels-j.
                # up_convs[k-1][j] emits decoder_channels[k-1] channels, which
                # varies with k rather than being decoder_channels[i] each time.
                skip_channels = [self.widths[k - 1] for k in range(i + 1, self.num_levels - j)]

                block = NestedBlock(
                    in_channels=in_ch,
                    out_channels=self.widths[i],
                    skip_channels=skip_channels,
                    dropout=dropout,
                )
                level_blocks.append(block)

                if i > 0:
                    up_conv = nn.ConvTranspose2d(
                        self.widths[i],
                        self.widths[i - 1],
                        kernel_size=2,
                        stride=2,
                    )
                    level_ups.append(up_conv)

            self.nested_blocks.append(level_blocks)
            if level_ups:
                self.up_convs.append(level_ups)

    def forward(
        self, encoder_features: list[torch.Tensor]
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        nested_outputs: list[list[torch.Tensor]] = [[] for _ in range(self.num_levels)]

        for i in range(self.num_levels - 1, -1, -1):
            for j in range(self.num_levels - i):
                if j == 0:
                    x = encoder_features[i]
                else:
                    x = nested_outputs[i][j - 1]

                nested_upsampled = []
                for k in range(i + 1, self.num_levels):
                    if j < len(nested_outputs[k]):
                        up_conv = cast(nn.ModuleList, self.up_convs[k - 1])[j]
                        up = up_conv(nested_outputs[k][j])
                        if up.shape[2:] != x.shape[2:]:
                            up = F.interpolate(
                                up, size=x.shape[2:], mode="bilinear", align_corners=False
                            )
                        nested_upsampled.append(up)

                block = cast(nn.ModuleList, self.nested_blocks[i])[j]
                out = block(x, nested_upsampled)
                nested_outputs[i].append(out)

        final_output = nested_outputs[0][-1]

        if self.deep_supervision:
            deep_outputs = [nested_outputs[0][j] for j in range(self.num_levels)]
            return final_output, deep_outputs

        return final_output


class UNetPlusPlus(nn.Module):
    """UNet++: Nested U-Net with Dense Skip Connections.

    Args:
        encoder_name: Backbone architecture (e.g., "efficientnet-b4", "resnet50")
        encoder_pretrained: Whether to use pretrained encoder weights
        in_channels: Number of input channels
        num_classes: Number of output classes
        decoder_channels: Decoder channel dimensions
        dropout: Dropout rate
        deep_supervision: Enable deep supervision for training
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet-b4",
        encoder_pretrained: bool = True,
        in_channels: int = 3,
        num_classes: int = 1,
        decoder_channels: list[int] | None = None,
        dropout: float = 0.0,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        # Resolved here rather than in the signature: a list default
        # would be shared by every instance of this class.
        decoder_channels = [256, 128, 64, 32] if decoder_channels is None else decoder_channels

        self.encoder_name = encoder_name
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        self.encoder: nn.Module
        if "efficientnet" in encoder_name:
            self.encoder = EfficientNetEncoder(
                model_name=encoder_name,
                pretrained=encoder_pretrained,
                in_channels=in_channels,
            )
        elif "resnet" in encoder_name:
            self.encoder = ResNetEncoder(
                model_name=encoder_name,
                pretrained=encoder_pretrained,
                in_channels=in_channels,
            )
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")

        self.decoder = UNetPlusPlusDecoder(
            encoder_channels=self.encoder.channels,
            decoder_channels=decoder_channels,
            dropout=dropout,
            deep_supervision=deep_supervision,
        )

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1] // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout / 2) if dropout > 0 else nn.Identity(),
            nn.Conv2d(decoder_channels[-1] // 2, num_classes, kernel_size=1),
        )

        if deep_supervision:
            # Every deep-supervision output comes from level 0, so they all
            # carry that level's width rather than one width per entry.
            self.deep_heads = nn.ModuleList(
                [
                    nn.Conv2d(self.decoder.widths[0], num_classes, kernel_size=1)
                    for _ in range(self.decoder.num_levels)
                ]
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
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        input_size = x.shape[2:]

        encoder_features = self.encoder(x)

        if self.deep_supervision:
            decoder_output, deep_outputs = self.decoder(encoder_features)
        else:
            decoder_output = self.decoder(encoder_features)

        output = self.segmentation_head(decoder_output)

        if output.shape[2:] != input_size:
            output = F.interpolate(output, size=input_size, mode="bilinear", align_corners=False)

        if self.deep_supervision:
            deep_preds = []
            for i, deep_out in enumerate(deep_outputs):
                deep_pred = self.deep_heads[i](deep_out)
                if deep_pred.shape[2:] != input_size:
                    deep_pred = F.interpolate(
                        deep_pred, size=input_size, mode="bilinear", align_corners=False
                    )
                deep_preds.append(deep_pred)
            return output, deep_preds

        return output

    def get_encoder_params(self) -> list[nn.Parameter]:
        return list(self.encoder.parameters())

    def get_decoder_params(self) -> list[nn.Parameter]:
        return list(self.decoder.parameters()) + list(self.segmentation_head.parameters())
