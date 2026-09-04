"""Tests for encoder naming and model construction.

The public encoder names follow segmentation-models-pytorch (``efficientnet-b4``),
which is what the YAML configs use. timm spells them with underscores, so the
custom architectures must translate before calling it.
"""

from __future__ import annotations

import pytest
import torch

from histocoreml.models import ArchitectureConfig, get_model
from histocoreml.models.encoder_names import to_timm_name
from histocoreml.models.factory import SUPPORTED_ENCODERS

_COMBINATIONS = [
    (arch, enc)
    for arch, encoders in SUPPORTED_ENCODERS.items()
    for enc in encoders
    if enc != "custom"
]


class TestEncoderNameTranslation:
    @pytest.mark.parametrize(
        ("public", "timm_name"),
        [
            ("efficientnet-b0", "efficientnet_b0"),
            ("efficientnet-b4", "efficientnet_b4"),
            ("efficientnet-b7", "efficientnet_b7"),
            ("resnet18", "resnet18"),
            ("resnet50", "resnet50"),
        ],
    )
    def test_translation(self, public: str, timm_name: str):
        assert to_timm_name(public) == timm_name

    def test_every_supported_encoder_is_known_to_timm(self):
        timm = pytest.importorskip("timm")
        known = set(timm.list_models())

        for _, encoder in _COMBINATIONS:
            assert to_timm_name(encoder) in known, encoder


class TestModelConstruction:
    @pytest.mark.slow
    @pytest.mark.parametrize(("architecture", "encoder"), _COMBINATIONS)
    def test_builds_and_preserves_spatial_size(self, architecture: str, encoder: str):
        pytest.importorskip("timm")

        model = get_model(
            ArchitectureConfig(
                architecture=architecture,
                encoder=encoder,
                encoder_pretrained=False,
                num_classes=1,
            )
        ).eval()

        with torch.no_grad():
            output = model(torch.rand(1, 3, 128, 128))

        assert output.shape == (1, 1, 128, 128)

    def test_multi_class_output(self):
        pytest.importorskip("timm")
        model = get_model(
            ArchitectureConfig(
                architecture="unet++",
                encoder="efficientnet-b0",
                encoder_pretrained=False,
                num_classes=3,
            )
        ).eval()

        with torch.no_grad():
            output = model(torch.rand(2, 3, 128, 128))

        assert output.shape == (2, 3, 128, 128)

    def test_deep_supervision_heads_match_their_inputs(self):
        pytest.importorskip("timm")
        model = get_model(
            ArchitectureConfig(
                architecture="unet++",
                encoder="resnet18",
                encoder_pretrained=False,
                deep_supervision=True,
            )
        ).eval()

        with torch.no_grad():
            output, deep_outputs = model(torch.rand(1, 3, 128, 128))

        assert output.shape == (1, 1, 128, 128)
        assert all(d.shape == (1, 1, 128, 128) for d in deep_outputs)

    def test_unsupported_options_are_dropped_not_fatal(self):
        """DeepLabV3+ takes no dropout; passing one must not raise TypeError."""
        pytest.importorskip("timm")
        model = get_model(
            ArchitectureConfig(
                architecture="deeplabv3+",
                encoder="resnet50",
                encoder_pretrained=False,
                dropout=0.3,
            )
        ).eval()

        with torch.no_grad():
            assert model(torch.rand(1, 3, 128, 128)).shape == (1, 1, 128, 128)

    def test_unknown_architecture_raises(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            get_model(ArchitectureConfig(architecture="nope", encoder="resnet50"))

    def test_unsupported_encoder_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            get_model(ArchitectureConfig(architecture="deeplabv3+", encoder="resnet18"))
