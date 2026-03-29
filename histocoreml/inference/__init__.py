"""HistoCoreML inference — model backends for segmentation."""

from histocoreml.inference.base_model import BaseSegmentationModel
from histocoreml.inference.torchscript_model import TorchScriptModel

__all__ = ["BaseSegmentationModel", "TorchScriptModel"]
