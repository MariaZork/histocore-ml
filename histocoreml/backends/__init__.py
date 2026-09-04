"""Model backends — how a trained model is loaded and called.

Each backend wraps one weights format behind the same interface, so a pipeline
can run a TorchScript export, an ONNX graph or a raw training checkpoint without
knowing which it has:

============  ===========================================================
Suffix        Backend
============  ===========================================================
``.pt``       :class:`TorchScriptModel` — a serialised graph
``.onnx``     :class:`ONNXModel` — ONNX Runtime, no PyTorch needed
``.pth``      :class:`CheckpointModel` — a state dict rebuilt into its
              architecture via segmentation-models-pytorch
============  ===========================================================

:func:`get_inference_model` picks one from ``ModelConfig.backend``, which
defaults to ``"auto"`` and reads the suffix.

This package is deliberately separate from :mod:`histocoreml.pipelines.inference`:
here is *how to run a model*, there is *the run itself* — reading a slide,
tiling it, assembling the mask.

Usage::

    from histocoreml.backends import get_inference_model
    from histocoreml.config import ModelConfig

    cfg = ModelConfig(model_path="checkpoints/best.pth", architecture="unet++")
    with get_inference_model(cfg) as model:
        masks = model.predict_batch(batch)
"""

from histocoreml.backends.base_model import BaseInferenceModel, BaseSegmentationModel
from histocoreml.backends.checkpoint_model import CheckpointModel
from histocoreml.backends.factory import get_inference_model
from histocoreml.backends.onnx_model import ONNXModel
from histocoreml.backends.torchscript_model import TorchScriptModel

__all__ = [
    "BaseInferenceModel",
    "BaseSegmentationModel",
    "TorchScriptModel",
    "ONNXModel",
    "CheckpointModel",
    "get_inference_model",
]
