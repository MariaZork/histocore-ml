"""Factory for selecting an inference backend from a :class:`ModelConfig`."""

from __future__ import annotations

import logging
from pathlib import Path

from histocoreml.backends.base_model import BaseInferenceModel
from histocoreml.config import ModelConfig

logger = logging.getLogger(__name__)

_SUFFIX_BACKENDS = {
    ".pt": "torchscript",
    ".ts": "torchscript",
    ".torchscript": "torchscript",
    ".onnx": "onnx",
    ".pth": "checkpoint",
    ".ckpt": "checkpoint",
}


def get_inference_model(cfg: ModelConfig) -> BaseInferenceModel:
    """Return the inference backend for ``cfg.model_path``.

    With ``cfg.backend='auto'`` the backend follows the file suffix, so a
    TorchScript export and a training checkpoint can be pointed at the same
    pipeline without changing anything else.

    Args:
        cfg: Model configuration naming the weights and (optionally) a backend.

    Returns:
        An unloaded backend — call ``load()`` or use it as a context manager.

    Raises:
        ValueError: If the backend is unknown, or 'auto' cannot infer one.
    """
    backend = cfg.backend.lower()
    if backend == "auto":
        suffix = Path(cfg.model_path).suffix.lower()
        inferred = _SUFFIX_BACKENDS.get(suffix)
        if inferred is None:
            raise ValueError(
                f"Cannot infer an inference backend from '{suffix}'. "
                f"Set ModelConfig(backend=...) explicitly. "
                f"Known suffixes: {sorted(_SUFFIX_BACKENDS)}"
            )
        backend = inferred
        logger.debug("Backend '%s' inferred from suffix '%s'", backend, suffix)

    if backend == "torchscript":
        from histocoreml.backends.torchscript_model import TorchScriptModel  # noqa: PLC0415

        return TorchScriptModel(cfg)
    if backend == "onnx":
        from histocoreml.backends.onnx_model import ONNXModel  # noqa: PLC0415

        return ONNXModel(cfg)
    if backend == "checkpoint":
        from histocoreml.backends.checkpoint_model import CheckpointModel  # noqa: PLC0415

        return CheckpointModel(cfg)

    raise ValueError(
        f"Unknown inference backend '{cfg.backend}'. "
        "Available: 'auto', 'torchscript', 'onnx', 'checkpoint'."
    )
