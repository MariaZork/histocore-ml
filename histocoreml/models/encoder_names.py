"""Translation between the public encoder names and the ones timm expects.

HistoCoreML names encoders the way ``segmentation-models-pytorch`` does —
``efficientnet-b4`` with a hyphen — because that is what the YAML configs and
:data:`~histocoreml.models.factory.SUPPORTED_ENCODERS` use, and what smp
accepts when :class:`~histocoreml.training.SegmentationTrainer` builds a model.

``timm`` spells the same architectures with underscores (``efficientnet_b4``)
and raises ``RuntimeError: Unknown model`` otherwise. The custom architectures
in this package call timm directly, so they must translate first.
"""

from __future__ import annotations

__all__ = ["to_timm_name"]

# Names that differ by more than punctuation go here; everything else is
# handled by the hyphen-to-underscore rule below.
_EXPLICIT: dict[str, str] = {}


def to_timm_name(encoder_name: str) -> str:
    """Return the ``timm`` model name for a public encoder name.

    Args:
        encoder_name: Public name, e.g. ``"efficientnet-b4"`` or ``"resnet50"``.

    Returns:
        The name to hand to :func:`timm.create_model`. Names timm already
        accepts (every ResNet, for instance) are returned unchanged.

    Example::

        >>> to_timm_name("efficientnet-b4")
        'efficientnet_b4'
        >>> to_timm_name("resnet50")
        'resnet50'
    """
    if encoder_name in _EXPLICIT:
        return _EXPLICIT[encoder_name]
    return encoder_name.replace("-", "_")
