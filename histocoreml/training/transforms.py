"""Declarative albumentations pipeline builder.

Lets augmentation be described in YAML instead of code, so an experiment
config fully determines the training pipeline::

    data:
      augmentation:
        enabled: true
        train:
          - name: HorizontalFlip
            params: {p: 0.5}
          - name: ShiftScaleRotate
            params: {shift_limit: 0.1, scale_limit: 0.1, rotate_limit: 15, p: 0.5}
        valid:
          - name: Normalize
            params: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}

Each entry names any callable exposed by ``albumentations`` and passes
``params`` straight to its constructor.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

logger = logging.getLogger(__name__)


def build_transforms(spec: Sequence[dict[str, Any]] | None) -> Callable | None:
    """Build an ``albumentations.Compose`` from a list of ``{name, params}`` dicts.

    Args:
        spec: Augmentation entries. Each needs a ``name`` matching an
              albumentations class; ``params`` (optional) are constructor kwargs.

    Returns:
        A composed transform, or ``None`` when *spec* is empty or albumentations
        is not installed — callers treat ``None`` as "no augmentation".
    """
    if not spec:
        return None

    try:
        import albumentations as A  # noqa: PLC0415, N812
    except ImportError:
        logger.warning("albumentations not installed — no augmentations applied")
        return None

    ops = []
    for entry in spec:
        name = entry["name"]
        params = entry.get("params", {}) or {}
        cls = getattr(A, name, None)
        if cls is None:
            logger.warning("Unknown albumentations transform '%s' — skipping", name)
            continue
        ops.append(cls(**params))

    return A.Compose(ops) if ops else None


def build_augmentation_pair(
    augmentation_cfg: dict[str, Any] | None,
) -> tuple[Callable | None, Callable | None]:
    """Build the ``(train, valid)`` transforms from a YAML ``augmentation`` block.

    Honours the block's ``enabled`` flag: when it is false both transforms are
    ``None``, which is the usual way to switch augmentation off for a debug run.

    Args:
        augmentation_cfg: The ``data.augmentation`` mapping, or ``None``.

    Returns:
        ``(train_transform, valid_transform)``, either of which may be ``None``.
    """
    cfg = augmentation_cfg or {}
    if not cfg.get("enabled", True):
        return None, None
    return build_transforms(cfg.get("train")), build_transforms(cfg.get("valid"))
