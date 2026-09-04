"""Reproducibility helpers."""

from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42, deterministic: bool = False) -> int:
    """Seed ``random``, ``numpy`` and ``torch`` (CPU + all CUDA devices).

    Args:
        seed:          Seed value applied to every RNG.
        deterministic: Also force cuDNN into deterministic mode. Slower, but
                       makes convolution kernels reproducible run to run.

    Returns:
        The seed that was applied, so callers can log or persist it.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.debug("Seeded RNGs with %d (deterministic=%s)", seed, deterministic)
    return seed
