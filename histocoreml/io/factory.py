"""Factory for selecting the appropriate WSI reader backend.

Supported backends:
- ``openslide``  — SVS, TIFF, NDPI, SCN, MRXS, VMS, BIF (default)
- ``tifffile``   — Generic TIFF / BigTIFF / OME-TIFF (no openslide required)

The backend is chosen automatically based on the file extension, or can be
forced via the ``backend`` argument.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from histocoreml.io.base_reader import BaseWSIReader

logger = logging.getLogger(__name__)

_OPENSLIDE_EXTENSIONS = {
    ".svs", ".tif", ".tiff", ".ndpi", ".scn", ".vms", ".vmu", ".bif", ".mrxs",
}

_TIFFFILE_EXTENSIONS = {
    ".tif", ".tiff", ".btf", ".tf8", ".tf2",
}

BackendType = Literal["openslide", "tifffile", "auto"]


def get_reader(path: Path | str, backend: BackendType = "auto") -> BaseWSIReader:
    """Return an unopened WSI reader for *path*.

    Args:
        path:    Path to the WSI file.
        backend: ``"openslide"`` | ``"tifffile"`` | ``"auto"`` (default).
                 ``"auto"`` selects openslide for known extensions, tifffile
                 otherwise.

    Returns:
        An unopened :class:`BaseWSIReader` instance.

    Raises:
        ValueError: If *backend* is not recognised.
        ImportError: If the selected backend is not installed.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if backend == "auto":
        backend = "openslide" if suffix in _OPENSLIDE_EXTENSIONS else "tifffile"
        if suffix not in _OPENSLIDE_EXTENSIONS and suffix not in _TIFFFILE_EXTENSIONS:
            logger.warning(
                "Extension %s not in known sets; attempting openslide anyway.", suffix
            )
            backend = "openslide"

    if backend == "openslide":
        from histocoreml.io.openslide_reader import OpenSlideReader  # noqa: PLC0415
        logger.debug("Selected OpenSlideReader for %s", path.name)
        return OpenSlideReader(path)

    if backend == "tifffile":
        from histocoreml.io.tifffile_reader import TifffileReader  # noqa: PLC0415
        logger.debug("Selected TifffileReader for %s", path.name)
        return TifffileReader(path)

    raise ValueError(f"Unknown backend '{backend}'. Choose from: openslide, tifffile, auto")
