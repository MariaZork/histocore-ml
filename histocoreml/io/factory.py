"""Factory for selecting the appropriate WSI reader backend.

Supported backends:
- ``openslide``  — SVS, NDPI, SCN, MRXS, VMS, BIF and vendor TIFFs (Aperio, etc.)
- ``tifffile``   — Plain / tiled / OME-TIFF, BigTIFF (e.g. HuBMAP, CAMELYON16)

Auto-selection strategy
-----------------------
``"auto"`` (default) tries OpenSlide first for extensions it supports.
If OpenSlide raises ``OpenSlideUnsupportedFormatError`` (common for plain
tiled TIFFs such as HuBMAP kidney images), it transparently falls back to
TifffileReader.  Explicitly passing ``backend="tifffile"`` skips the probe.

The backend can also be forced per-call or set globally via the env var
``HISTOCOREML_WSI_BACKEND=tifffile``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

from histocoreml.io.base_reader import BaseWSIReader

logger = logging.getLogger(__name__)

# Extensions where we should *try* OpenSlide (may still fall back to tifffile)
_OPENSLIDE_EXTENSIONS = {
    ".svs",
    ".tif",
    ".tiff",
    ".ndpi",
    ".scn",
    ".vms",
    ".vmu",
    ".bif",
    ".mrxs",
}

# Extensions that go straight to tifffile (OpenSlide doesn't handle them)
_TIFFFILE_ONLY_EXTENSIONS = {
    ".btf",
    ".tf8",
    ".tf2",
}

BackendType = Literal["openslide", "tifffile", "auto"]

# Honour a global override so users don't have to pass backend= everywhere
_ENV_BACKEND: BackendType = os.environ.get(  # type: ignore[assignment]
    "HISTOCOREML_WSI_BACKEND", "auto"
)


def get_reader(path: Path | str, backend: BackendType = "auto") -> BaseWSIReader:
    """Return an unopened WSI reader for *path*.

    Args:
        path:    Path to the WSI file.
        backend: ``"openslide"`` | ``"tifffile"`` | ``"auto"`` (default).

                 ``"auto"`` probes OpenSlide for known extensions and falls
                 back to TifffileReader on ``OpenSlideUnsupportedFormatError``
                 (e.g. plain tiled TIFFs produced by HuBMAP or CAMELYON16).

    Returns:
        An **unopened** :class:`BaseWSIReader` instance.
        Use as a context manager or call ``.open()`` before reading.

    Raises:
        ValueError:  If *backend* is not one of the recognised strings.
        ImportError: If the required backend library is not installed.
        FileNotFoundError: If *path* does not exist (raised by the reader).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    # Environment variable can override the caller's default "auto"
    if backend == "auto" and _ENV_BACKEND != "auto":
        backend = _ENV_BACKEND

    if backend == "auto":
        if suffix in _TIFFFILE_ONLY_EXTENSIONS:
            # These are never handled by OpenSlide — skip the probe
            logger.debug("Extension %s → TifffileReader (direct)", path.name)
            return _make_tifffile(path)

        if suffix in _OPENSLIDE_EXTENSIONS:
            return _try_openslide_then_tifffile(path)

        # Unknown extension — try openslide, then tifffile
        logger.warning(
            "Extension '%s' is not in any known set; probing OpenSlide then TiffFile.", suffix
        )
        return _try_openslide_then_tifffile(path)

    if backend == "openslide":
        return _make_openslide(path)

    if backend == "tifffile":
        return _make_tifffile(path)

    raise ValueError(f"Unknown backend '{backend}'. Choose from: openslide, tifffile, auto")


# ── Private helpers ───────────────────────────────────────────────────────────


def _make_openslide(path: Path) -> BaseWSIReader:
    from histocoreml.io.openslide_reader import OpenSlideReader  # noqa: PLC0415

    logger.debug("Backend → OpenSlideReader  (%s)", path.name)
    return OpenSlideReader(path)


def _make_tifffile(path: Path) -> BaseWSIReader:
    from histocoreml.io.tifffile_reader import TifffileReader  # noqa: PLC0415

    logger.debug("Backend → TifffileReader  (%s)", path.name)
    return TifffileReader(path)


def _try_openslide_then_tifffile(path: Path) -> BaseWSIReader:
    """Probe OpenSlide; fall back to TifffileReader on unsupported format.

    This is the key fix for HuBMAP / CAMELYON16 plain tiled TIFFs which
    OpenSlide rejects with ``OpenSlideUnsupportedFormatError``.
    """
    try:
        import openslide  # noqa: F401, PLC0415  (probing availability)

        # Probe: try to open the file — if it fails we catch and fall through
        reader = _make_openslide(path)
        reader.open()  # will raise OpenSlideUnsupportedFormatError for plain TIFFs
        reader.close()  # success — return a fresh unopened instance
        logger.debug("OpenSlide probe OK for %s", path.name)
        return _make_openslide(path)

    except ImportError:
        logger.debug("openslide-python not installed — using TifffileReader for %s", path.name)
        return _make_tifffile(path)

    except Exception as exc:  # noqa: BLE001
        # Deliberately broad: any OpenSlide failure should fall back to tifffile
        # rather than abort. OpenSlideUnsupportedFormatError is not importable
        # when openslide-python is absent, so it cannot be named here.
        _is_unsupported = "Unsupported" in type(exc).__name__ or "unsupported" in str(exc).lower()
        if _is_unsupported:
            logger.info(
                "OpenSlide cannot read %s (%s) — falling back to TifffileReader",
                path.name,
                type(exc).__name__,
            )
        else:
            logger.warning(
                "OpenSlide raised unexpected error for %s: %s — falling back to TifffileReader",
                path.name,
                exc,
            )
        return _make_tifffile(path)
