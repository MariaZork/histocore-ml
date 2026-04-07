"""Factory function for selecting the appropriate mask writer."""

from __future__ import annotations

from histocoreml.config import OutputConfig
from histocoreml.output.base_writer import BaseMaskWriter
from histocoreml.output.rle_writer import RLEMaskWriter
from histocoreml.output.writers import (
    GeoJSONMaskWriter,
    NumpyMaskWriter,
    TiffMaskWriter,
    ZarrMaskWriter,
)

_WRITER_REGISTRY: dict[str, type[BaseMaskWriter]] = {
    "tiff":    TiffMaskWriter,
    "tif":     TiffMaskWriter,
    "npy":     NumpyMaskWriter,
    "numpy":   NumpyMaskWriter,
    "rle":     RLEMaskWriter,
    "zarr":    ZarrMaskWriter,
    "geojson": GeoJSONMaskWriter,
}


def get_writer(cfg: OutputConfig) -> BaseMaskWriter:
    """Return the appropriate :class:`BaseMaskWriter` for ``cfg.output_format``.

    Args:
        cfg: Output configuration specifying the desired format.

    Returns:
        An instantiated, ready-to-use writer.

    Raises:
        ValueError: If the format is not registered.
    """
    fmt = cfg.output_format.lower()
    cls = _WRITER_REGISTRY.get(fmt)
    if cls is None:
        raise ValueError(
            f"Unsupported output format '{fmt}'. "
            f"Available formats: {sorted(_WRITER_REGISTRY)}"
        )
    return cls(cfg)
