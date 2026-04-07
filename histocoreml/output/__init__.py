"""HistoCoreML output — mask writers for TIFF, NPY, RLE, GeoJSON, Zarr."""

from histocoreml.output.base_writer import BaseMaskWriter, WriteResult
from histocoreml.output.factory import get_writer
from histocoreml.output.rle_codec import (
    CocoRLE,
    PlainRLE,
    coco_rle_decode,
    coco_rle_encode,
    plain_rle_decode,
    plain_rle_encode,
)
from histocoreml.output.rle_writer import RLEMaskReader, RLEMaskWriter
from histocoreml.output.writers import NumpyMaskWriter, TiffMaskWriter

__all__ = [
    "BaseMaskWriter", "WriteResult",
    "get_writer",
    "PlainRLE", "CocoRLE",
    "plain_rle_encode", "plain_rle_decode",
    "coco_rle_encode", "coco_rle_decode",
    "TiffMaskWriter", "NumpyMaskWriter",
    "RLEMaskWriter", "RLEMaskReader",
]
