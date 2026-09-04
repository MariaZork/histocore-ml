"""HistoCoreML output — mask writers for TIFF, NPY, RLE, GeoJSON, Zarr."""

from histocoreml.output.base_writer import BaseMaskWriter, WriteResult
from histocoreml.output.factory import get_writer
from histocoreml.output.patch_thumbnail_saver import (
    finalise_montage,
    patch_to_rgb_uint8,
    visualise_dataset_samples,
    write_patch_thumbnail,
)
from histocoreml.output.rle_codec import (
    CocoRLE,
    PlainRLE,
    coco_rle_decode,
    coco_rle_encode,
    plain_rle_decode,
    plain_rle_encode,
    rle_decode,
    rle_encode,
)
from histocoreml.output.rle_writer import RLEMaskReader, RLEMaskWriter
from histocoreml.output.writers import NumpyMaskWriter, TiffMaskWriter

__all__ = [
    # Mask writers
    "BaseMaskWriter",
    "WriteResult",
    "get_writer",
    "TiffMaskWriter",
    "NumpyMaskWriter",
    "RLEMaskWriter",
    "RLEMaskReader",
    # RLE codec
    "PlainRLE",
    "CocoRLE",
    "plain_rle_encode",
    "plain_rle_decode",
    "coco_rle_encode",
    "coco_rle_decode",
    "rle_encode",
    "rle_decode",
    # QC visualisation
    "write_patch_thumbnail",
    "finalise_montage",
    "patch_to_rgb_uint8",
    "visualise_dataset_samples",
]
