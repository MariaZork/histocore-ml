"""HistoCoreML I/O — WSI readers and file format abstractions."""

from histocoreml.io.base_reader import BaseWSIReader, WSIMetadata
from histocoreml.io.factory import get_reader
from histocoreml.io.openslide_reader import OpenSlideReader
from histocoreml.io.tifffile_reader import TifffileReader

__all__ = [
    "BaseWSIReader",
    "WSIMetadata",
    "get_reader",
    "OpenSlideReader",
    "TifffileReader",
]
