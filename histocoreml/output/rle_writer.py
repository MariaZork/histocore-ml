"""RLE mask writer and reader for space-efficient WSI mask storage."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from histocoreml.config import OutputConfig
from histocoreml.io.base_reader import WSIMetadata
from histocoreml.output.base_writer import BaseMaskWriter, WriteResult
from histocoreml.output.rle_codec import (
    CocoRLE, PlainRLE,
    coco_rle_decode, coco_rle_encode, coco_rle_from_dict,
    plain_rle_decode, plain_rle_encode, plain_rle_from_dict, plain_rle_to_dict,
)

logger = logging.getLogger(__name__)

_PLAIN = "plain"
_COCO  = "coco"
_VALID_SUBFORMATS = {_PLAIN, _COCO}


class RLEMaskWriter(BaseMaskWriter):
    """Write a binary mask as a Run-Length Encoded JSON file."""

    def write(self, mask: np.ndarray, metadata: WSIMetadata, stem: str) -> WriteResult:
        subformat = getattr(self._cfg, "rle_subformat", _PLAIN).lower()
        if subformat not in _VALID_SUBFORMATS:
            raise ValueError(f"Unknown rle_subformat '{subformat}'. Choose from: {sorted(_VALID_SUBFORMATS)}")

        out_path = self._cfg.output_dir / f"{stem}_mask.json"

        if subformat == _COCO:
            payload, run_count, ratio = self._build_coco_payload(mask, metadata, stem)
        else:
            payload, run_count, ratio = self._build_plain_payload(mask, metadata, stem)

        logger.info("Writing RLE mask (%s) → %s  [runs=%d, ratio=%.1f×]",
                    subformat, out_path, run_count, ratio)

        with out_path.open("w") as fh:
            json.dump(payload, fh, separators=(",", ":"))

        return WriteResult(
            path=out_path, shape=tuple(mask.shape[:2]),
            format=f"rle_{subformat}",
            metadata={"run_count": run_count, "compression_ratio": ratio},
        )

    @staticmethod
    def _build_plain_payload(mask, metadata, stem):
        rle: PlainRLE = plain_rle_encode(mask)
        payload = {
            "format": "plain_rle",
            "shape": list(rle.shape),
            "mpp": metadata.mpp,
            "slide_path": stem,
            "run_count": len(rle.runs),
            "compression_ratio": round(rle.compression_ratio(), 2),
            **plain_rle_to_dict(rle),
        }
        return payload, len(rle.runs), rle.compression_ratio()

    @staticmethod
    def _build_coco_payload(mask, metadata, stem):
        rle: CocoRLE = coco_rle_encode(mask)
        h, w = rle.size
        ratio = round((h * w) / max(len(rle.counts) * 4, 1), 2)
        payload = {
            "format": "coco_rle",
            "segmentation": rle.to_coco_dict(),
            "mpp": metadata.mpp,
            "slide_path": stem,
            "run_count": len(rle.counts),
            "compression_ratio": ratio,
        }
        return payload, len(rle.counts), ratio


class RLEMaskReader:
    """Reconstruct a binary numpy mask from a JSON file written by :class:`RLEMaskWriter`."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"RLE file not found: {self._path}")
        with self._path.open() as fh:
            self._data: dict = json.load(fh)

    @property
    def format(self) -> str:
        return self._data.get("format", "plain_rle")

    @property
    def shape(self) -> tuple:
        if self.format == "coco_rle":
            return tuple(self._data["segmentation"]["size"])
        return tuple(self._data["shape"])

    @property
    def mpp(self):
        return self._data.get("mpp")

    @property
    def slide_path(self) -> str:
        return self._data.get("slide_path", "")

    @property
    def compression_ratio(self) -> float:
        return float(self._data.get("compression_ratio", 0.0))

    @property
    def metadata(self) -> dict:
        skip = {"runs", "segmentation"}
        return {k: v for k, v in self._data.items() if k not in skip}

    def decode(self) -> np.ndarray:
        if self.format == "coco_rle":
            return coco_rle_decode(coco_rle_from_dict(self._data))
        elif self.format == "plain_rle":
            return plain_rle_decode(plain_rle_from_dict(self._data))
        else:
            raise ValueError(f"Unrecognised RLE format '{self.format}' in {self._path}.")
