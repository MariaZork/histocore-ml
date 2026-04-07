"""HistoCoreML segmentation pipeline.

Usage::

    from histocoreml.config import PipelineConfig
    from histocoreml.pipeline import SegmentationPipeline
    from pathlib import Path

    cfg      = PipelineConfig.from_yaml("configs/default.yaml")
    pipeline = SegmentationPipeline(cfg)
    results  = pipeline.run([Path("slide.svs")])
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from histocoreml.config import PipelineConfig
from histocoreml.inference.torchscript_model import TorchScriptModel
from histocoreml.io.base_reader import BaseWSIReader, WSIMetadata
from histocoreml.io.factory import get_reader
from histocoreml.output.base_writer import WriteResult
from histocoreml.output.factory import get_writer
from histocoreml.output.overlay_writer import save_overlay as _save_overlay
from histocoreml.postprocessing.mask_assembler import MaskAssembler
from histocoreml.preprocessing.grid_generator import generate_patch_coords
from histocoreml.preprocessing.patch_dataset import build_dataloader
from histocoreml.utils import setup_logging
from histocoreml.utils.progress import progress_bar

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Summary of a completed pipeline run for a single WSI."""

    wsi_path: Path
    write_result: WriteResult
    patch_count: int
    elapsed_seconds: float
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


def _make_assembler(metadata: WSIMetadata, cfg: PipelineConfig) -> MaskAssembler:
    return MaskAssembler(
        metadata=metadata,
        model_cfg=cfg.model,
        tiling_cfg=cfg.tiling,
        downsample_factor=cfg.output.downsample_factor or 1,
    )


def _finalise_and_write(
    assembler: MaskAssembler,
    cfg: PipelineConfig,
    metadata: WSIMetadata,
    stem: str,
    patch_count: int,
    t0: float,
    wsi_path: Path,
    reader: BaseWSIReader | None = None,
) -> PipelineResult:
    logger.info("Finalising mask…")
    final_mask = assembler.finalise()

    writer = get_writer(cfg.output)
    write_result = writer.write(mask=final_mask, metadata=metadata, stem=stem)
    assembler.cleanup()

    if cfg.output.save_overlay and reader is not None:
        _save_overlay(
            slide_path=wsi_path,
            mask=final_mask,
            stem=stem,
            cfg=cfg.output,
            reader=reader,
        )

    elapsed = time.perf_counter() - t0
    logger.info(
        "Done: %s | %d patches | %.1fs | mask → %s",
        wsi_path.name, patch_count, elapsed, write_result.path,
    )
    return PipelineResult(
        wsi_path=wsi_path,
        write_result=write_result,
        patch_count=patch_count,
        elapsed_seconds=elapsed,
    )


class SegmentationPipeline:
    """End-to-end WSI segmentation pipeline.

    Patch I/O is parallelised via ``torch.utils.data.DataLoader``.
    Inference and mask assembly run in the main process.

    Usage::

        cfg      = PipelineConfig.from_yaml("configs/default.yaml")
        pipeline = SegmentationPipeline(cfg)
        results  = pipeline.run([Path("slide.svs")])
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self._cfg = cfg
        setup_logging(cfg.log_level)

    def run(self, wsi_paths: list[Path]) -> list[PipelineResult]:
        """Process a list of WSI files sequentially."""
        results: list[PipelineResult] = []
        for path in wsi_paths:
            try:
                results.append(self._process_slide(Path(path)))
            except Exception as exc:  # noqa: BLE001
                logger.error("Fatal error processing %s: %s", path, exc, exc_info=True)
                results.append(PipelineResult(
                    wsi_path=Path(path),
                    write_result=None,  # type: ignore[arg-type]
                    patch_count=0,
                    elapsed_seconds=0.0,
                    errors=[str(exc)],
                ))
        return results

    def _process_slide(self, path: Path) -> PipelineResult:
        t0 = time.perf_counter()
        logger.info("=" * 60)
        logger.info("Processing: %s", path.name)

        with get_reader(path) as reader:
            metadata = reader.get_metadata()
            coords = generate_patch_coords(metadata, self._cfg.model, self._cfg.tiling,
                                           slide_id=path.stem)

        if not coords:
            raise ValueError(f"No patch coordinates generated for {path.name}")

        logger.info(
            "Slide: %d × %d px | mpp: %s | %d coords",
            *metadata.dimensions,
            f"{metadata.mpp:.4f}" if metadata.mpp else "unknown",
            len(coords),
        )

        assembler = _make_assembler(metadata, self._cfg)
        patch_count = 0
        total_batches = (len(coords) + self._cfg.model.batch_size - 1) // self._cfg.model.batch_size

        with TorchScriptModel(self._cfg.model) as model, get_reader(path) as reader:
            loader = build_dataloader(
                slide_path=path,
                coords=coords,
                tiling_cfg=self._cfg.tiling,
                model_patch_size=self._cfg.model.patch_size,
                batch_size=self._cfg.model.batch_size,
            )

            with progress_bar(loader, total=total_batches, desc=path.stem) as pbar:
                for batch in pbar:
                    if batch is None:
                        continue
                    masks: np.ndarray = model.predict_batch(batch["images"])
                    assembler.add_batch(masks, batch["coords"])
                    patch_count += len(batch["coords"])

        with get_reader(path) as overlay_reader:
            return _finalise_and_write(
                assembler=assembler,
                cfg=self._cfg,
                metadata=metadata,
                stem=path.stem,
                patch_count=patch_count,
                t0=t0,
                wsi_path=path,
                reader=overlay_reader,
            )


def create_pipeline(cfg: PipelineConfig) -> SegmentationPipeline:
    """Instantiate and return a :class:`SegmentationPipeline`."""
    return SegmentationPipeline(cfg)
