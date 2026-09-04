"""End-to-end segmentation training from whole-slide images.

Patches are never materialised: coordinates are indexed once, pixels are read
inside the DataLoader workers, and inference optionally runs on a test directory
once training is done. To train from tiles already on disk instead, pair
:func:`~histocoreml.training.build_train_dataloader` with
:class:`~histocoreml.training.SegmentationTrainer`.

Everything is driven by an :class:`~histocoreml.config.ExperimentConfig`, so a
dataset is onboarded by writing a YAML file rather than a script.

:func:`extract_patches_to_disk` covers the other workflow: tile every slide
once, write the patches as PNG pairs, then run many experiments over them
without re-reading the slides each epoch. The output layout is exactly what
:func:`~histocoreml.training.build_train_dataloader` (and
``histo-train --images/--masks``) expects.

Usage::

    from histocoreml.config import ExperimentConfig
    from histocoreml.pipelines import SegmentationTrainingPipeline

    cfg = ExperimentConfig.from_yaml("configs/hubmap_glomeruli.yaml")
    pipeline = SegmentationTrainingPipeline(cfg)
    result = pipeline.run()
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from histocoreml.config import (
    ExperimentConfig,
    SegmentationPipelineConfig,
    TilingConfig,
    TrainingResult,
)
from histocoreml.output.patch_thumbnail_saver import (
    patch_to_rgb_uint8,
    visualise_dataset_samples,
)
from histocoreml.pipelines.base import BaseTrainingPipeline
from histocoreml.training.dataset import (
    MaskProvider,
    RLEMaskProvider,
    SegmentationDataset,
)
from histocoreml.training.losses import get_loss
from histocoreml.training.trainer import SegmentationTrainer
from histocoreml.training.transforms import build_augmentation_pair
from histocoreml.utils.progress import create_progress_bar
from histocoreml.utils.seed import seed_everything

logger = logging.getLogger(__name__)

_SLIDE_GLOBS = ("*.tiff", "*.tif", "*.svs", "*.ndpi")


class SegmentationTrainingPipeline(BaseTrainingPipeline[ExperimentConfig, TrainingResult]):
    """Trains a segmentation model on WSI patches read on the fly.

    Args:
        cfg:            Parsed experiment configuration.
        mask_provider:  Ground-truth source. Defaults to
                        :class:`~histocoreml.training.dataset.RLEMaskProvider`
                        built from ``data.train_csv``.
        debug:          Truncate both splits to ``training.debug_samples`` patches.
        resume:         Checkpoint to resume training from.
        checkpoint:     Checkpoint to run inference with. Defaults to
                        ``<output_dir>/checkpoints/best.pth``.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        mask_provider: MaskProvider | None = None,
        debug: bool = False,
        resume: Path | None = None,
        checkpoint: Path | None = None,
    ) -> None:
        super().__init__(cfg)
        self.config: ExperimentConfig = cfg
        self.debug = debug or cfg.debug
        self.resume = resume
        self.checkpoint = checkpoint

        self.output_dir = cfg.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(cfg.seed)

        self.training_cfg = cfg.training_config()
        self.tiling_cfg = cfg.tiling_config()
        self.mask_provider = mask_provider or RLEMaskProvider.from_csv(Path(cfg.data["train_csv"]))

    # ── Data ──────────────────────────────────────────────────────────────────

    def split_slide_ids(self) -> tuple[list[str], list[str]]:
        """Split the provider's slides into train/val by ``data.val_split``.

        Splitting by slide rather than by patch keeps patches from one slide out
        of both sides, which would otherwise inflate validation scores.
        """
        slide_ids = self.mask_provider.slide_ids()
        random.shuffle(slide_ids)
        val_n = max(1, int(len(slide_ids) * float(self.config.data.get("val_split", 0.2))))
        return slide_ids[:-val_n], slide_ids[-val_n:]

    def build_dataset(
        self,
        slide_ids: list[str],
        transform: Callable[..., dict] | None,
    ) -> SegmentationDataset:
        """Build a dataset over *slide_ids* with *transform* applied."""
        return SegmentationDataset(
            slide_dir=Path(self.config.data["train_dir"]),
            mask_provider=self.mask_provider,
            tiling_cfg=self.tiling_cfg,
            patch_size=self.config.patch_size,
            target_mpp=self.config.target_mpp,
            slide_ids=slide_ids,
            transform=transform,
        )

    def build_loader(self, dataset: SegmentationDataset, *, shuffle: bool) -> DataLoader:
        """Wrap *dataset* in a DataLoader configured from the ``data`` section."""
        return DataLoader(
            dataset,
            batch_size=int(self.config.data.get("batch_size", 8)),
            shuffle=shuffle,
            num_workers=int(self.config.data.get("num_workers", 4)),
            pin_memory=bool(self.config.data.get("pin_memory", True)),
            drop_last=shuffle,
        )

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, train: bool = True, infer: bool = True) -> TrainingResult:
        """Run training and/or inference.

        Args:
            train: Run the training stage.
            infer: Run inference over ``data.test_dir`` when that directory exists.

        Returns:
            A :class:`~histocoreml.config.TrainingResult`. Failures are recorded
            in ``errors`` rather than raised, matching the inference pipelines.
        """
        start = time.perf_counter()
        result = TrainingResult()

        try:
            if train:
                self._train(result)
            if infer and Path(self.config.data.get("test_dir", "")).exists():
                result.metadata["inference"] = self._infer()
        except Exception as exc:
            logger.exception("Training pipeline failed")
            result.errors.append(str(exc))

        result.elapsed_seconds = time.perf_counter() - start
        logger.info(
            "Pipeline complete in %.1fs. Outputs → %s",
            result.elapsed_seconds,
            self.output_dir,
        )
        return result

    def _train(self, result: TrainingResult) -> None:
        logger.info("%s\nTRAINING\n%s", "─" * 70, "─" * 70)

        train_transform, val_transform = build_augmentation_pair(
            self.config.data.get("augmentation")
        )
        train_ids, val_ids = self.split_slide_ids()
        logger.info("Slides — train: %d  val: %d", len(train_ids), len(val_ids))

        train_ds = self.build_dataset(train_ids, train_transform)
        val_ds = self.build_dataset(val_ids, val_transform)

        if self.debug:
            n = self.config.debug_samples
            train_ds.subset(n)
            val_ds.subset(max(1, n // 5))

        logger.info("Patches — train: %d  val: %d", len(train_ds), len(val_ds))
        if len(train_ds) == 0:
            raise ValueError(
                "No training patches found! Check:\n"
                f"  1. Slide directory exists: {self.config.data['train_dir']}\n"
                f"  2. CSV file is valid: {self.config.data['train_csv']}\n"
                f"  3. Tissue threshold is not too strict: {self.tiling_cfg.tissue_threshold}\n"
                "  4. Slide files exist and their ids match the CSV"
            )

        visualise_dataset_samples(train_ds, self.output_dir, stem="epoch_000")

        trainer = SegmentationTrainer(
            self.training_cfg,
            criterion=get_loss(**self.config.loss_spec()),
        )
        if self.resume:
            logger.info("Resuming from %s", self.resume)
            trainer.load_checkpoint(self.resume)

        viz_interval = int(self.config.training.get("viz_interval", 5))

        def on_epoch_end(epoch: int, _metrics: dict[str, float]) -> None:
            if viz_interval > 0 and epoch % viz_interval == 0:
                visualise_dataset_samples(train_ds, self.output_dir, stem=f"epoch_{epoch:03d}")

        history = trainer.fit(
            self.build_loader(train_ds, shuffle=True),
            self.build_loader(val_ds, shuffle=False),
            on_epoch_end=on_epoch_end,
        )

        result.history = history
        result.best_metric = trainer.state.best_metric
        result.epochs_trained = trainer.state.epoch
        result.checkpoint_path = trainer.checkpoint_dir / "best.pth"

    def _infer(self) -> dict[str, int]:
        logger.info("%s\nINFERENCE\n%s", "─" * 70, "─" * 70)

        ckpt = self.checkpoint or self.output_dir / "checkpoints" / "best.pth"
        if not ckpt.exists():
            logger.warning("No checkpoint at %s — skipping inference", ckpt)
            return {}

        # Imported here so training does not require the inference stack.
        from histocoreml.pipelines.inference.segmentation import (  # noqa: PLC0415
            SegmentationInferencePipeline,
        )

        pipeline = SegmentationInferencePipeline(
            SegmentationPipelineConfig(
                model=self.config.inference_model_config(ckpt),
                tiling=self.config.inference_tiling_config(),
                output=self.config.output_config(self.output_dir / "test_predictions"),
            )
        )

        test_dir = Path(self.config.data["test_dir"])
        slides = sorted(p for pattern in _SLIDE_GLOBS for p in test_dir.glob(pattern))
        logger.info("Running inference on %d slides", len(slides))

        results = pipeline.run(slides)
        successful = sum(1 for r in results if r.success)
        logger.info("Inference done: %d/%d successful", successful, len(slides))
        return {"total": len(slides), "successful": successful}


# ── Pre-extraction ────────────────────────────────────────────────────────────


@dataclass
class ExtractionStats:
    """What :func:`extract_patches_to_disk` wrote.

    Attributes:
        slides_processed: Slides that contributed at least one patch.
        slides_skipped:   Slides with no tissue patches, or that failed to read.
        patches_written:  PNG pairs written.
        images_dir:       Directory holding the patch images.
        masks_dir:        Directory holding the matching masks.
    """

    slides_processed: int = 0
    slides_skipped: int = 0
    patches_written: int = 0
    images_dir: Path = Path()
    masks_dir: Path = Path()


def extract_patches_to_disk(
    slide_dir: Path | str,
    mask_provider: MaskProvider,
    output_dir: Path | str,
    tiling_cfg: TilingConfig,
    patch_size: int = 512,
    target_mpp: float = 0.5,
    slide_ids: Sequence[str] | None = None,
    skip_empty_masks: bool = False,
    limit: int | None = None,
) -> ExtractionStats:
    """Tile slides once and write ``images/`` and ``masks/`` PNG pairs.

    :class:`SegmentationTrainingPipeline` re-reads slides every epoch, which
    is the right trade when a dataset is used once. Extracting up front costs
    disk and a slow first pass, but pays off across repeated experiments — and
    lets the patches be inspected, filtered or shared.

    Tiling, tissue filtering and mask alignment all go through
    :class:`~histocoreml.training.dataset.SegmentationDataset`, so
    extracted patches are identical to what on-the-fly training would have fed
    the model.

    Args:
        slide_dir:        Directory holding the slide files.
        mask_provider:    Ground-truth source, e.g.
                          :class:`~histocoreml.training.dataset.RLEMaskProvider`.
        output_dir:       Root for the ``images/`` and ``masks/`` subdirectories.
        tiling_cfg:       Overlap and tissue-filter thresholds.
        patch_size:       Patch side length at *target_mpp*.
        target_mpp:       Resolution to tile at, in microns per pixel.
        slide_ids:        Restrict to these slides. Defaults to all of them.
        skip_empty_masks: Drop patches whose mask has no foreground. Useful for
                          sparse targets, but it biases the class balance — the
                          model then never sees pure-background tissue.
        limit:            Stop after this many patches (for smoke tests).

    Returns:
        :class:`ExtractionStats` describing what was written.

    Example::

        provider = RLEMaskProvider.from_csv(Path("data/train.csv"))
        stats = extract_patches_to_disk(
            slide_dir=Path("data/train"),
            mask_provider=provider,
            output_dir=Path("patches"),
            tiling_cfg=TilingConfig(overlap=0),
        )
        loader = build_train_dataloader(stats.images_dir, stats.masks_dir)
    """
    from PIL import Image  # noqa: PLC0415

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # transform=None: augmentation belongs at training time, not on disk.
    dataset = SegmentationDataset(
        slide_dir=slide_dir,
        mask_provider=mask_provider,
        tiling_cfg=tiling_cfg,
        patch_size=patch_size,
        target_mpp=target_mpp,
        slide_ids=slide_ids,
        transform=None,
    )

    stats = ExtractionStats(images_dir=images_dir, masks_dir=masks_dir)
    per_slide: dict[str, int] = {}

    total = len(dataset) if limit is None else min(len(dataset), limit)
    pbar = create_progress_bar(total, desc="Extracting patches", unit="patch")

    for idx in range(total):
        slide_id, coord = dataset.patch_list[idx]
        pbar.update(1)

        sample = dataset[idx]
        mask = (sample["mask"].numpy().squeeze() > 0.5).astype(np.uint8)
        if skip_empty_masks and not mask.any():
            continue

        image = patch_to_rgb_uint8(sample["image"].numpy())
        name = f"{slide_id}_x{coord.x}_y{coord.y}.png"
        Image.fromarray(image).save(images_dir / name)
        Image.fromarray(mask * 255).save(masks_dir / name)

        stats.patches_written += 1
        per_slide[slide_id] = per_slide.get(slide_id, 0) + 1

    pbar.close()

    stats.slides_processed = len(per_slide)
    stats.slides_skipped = len(dataset.slide_ids) - stats.slides_processed

    manifest = {
        "patch_size": patch_size,
        "target_mpp": target_mpp,
        "overlap": tiling_cfg.overlap,
        "tissue_threshold": tiling_cfg.tissue_threshold,
        "skip_empty_masks": skip_empty_masks,
        "patches_written": stats.patches_written,
        "patches_per_slide": per_slide,
    }
    (output_dir / "extraction_manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info(
        "Extracted %d patches from %d slides (%d skipped) -> %s",
        stats.patches_written,
        stats.slides_processed,
        stats.slides_skipped,
        output_dir,
    )
    return stats
