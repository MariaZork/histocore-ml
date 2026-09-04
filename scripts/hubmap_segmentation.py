#!/usr/bin/env python3
"""HuBMAP glomeruli segmentation — end-to-end training entry point.

Driven entirely by an external YAML config (configs/hubmap_glomeruli.yaml).
All logic lives in HistoCoreML; this script only parses arguments and invokes
:class:`~histocoreml.pipelines.SegmentationTrainingPipeline`, which:

1. Parses the YAML into typed configs (:class:`~histocoreml.config.ExperimentConfig`)
2. Tiles the slides on the fly with tissue filtering, decoding RLE masks per patch
3. Builds albumentations transforms from the ``data.augmentation`` block
4. Trains with :class:`~histocoreml.training.SegmentationTrainer`
   (AMP, early stopping, cosine annealing, TensorBoard)
5. Runs inference on ``data.test_dir`` from the best checkpoint

Usage::

    python scripts/hubmap_segmentation.py --config configs/hubmap_glomeruli.yaml

    # Debug mode (small patch budget)
    python scripts/hubmap_segmentation.py --config configs/hubmap_glomeruli.yaml --debug

    # Resume training
    python scripts/hubmap_segmentation.py \\
        --config configs/hubmap_glomeruli.yaml \\
        --resume outputs/hubmap/checkpoints/best.pth

    # Inference only
    python scripts/hubmap_segmentation.py \\
        --config configs/hubmap_glomeruli.yaml \\
        --inference-only \\
        --checkpoint outputs/hubmap/checkpoints/best.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from histocoreml.config import ExperimentConfig
from histocoreml.pipelines import SegmentationTrainingPipeline
from histocoreml.utils import setup_logging


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hubmap_segmentation",
        description="HuBMAP glomeruli segmentation — config-driven pipeline",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hubmap_glomeruli.yaml"),
        help="Path to YAML config (default: configs/hubmap_glomeruli.yaml)",
    )
    p.add_argument("--debug", action="store_true", help="Smoke test on a few patches")
    p.add_argument("--resume", type=Path, help="Resume training from this checkpoint")
    p.add_argument("--inference-only", action="store_true", help="Skip training")
    p.add_argument("--checkpoint", type=Path, help="Checkpoint to run inference with")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    try:
        cfg = ExperimentConfig.from_yaml(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level=cfg.log_level, log_file=cfg.log_file(), force=True)

    pipeline = SegmentationTrainingPipeline(
        cfg,
        debug=args.debug,
        resume=args.resume,
        checkpoint=args.checkpoint,
    )
    result = pipeline.run(train=not args.inference_only)

    for error in result.errors:
        print(f"[ERROR] {error}", file=sys.stderr)
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
