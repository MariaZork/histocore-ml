"""HistoCoreML command-line interface.

Entry points
------------
``histo-segment``  — Run the WSI segmentation pipeline
``histo-embed``    — Extract foundation model embeddings
``histo-extract``  — Extract biomarkers from a slide + mask
``histo-train``    — Train a segmentation model

Quick usage::

    histo-segment  -c configs/default.yaml   -i data/*.svs --save-overlay
    histo-embed    -c configs/uni.yaml        -i data/*.svs -o embeddings/
    histo-extract  -c configs/biomarker.yaml  -i data/slide.svs --mask outputs/slide_mask.npy
    histo-train    -c configs/training.yaml   --images data/images --masks data/masks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── histo-segment ─────────────────────────────────────────────────────────────

def _segment_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="histo-segment",
                                description="WSI segmentation pipeline.")
    p.add_argument("-c", "--config",  type=Path, required=True, metavar="YAML")
    p.add_argument("-i", "--input",   nargs="+", type=Path, required=True, metavar="FILE")
    p.add_argument("--output-dir",    type=Path, default=None)
    p.add_argument(
        "--output-format",
        choices=["tiff", "npy", "rle", "zarr", "geojson"],
        default=None,
    )
    p.add_argument("--device",        type=str, default=None)
    p.add_argument("--batch-size",    type=int, default=None)
    p.add_argument("--save-overlay",  action="store_true", default=None)
    p.add_argument("--overlay-alpha", type=float, default=None)
    p.add_argument("--overlay-max-edge", type=int, default=None)
    p.add_argument(
        "--normalise",
        action="store_true",
        help="Apply Macenko stain normalisation.",
    )
    return p


def main_segment(argv: list[str] | None = None) -> int:
    args = _segment_parser().parse_args(argv)

    from dataclasses import replace  # noqa: PLC0415

    from histocoreml.config import PipelineConfig  # noqa: PLC0415
    from histocoreml.pipeline import SegmentationPipeline  # noqa: PLC0415

    cfg = PipelineConfig.from_yaml(args.config)

    overrides_model, overrides_output = {}, {}
    if args.device:
        overrides_model["device"] = args.device
    if args.batch_size:
        overrides_model["batch_size"] = args.batch_size
    if args.output_dir:
        overrides_output["output_dir"] = args.output_dir
    if args.output_format:
        overrides_output["output_format"] = args.output_format
    if args.save_overlay:
        overrides_output["save_overlay"] = True
    if args.overlay_alpha is not None:
        overrides_output["overlay_alpha"] = args.overlay_alpha
    if args.overlay_max_edge is not None:
        overrides_output["overlay_max_edge"] = args.overlay_max_edge

    if overrides_model:
        cfg = replace(cfg, model=replace(cfg.model, **overrides_model))
    if overrides_output:
        cfg = replace(cfg, output=replace(cfg.output, **overrides_output))

    wsi_paths = [p for p in args.input if p.exists()]
    if not wsi_paths:
        print("[ERROR] No valid input files.", file=sys.stderr)
        return 1

    pipeline = SegmentationPipeline(cfg)
    print(f"Output format: {cfg.output.output_format}")
    results = pipeline.run(wsi_paths)

    success = sum(1 for r in results if r.success)
    print(f"\n{'='*60}")
    print(f"Completed: {success}/{len(results)} slides succeeded.")
    for r in results:
        status = "✓" if r.success else "✗"
        print(f"  {status} {r.wsi_path.name}")
        if r.success:
            print(f"      → {r.write_result.path}  ({r.elapsed_seconds:.1f}s)")
        else:
            for err in r.errors:
                print(f"      → {err}")

    return 0 if all(r.success for r in results) else 1


# ── histo-embed ───────────────────────────────────────────────────────────────

def _embed_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="histo-embed",
                                description="Foundation model feature extraction.")
    p.add_argument("-i", "--input",   nargs="+", type=Path, required=True)
    p.add_argument("-o", "--output-dir", type=Path, default=Path("embeddings"))
    p.add_argument("--model",         default="uni", help="uni | vit | ctranspath")
    p.add_argument("--model-path",    type=Path, default=None)
    p.add_argument("--target-mpp",    type=float, default=0.5)
    p.add_argument("--batch-size",    type=int, default=32)
    p.add_argument("--device",        default="cpu")
    p.add_argument("--embedding-dim", type=int, default=1024)
    return p


def main_embed(argv: list[str] | None = None) -> int:
    args = _embed_parser().parse_args(argv)

    from histocoreml.config import FoundationConfig  # noqa: PLC0415
    from histocoreml.foundation import EmbeddingPipeline, get_encoder  # noqa: PLC0415

    cfg = FoundationConfig(
        model_name=args.model,
        model_path=args.model_path,
        embedding_dim=args.embedding_dim,
        target_mpp=args.target_mpp,
        batch_size=args.batch_size,
        device=args.device,
    )

    encoder  = get_encoder(cfg)
    pipeline = EmbeddingPipeline(cfg, encoder)
    results  = pipeline.run(args.input, output_dir=args.output_dir)

    success = sum(1 for r in results if r.success)
    print(f"Embedded: {success}/{len(results)} slides.")
    return 0 if success == len(results) else 1


# ── histo-extract ─────────────────────────────────────────────────────────────

def _extract_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="histo-extract",
                                description="Biomarker extraction from WSI + mask.")
    p.add_argument("-i", "--input",  type=Path, required=True)
    p.add_argument("--mask",         type=Path, default=None)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--tasks",        nargs="+",
                   default=["cell_density", "nuclei_morphology", "spatial_graph"])
    p.add_argument("--target-mpp",   type=float, default=0.25)
    return p


def main_extract(argv: list[str] | None = None) -> int:
    import numpy as np  # noqa: PLC0415

    from histocoreml.biomarkers import BiomarkerExtractor  # noqa: PLC0415
    from histocoreml.config import BiomarkerConfig  # noqa: PLC0415

    args = _extract_parser().parse_args(argv)

    mask = None
    if args.mask:
        suffix = args.mask.suffix.lower()
        if suffix == ".npy":
            mask = np.load(str(args.mask))
        else:
            from PIL import Image  # noqa: PLC0415
            mask = (np.array(Image.open(args.mask).convert("L")) > 127).astype(np.uint8)

    cfg       = BiomarkerConfig(tasks=args.tasks, target_mpp=args.target_mpp)
    extractor = BiomarkerExtractor(cfg)
    report    = extractor.run(args.input, mask=mask)

    out = args.output or Path("biomarkers") / f"{args.input.stem}_biomarkers.json"
    report.save(out)
    print(f"Report saved → {out}")
    return 0 if report.success else 1


# ── histo-train ───────────────────────────────────────────────────────────────

def _train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="histo-train",
                                description="Train a segmentation model.")
    p.add_argument("--images",     type=Path, required=True, help="Training image directory.")
    p.add_argument("--masks",      type=Path, required=True, help="Training mask directory.")
    p.add_argument("--val-images", type=Path, default=None)
    p.add_argument("--val-masks",  type=Path, default=None)
    p.add_argument("--arch",       default="unet")
    p.add_argument("--encoder",    default="resnet50")
    p.add_argument("--loss",       default="dice_bce")
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    return p


def main_train(argv: list[str] | None = None) -> int:
    args = _train_parser().parse_args(argv)

    from histocoreml.config import TrainingConfig  # noqa: PLC0415
    from histocoreml.training import SegmentationTrainer, build_train_dataloader  # noqa: PLC0415

    cfg = TrainingConfig(
        architecture=args.arch,
        encoder=args.encoder,
        loss=args.loss,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
    )

    train_loader = build_train_dataloader(args.images, args.masks, batch_size=cfg.batch_size)
    val_images = args.val_images or args.images
    val_masks = args.val_masks or args.masks
    val_loader = build_train_dataloader(
        val_images,
        val_masks,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    trainer = SegmentationTrainer(cfg)
    history = trainer.fit(train_loader, val_loader)

    best_dice = max(history["val_dice"]) if history["val_dice"] else 0.0
    print(f"Training complete. Best Dice: {best_dice:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main_segment())
