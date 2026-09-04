"""Tests for pre-extracting WSI patches to disk."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from histocoreml.config import TilingConfig
from histocoreml.pipelines.training.segmentation import extract_patches_to_disk
from histocoreml.training.dataset import RLEMaskProvider, SegmentationDataset


def _rle(mask: np.ndarray) -> str:
    pixels = mask.flatten(order="F")
    padded = np.concatenate([[0], pixels, [0]])
    runs = np.where(padded[1:] != padded[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    return " ".join(str(x) for x in runs)


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Two 512x512 slides whose left half is tissue, with a small mask."""
    root = tmp_path / "data"
    (root / "train").mkdir(parents=True)

    rows = ["id,encoding"]
    for i in range(2):
        image = np.full((512, 512, 3), 250, dtype=np.uint8)
        image[:, :256] = 120
        tifffile.imwrite(root / "train" / f"slide_{i}.tiff", image)

        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[130:200, 130:200] = 1
        rows.append(f"slide_{i},{_rle(mask)}")

    (root / "train.csv").write_text("\n".join(rows) + "\n")
    return root


@pytest.fixture
def tiling_cfg() -> TilingConfig:
    return TilingConfig(overlap=0, tissue_threshold=0.5)


def _extract(root: Path, out: Path, tiling_cfg: TilingConfig, **kwargs):
    return extract_patches_to_disk(
        slide_dir=root / "train",
        mask_provider=RLEMaskProvider.from_csv(root / "train.csv"),
        output_dir=out,
        tiling_cfg=tiling_cfg,
        patch_size=128,
        target_mpp=0.5,
        **kwargs,
    )


class TestExtraction:
    def test_writes_matching_image_and_mask_pairs(self, dataset_root, tmp_path, tiling_cfg):
        stats = _extract(dataset_root, tmp_path / "out", tiling_cfg)

        images = sorted(p.name for p in stats.images_dir.glob("*.png"))
        masks = sorted(p.name for p in stats.masks_dir.glob("*.png"))

        assert stats.patches_written > 0
        assert images == masks  # every image has its mask, same stem
        assert len(images) == stats.patches_written

    def test_stats_report_slides(self, dataset_root, tmp_path, tiling_cfg):
        stats = _extract(dataset_root, tmp_path / "out", tiling_cfg)

        assert stats.slides_processed == 2
        assert stats.slides_skipped == 0

    def test_filenames_encode_slide_and_position(self, dataset_root, tmp_path, tiling_cfg):
        stats = _extract(dataset_root, tmp_path / "out", tiling_cfg)

        for path in stats.images_dir.glob("*.png"):
            assert path.stem.startswith("slide_")
            assert "_x" in path.stem and "_y" in path.stem

    def test_manifest_records_the_settings(self, dataset_root, tmp_path, tiling_cfg):
        out = tmp_path / "out"
        stats = _extract(dataset_root, out, tiling_cfg)

        manifest = json.loads((out / "extraction_manifest.json").read_text())

        assert manifest["patch_size"] == 128
        assert manifest["target_mpp"] == 0.5
        assert manifest["overlap"] == 0
        assert manifest["patches_written"] == stats.patches_written
        assert sum(manifest["patches_per_slide"].values()) == stats.patches_written

    def test_masks_are_binary_png(self, dataset_root, tmp_path, tiling_cfg):
        stats = _extract(dataset_root, tmp_path / "out", tiling_cfg)

        for path in list(stats.masks_dir.glob("*.png"))[:5]:
            values = set(np.unique(np.array(Image.open(path))).tolist())
            assert values <= {0, 255}

    def test_images_are_rgb_at_patch_size(self, dataset_root, tmp_path, tiling_cfg):
        stats = _extract(dataset_root, tmp_path / "out", tiling_cfg)

        for path in list(stats.images_dir.glob("*.png"))[:5]:
            assert np.array(Image.open(path)).shape == (128, 128, 3)


class TestExtractionOptions:
    def test_slide_ids_restricts_output(self, dataset_root, tmp_path, tiling_cfg):
        stats = _extract(dataset_root, tmp_path / "out", tiling_cfg, slide_ids=["slide_0"])

        assert stats.slides_processed == 1
        assert all(p.stem.startswith("slide_0") for p in stats.images_dir.glob("*.png"))

    def test_limit_caps_the_number_written(self, dataset_root, tmp_path, tiling_cfg):
        stats = _extract(dataset_root, tmp_path / "out", tiling_cfg, limit=3)

        assert stats.patches_written == 3

    def test_skip_empty_masks_drops_background_only_patches(
        self, dataset_root, tmp_path, tiling_cfg
    ):
        everything = _extract(dataset_root, tmp_path / "all", tiling_cfg)
        foreground = _extract(dataset_root, tmp_path / "fg", tiling_cfg, skip_empty_masks=True)

        assert foreground.patches_written < everything.patches_written
        assert foreground.patches_written > 0
        for path in foreground.masks_dir.glob("*.png"):
            assert np.array(Image.open(path)).max() > 0


class TestExtractionFidelity:
    def test_extracted_patches_match_on_the_fly_training(self, dataset_root, tmp_path, tiling_cfg):
        """Pre-extraction must not change what the model sees."""
        stats = _extract(dataset_root, tmp_path / "out", tiling_cfg)

        dataset = SegmentationDataset(
            slide_dir=dataset_root / "train",
            mask_provider=RLEMaskProvider.from_csv(dataset_root / "train.csv"),
            tiling_cfg=tiling_cfg,
            patch_size=128,
            target_mpp=0.5,
        )

        for idx in range(len(dataset)):
            slide_id, coord = dataset.patch_list[idx]
            sample = dataset[idx]
            name = f"{slide_id}_x{coord.x}_y{coord.y}.png"

            live_image = (sample["image"].numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
            live_mask = (sample["mask"].numpy().squeeze() > 0.5).astype(np.uint8)

            np.testing.assert_array_equal(np.array(Image.open(stats.images_dir / name)), live_image)
            np.testing.assert_array_equal(
                (np.array(Image.open(stats.masks_dir / name)) > 127).astype(np.uint8),
                live_mask,
            )

    def test_output_feeds_build_train_dataloader(self, dataset_root, tmp_path, tiling_cfg):
        """The layout is what histo-train --images/--masks consumes."""
        from histocoreml.training import build_train_dataloader

        stats = _extract(dataset_root, tmp_path / "out", tiling_cfg)

        loader = build_train_dataloader(
            stats.images_dir, stats.masks_dir, batch_size=2, num_workers=0
        )
        batch = next(iter(loader))

        assert batch["image"].shape == (2, 3, 128, 128)
        assert batch["mask"].shape == (2, 1, 128, 128)
