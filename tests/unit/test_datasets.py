"""Tests for the on-the-fly WSI training stack.

Covers the pieces that used to live in ``scripts/hubmap_segmentation.py``:
the RLE mask provider, the WSI patch dataset, the tissue coordinate filter,
the declarative transform builder and the experiment config loader.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
import yaml

from histocoreml.config import ExperimentConfig, TilingConfig
from histocoreml.preprocessing.grid_generator import generate_patch_coords, generate_patch_grid
from histocoreml.preprocessing.patch_coord import PatchCoord
from histocoreml.preprocessing.patch_utils import ensure_rgb, filter_coords_by_tissue
from histocoreml.training.dataset import RLEMaskProvider, SegmentationDataset
from histocoreml.training.transforms import build_augmentation_pair, build_transforms


def _rle_encode_fortran(mask: np.ndarray) -> str:
    """Encode a mask the way HuBMAP does: column-major, 1-indexed starts."""
    pixels = mask.flatten(order="F")
    padded = np.concatenate([[0], pixels, [0]])
    runs = np.where(padded[1:] != padded[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    return " ".join(str(x) for x in runs)


@pytest.fixture
def tissue_slide(tmp_path: Path) -> Path:
    """A 512x512 TIFF whose left half is tissue and right half is white glass."""
    path = tmp_path / "slide_a.tiff"
    level0 = np.full((512, 512, 3), 250, dtype=np.uint8)  # white background
    level0[:, :256] = 120  # tissue on the left half
    with tifffile.TiffWriter(path) as writer:
        writer.write(level0)
    return path


class TestEnsureRGB:
    def test_grayscale_becomes_three_channels(self):
        assert ensure_rgb(np.zeros((8, 8), dtype=np.uint8)).shape == (8, 8, 3)

    def test_rgba_alpha_dropped(self):
        assert ensure_rgb(np.zeros((8, 8, 4), dtype=np.uint8)).shape == (8, 8, 3)

    def test_rgb_passes_through(self):
        patch = np.full((8, 8, 3), 7, dtype=np.uint8)
        np.testing.assert_array_equal(ensure_rgb(patch), patch)


class TestFilterCoordsByTissue:
    def _coords(self, n: int = 4, patch_size: int = 128) -> list[PatchCoord]:
        return [
            PatchCoord(
                x=i * patch_size,
                y=0,
                level=0,
                patch_size=patch_size,
                col_idx=i,
                row_idx=0,
            )
            for i in range(n)
        ]

    def test_keeps_only_tissue_regions(self):
        thumbnail = np.full((64, 64, 3), 250, dtype=np.uint8)
        thumbnail[:, :32] = 100  # tissue in the left half
        cfg = TilingConfig(overlap=0, tissue_threshold=0.5)

        kept = filter_coords_by_tissue(self._coords(), thumbnail, (512, 512), cfg)

        assert [c.col_idx for c in kept] == [0, 1]

    def test_empty_inputs_return_empty(self):
        cfg = TilingConfig(overlap=0)
        empty_thumb = np.empty((0, 0, 3), np.uint8)
        assert filter_coords_by_tissue([], np.zeros((4, 4, 3), np.uint8), (64, 64), cfg) == []
        assert filter_coords_by_tissue(self._coords(), empty_thumb, (64, 64), cfg) == []

    def test_all_background_keeps_nothing(self):
        thumbnail = np.full((64, 64, 3), 250, dtype=np.uint8)
        cfg = TilingConfig(overlap=0, tissue_threshold=0.5)
        assert filter_coords_by_tissue(self._coords(), thumbnail, (512, 512), cfg) == []


class TestGeneratePatchGrid:
    def test_matches_model_config_wrapper(self, tissue_slide: Path):
        from histocoreml.config import ModelConfig
        from histocoreml.io.factory import get_reader

        with get_reader(tissue_slide) as reader:
            metadata = reader.get_metadata()

        tiling = TilingConfig(overlap=0)
        model_cfg = ModelConfig(model_path=Path("unused.pt"), patch_size=128, target_mpp=0.5)

        via_wrapper = generate_patch_coords(metadata, model_cfg, tiling, slide_id="s")
        direct = generate_patch_grid(
            metadata, patch_size=128, target_mpp=0.5, tiling_cfg=tiling, slide_id="s"
        )

        assert via_wrapper == direct

    def test_overlap_larger_than_patch_raises(self, tissue_slide: Path):
        from histocoreml.io.factory import get_reader

        with get_reader(tissue_slide) as reader:
            metadata = reader.get_metadata()

        with pytest.raises(ValueError, match="must be larger than overlap"):
            generate_patch_grid(
                metadata,
                patch_size=64,
                target_mpp=0.5,
                tiling_cfg=TilingConfig(overlap=64),
            )


class TestRLEMaskProvider:
    def test_roundtrip_decode(self):
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[4:8, 4:8] = 1
        provider = RLEMaskProvider({"slide": _rle_encode_fortran(mask)})

        np.testing.assert_array_equal(provider.get_mask("slide", (16, 16)), mask)

    def test_slide_ids(self):
        provider = RLEMaskProvider({"a": "1 1", "b": "1 1"})
        assert sorted(provider.slide_ids()) == ["a", "b"]

    def test_cache_is_bounded(self):
        encodings = dict.fromkeys(("a", "b", "c"), "1 1")
        provider = RLEMaskProvider(encodings, cache_size=2)
        for name in encodings:
            provider.get_mask(name, (4, 4))
        assert len(provider._cache) == 2

    def test_from_csv(self, tmp_path: Path):
        csv_path = tmp_path / "train.csv"
        csv_path.write_text("id,encoding\nslide_a,1 4\nslide_b,2 2\n")

        provider = RLEMaskProvider.from_csv(csv_path)

        assert sorted(provider.slide_ids()) == ["slide_a", "slide_b"]
        assert provider.get_mask("slide_a", (2, 2)).sum() == 4

    def test_from_csv_rejects_wrong_columns(self, tmp_path: Path):
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("slide,rle\na,1 1\n")

        with pytest.raises(ValueError, match="missing column"):
            RLEMaskProvider.from_csv(csv_path)


class TestSegmentationDataset:
    def _dataset(self, slide_dir: Path, **kwargs) -> SegmentationDataset:
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[130:200, 130:200] = 1  # sits inside the patch starting at (128, 128)
        provider = RLEMaskProvider({"slide_a": _rle_encode_fortran(mask)})
        return SegmentationDataset(
            slide_dir=slide_dir,
            mask_provider=provider,
            tiling_cfg=TilingConfig(overlap=0, tissue_threshold=0.5),
            patch_size=128,
            target_mpp=0.5,
            **kwargs,
        )

    def test_indexes_only_tissue_patches(self, tissue_slide: Path):
        dataset = self._dataset(tissue_slide.parent)

        assert len(dataset) > 0
        # Tissue occupies the left half, so no patch starts beyond x=256.
        assert all(coord.x < 256 for _, coord in dataset.patch_list)

    def test_item_shapes_and_ranges(self, tissue_slide: Path):
        dataset = self._dataset(tissue_slide.parent)
        item = dataset[0]

        assert item["image"].shape == (3, 128, 128)
        assert item["mask"].shape == (1, 128, 128)
        assert 0.0 <= float(item["image"].min()) and float(item["image"].max()) <= 1.0
        assert set(np.unique(item["mask"].numpy())) <= {0.0, 1.0}

    def test_mask_aligns_with_slide_coordinates(self, tissue_slide: Path):
        dataset = self._dataset(tissue_slide.parent)

        # The mask covers rows/cols 130-200, so the patch at (128, 128) must
        # carry foreground while the patch at (0, 0) must not.
        by_origin = {(c.x, c.y): i for i, (_, c) in enumerate(dataset.patch_list)}
        assert dataset[by_origin[(128, 128)]]["mask"].sum() > 0
        assert dataset[by_origin[(0, 0)]]["mask"].sum() == 0

    def test_slide_ids_filter_applies(self, tissue_slide: Path):
        dataset = self._dataset(tissue_slide.parent, slide_ids=["missing_slide"])
        assert len(dataset) == 0

    def test_subset_truncates(self, tissue_slide: Path):
        dataset = self._dataset(tissue_slide.parent)
        dataset.subset(2)
        assert len(dataset) == 2

    def test_missing_slide_is_skipped(self, tmp_path: Path):
        dataset = self._dataset(tmp_path / "empty")
        assert len(dataset) == 0


class TestBuildTransforms:
    def test_none_and_empty_return_none(self):
        assert build_transforms(None) is None
        assert build_transforms([]) is None

    def test_unknown_transform_is_skipped(self):
        pytest.importorskip("albumentations")
        assert build_transforms([{"name": "NoSuchTransform"}]) is None

    def test_builds_pipeline(self):
        pytest.importorskip("albumentations")
        transform = build_transforms([{"name": "HorizontalFlip", "params": {"p": 1.0}}])

        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[:, 0] = 255
        out = transform(image=image, mask=np.zeros((8, 8), dtype=np.uint8))

        assert out["image"][:, -1].max() == 255  # flipped

    def test_disabled_block_yields_no_transforms(self):
        train, valid = build_augmentation_pair(
            {"enabled": False, "train": [{"name": "HorizontalFlip"}]}
        )
        assert train is None and valid is None


class TestExperimentConfig:
    @pytest.fixture
    def config_path(self, tmp_path: Path) -> Path:
        document = {
            "experiment": {"name": "exp", "output_dir": str(tmp_path / "out"), "seed": 7},
            "data": {
                "train_dir": "data/train",
                "train_csv": "data/train.csv",
                "patch_size": 256,
                "patch_overlap": 0.25,
                "batch_size": 4,
                "num_workers": 2,
                "tissue_threshold": 0.1,
            },
            "model": {"name": "unet++", "encoder_name": "resnet34", "encoder_pretrained": False},
            "training": {
                "epochs": 3,
                "optimizer": {"name": "AdamW", "lr": 2e-4, "weight_decay": 1e-6},
                "loss": {"name": "dice_bce", "dice_weight": 0.7, "bce_weight": 0.3},
                "early_stopping": {"patience": 5},
                "mixed_precision": False,
            },
            "inference": {"patch_size": 512, "threshold": 0.7, "output_format": "npy"},
        }
        path = tmp_path / "exp.yaml"
        path.write_text(yaml.safe_dump(document))
        return path

    def test_training_config_mapping(self, config_path: Path):
        cfg = ExperimentConfig.from_yaml(config_path).training_config()

        assert cfg.architecture == "unet++"
        assert cfg.encoder == "resnet34"
        assert cfg.pretrained is False
        assert cfg.epochs == 3
        assert cfg.learning_rate == pytest.approx(2e-4)
        assert cfg.optimizer == "adamw"
        assert cfg.early_stopping_patience == 5
        assert cfg.mixed_precision is False

    def test_overlap_fraction_becomes_pixels(self, config_path: Path):
        cfg = ExperimentConfig.from_yaml(config_path)
        assert cfg.tiling_config().overlap == 64  # 256 * 0.25

    def test_loss_spec_carries_weights(self, config_path: Path):
        spec = ExperimentConfig.from_yaml(config_path).loss_spec()
        assert spec == {"name": "dice_bce", "dice_weight": 0.7, "bce_weight": 0.3}

    def test_inference_model_config_carries_architecture(self, config_path: Path):
        cfg = ExperimentConfig.from_yaml(config_path)
        model_cfg = cfg.inference_model_config(Path("best.pth"))

        assert model_cfg.architecture == "unet++"
        assert model_cfg.encoder == "resnet34"
        assert model_cfg.patch_size == 512
        assert model_cfg.threshold == pytest.approx(0.7)
        assert model_cfg.device != "auto"  # resolved to a real device

    def test_output_config_uses_inference_section(self, config_path: Path):
        cfg = ExperimentConfig.from_yaml(config_path)
        assert cfg.output_config().output_format == "npy"

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            ExperimentConfig.from_yaml(tmp_path / "nope.yaml")

    def test_missing_section_raises(self, tmp_path: Path):
        path = tmp_path / "partial.yaml"
        path.write_text(yaml.safe_dump({"experiment": {"name": "x"}}))

        with pytest.raises(ValueError, match="missing required section"):
            ExperimentConfig.from_yaml(path)

    def test_shipped_configs_parse(self):
        cfg = ExperimentConfig.from_yaml("configs/hubmap_glomeruli.yaml")
        assert cfg.training_config().architecture == "unet++"
        assert cfg.tiling_config().overlap == 256


class TestDatasetNaming:
    """PatchDirectoryDataset was renamed from HistoSegDataset."""

    def test_new_name_is_exported(self):
        from histocoreml.training import PatchDirectoryDataset

        assert PatchDirectoryDataset.__name__ == "PatchDirectoryDataset"

    def test_deprecated_alias_still_resolves(self):
        from histocoreml.training import HistoSegDataset, PatchDirectoryDataset

        assert HistoSegDataset is PatchDirectoryDataset

    def test_both_datasets_live_in_one_module(self):
        from histocoreml.training import dataset

        assert hasattr(dataset, "SegmentationDataset")
        assert hasattr(dataset, "PatchDirectoryDataset")
