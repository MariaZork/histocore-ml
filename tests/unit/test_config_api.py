"""Regression tests for the config/CLI API traps.

Each test pins down a mistake that previously shipped: calling ``from_yaml``
on the wrong class, config files whose schema no longer matched the loader,
and two different classes both exported as ``ModelConfig``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from histocoreml.config import (
    ExperimentConfig,
    PipelineConfig,
    SegmentationPipelineConfig,
)

_INFERENCE_CONFIGS = ["default.yaml", "gpu.yaml", "geojson.yaml", "rle_plain.yaml"]
_TRAINING_CONFIGS = ["hubmap_glomeruli.yaml", "breast_tumor.yaml"]
_ALL_CONFIGS = _INFERENCE_CONFIGS + _TRAINING_CONFIGS


class TestFromYamlOwnership:
    def test_from_yaml_lives_on_segmentation_config(self):
        assert hasattr(SegmentationPipelineConfig, "from_yaml")

    def test_base_pipeline_config_has_no_from_yaml(self):
        # The base class is a plain container; callers must name the concrete
        # config. Adding from_yaml here would resurrect the original confusion.
        assert not hasattr(PipelineConfig, "from_yaml")

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            SegmentationPipelineConfig.from_yaml(tmp_path / "absent.yaml")

    def test_document_with_neither_schema_raises(self, tmp_path: Path):
        path = tmp_path / "empty.yaml"
        path.write_text("output: {}\n")

        with pytest.raises(ValueError, match="missing the required 'model' section"):
            SegmentationPipelineConfig.from_yaml(path)


class TestShippedConfigs:
    """Every config in configs/ uses the one nested experiment schema."""

    @pytest.mark.parametrize("name", _ALL_CONFIGS)
    def test_all_configs_load_as_experiments(self, name: str):
        cfg = ExperimentConfig.from_yaml(Path("configs") / name)

        assert cfg.name
        assert cfg.patch_size > 0
        assert cfg.target_mpp > 0

    @pytest.mark.parametrize("name", _INFERENCE_CONFIGS)
    def test_inference_configs_drive_histo_segment(self, name: str):
        cfg = SegmentationPipelineConfig.from_yaml(Path("configs") / name)

        assert cfg.model.model_path
        assert cfg.model.batch_size > 0
        assert cfg.tiling.overlap < cfg.model.patch_size
        assert cfg.output.output_format

    @pytest.mark.parametrize("name", _TRAINING_CONFIGS)
    def test_training_configs_name_no_weights(self, name: str):
        # Their checkpoint is produced by training, so inference needs one passed in.
        with pytest.raises(ValueError, match="No model weights"):
            SegmentationPipelineConfig.from_yaml(Path("configs") / name)

    @pytest.mark.parametrize("name", _TRAINING_CONFIGS)
    def test_training_configs_accept_an_explicit_checkpoint(self, name: str, tmp_path: Path):
        cfg = ExperimentConfig.from_yaml(Path("configs") / name)
        checkpoint = tmp_path / "best.pth"

        model_cfg = cfg.segmentation_config(checkpoint).model

        assert model_cfg.model_path == checkpoint
        assert model_cfg.architecture == cfg.model["name"]

    def test_output_formats_are_registered(self):
        from histocoreml.output.factory import _WRITER_REGISTRY

        for name in _INFERENCE_CONFIGS:
            cfg = SegmentationPipelineConfig.from_yaml(Path("configs") / name)
            assert cfg.output.output_format.lower() in _WRITER_REGISTRY

    def test_gpu_config_differs_from_default_where_it_should(self):
        default = SegmentationPipelineConfig.from_yaml("configs/default.yaml")
        gpu = SegmentationPipelineConfig.from_yaml("configs/gpu.yaml")

        assert gpu.model.device == "cuda:0"
        assert gpu.model.batch_size > default.model.batch_size
        assert gpu.tiling.num_workers > default.tiling.num_workers
        assert gpu.tiling.prefetch_factor > default.tiling.prefetch_factor
        # Geometry is deliberately identical to default.
        assert gpu.model.patch_size == default.model.patch_size
        assert gpu.tiling.overlap == default.tiling.overlap


class TestFlatSchemaStillAccepted:
    """The flat schema predates the nested one; user configs may still use it."""

    def _write_flat(self, tmp_path: Path) -> Path:
        path = tmp_path / "flat.yaml"
        path.write_text(
            "log_level: INFO\n"
            "model:\n"
            f"  model_path: {tmp_path / 'model.pt'}\n"
            "  patch_size: 256\n"
            "  target_mpp: 0.5\n"
            "  device: cpu\n"
            "  batch_size: 4\n"
            "tiling:\n"
            "  overlap: 64\n"
            "output:\n"
            f"  output_dir: {tmp_path / 'out'}\n"
            "  output_format: npy\n"
        )
        return path

    def test_flat_config_loads(self, tmp_path: Path):
        cfg = SegmentationPipelineConfig.from_yaml(self._write_flat(tmp_path))

        assert cfg.model.patch_size == 256
        assert cfg.model.batch_size == 4
        assert cfg.tiling.overlap == 64
        assert cfg.output.output_format == "npy"


class TestExperimentConfigCoversInferenceSettings:
    """The nested schema must express everything the flat one could."""

    def test_every_output_knob_is_carried(self):
        cfg = SegmentationPipelineConfig.from_yaml("configs/default.yaml").output

        assert cfg.compression == "lzw"
        assert cfg.save_thumbnail is True
        assert cfg.save_overlay is True
        assert cfg.overlay_alpha == pytest.approx(0.40)
        assert cfg.overlay_max_edge == 2048
        assert cfg.downsample_factor is None

    def test_rle_subformat_is_carried(self):
        cfg = SegmentationPipelineConfig.from_yaml("configs/rle_plain.yaml").output

        assert cfg.output_format == "rle"
        assert cfg.rle_subformat == "plain"

    def test_prefetch_factor_is_carried(self):
        cfg = SegmentationPipelineConfig.from_yaml("configs/gpu.yaml").tiling

        assert cfg.prefetch_factor == 4

    def test_device_is_carried(self):
        cfg = SegmentationPipelineConfig.from_yaml("configs/gpu.yaml").model

        assert cfg.device == "cuda:0"


class TestModelConfigDisambiguation:
    def test_inference_and_architecture_configs_are_distinct(self):
        from histocoreml.config import ModelConfig as InferenceModelConfig
        from histocoreml.models import ArchitectureConfig

        assert InferenceModelConfig is not ArchitectureConfig

    def test_deprecated_alias_still_resolves(self):
        from histocoreml.models import ArchitectureConfig, ModelConfig

        assert ModelConfig is ArchitectureConfig

    def test_architecture_config_needs_no_weights(self):
        from histocoreml.models import ArchitectureConfig

        # Constructible with no arguments — it describes a net to build.
        assert ArchitectureConfig().architecture

    def test_inference_config_requires_model_path(self):
        from histocoreml.config import ModelConfig

        with pytest.raises(TypeError):
            ModelConfig()

    def test_top_level_convenience_exports_resolve(self):
        # These used to vanish silently: histocoreml.models did not export
        # ModelConfig, and __init__ swallowed the resulting ImportError.
        import histocoreml

        for name in ("get_model", "ArchitectureConfig", "list_models", "create_model_for_organ"):
            assert hasattr(histocoreml, name), name
            assert name in histocoreml.__all__


class TestStainNormalisationPlacement:
    """Stain normalisation is a property of the model, not of the tiling."""

    def test_model_config_carries_the_flag(self):
        from histocoreml.config import ModelConfig

        assert "stain_normalise" in ModelConfig.__dataclass_fields__

    def test_tiling_config_does_not(self):
        from histocoreml.config import TilingConfig

        # It used to live here, which implied changing overlap could change
        # whether patches were normalised.
        assert "normalise" not in TilingConfig.__dataclass_fields__

    def test_defaults_to_off(self):
        cfg = SegmentationPipelineConfig.from_yaml("configs/default.yaml")

        assert cfg.model.stain_normalise is False

    def test_configurable_from_yaml(self, tmp_path: Path):
        import yaml

        from histocoreml.config import ExperimentConfig

        document = {
            "experiment": {"name": "x", "output_dir": str(tmp_path)},
            "data": {"patch_size": 256},
            "model": {"checkpoint": str(tmp_path / "m.pt")},
            "inference": {"stain_normalise": True},
        }
        path = tmp_path / "exp.yaml"
        path.write_text(yaml.safe_dump(document))

        cfg = ExperimentConfig.from_yaml(path)

        assert cfg.inference_model_config().stain_normalise is True
