"""Which CLI entry points accept an experiment config, and how they fail.

`histo-segment` and `histo-train` are both config-driven; the config schema has
no section describing embedding or biomarker runs, so those two stay
flag-driven. These tests pin that boundary so it stays deliberate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from histocoreml.cli import _embed_parser, _extract_parser, _segment_parser, _train_parser

_SHIPPED_INFERENCE_CONFIGS = ["default.yaml", "gpu.yaml", "geojson.yaml", "rle_plain.yaml"]
_SHIPPED_TRAINING_CONFIGS = ["hubmap_glomeruli.yaml", "breast_tumor.yaml"]


def _options(parser) -> set[str]:
    return {opt for action in parser._actions for opt in action.option_strings}


class TestConfigFlagCoverage:
    def test_segment_accepts_a_config(self):
        assert {"-c", "--config"} <= _options(_segment_parser())

    def test_train_accepts_a_config(self):
        assert {"-c", "--config"} <= _options(_train_parser())

    @pytest.mark.parametrize("parser_fn", [_embed_parser, _extract_parser])
    def test_embed_and_extract_are_flag_driven(self, parser_fn):
        # No config section describes these runs, so they take explicit flags.
        assert "--config" not in _options(parser_fn())


class TestTrainArgumentValidation:
    def test_no_config_and_no_directories_is_an_error(self, capsys):
        assert main_train_exit([]) == 1
        assert "Provide --config, or both --images and --masks" in capsys.readouterr().err

    def test_inference_only_without_config_is_an_error(self, capsys):
        assert main_train_exit(["--inference-only"]) == 1
        assert "--inference-only requires --config" in capsys.readouterr().err

    def test_images_without_masks_is_an_error(self, capsys):
        assert main_train_exit(["--images", "imgs"]) == 1
        assert "Provide --config" in capsys.readouterr().err

    def test_missing_config_file_is_reported(self, tmp_path: Path, capsys):
        assert main_train_exit(["-c", str(tmp_path / "absent.yaml")]) == 1
        assert "Config file not found" in capsys.readouterr().err


def main_train_exit(argv: list[str]) -> int:
    from histocoreml.cli import main_train

    return main_train(argv)


class TestSegmentAcceptsEveryShippedConfig:
    @pytest.mark.parametrize("name", _SHIPPED_INFERENCE_CONFIGS)
    def test_inference_configs_parse(self, name: str):
        from histocoreml.config import SegmentationPipelineConfig

        cfg = SegmentationPipelineConfig.from_yaml(Path("configs") / name)
        assert cfg.model.model_path

    @pytest.mark.parametrize("name", _SHIPPED_TRAINING_CONFIGS)
    def test_training_configs_are_rejected_for_segmentation(self, name: str, capsys):
        from histocoreml.cli import main_segment

        # They name no weights, so histo-segment must say so rather than crash.
        exit_code = main_segment(["-c", str(Path("configs") / name), "-i", "slide.svs"])

        assert exit_code == 1
        assert "No model weights" in capsys.readouterr().err

    @pytest.mark.parametrize("name", _SHIPPED_TRAINING_CONFIGS)
    def test_training_configs_load_for_training(self, name: str):
        from histocoreml.config import ExperimentConfig

        cfg = ExperimentConfig.from_yaml(Path("configs") / name)
        assert cfg.training_config().epochs > 0


class TestSegmentFlagPlumbing:
    """Flags histo-segment advertises must actually reach the pipeline."""

    def test_normalise_flag_reaches_the_dataloader(self, tmp_path: Path, monkeypatch):
        from dataclasses import replace

        from histocoreml.config import SegmentationPipelineConfig

        cfg = SegmentationPipelineConfig.from_yaml("configs/default.yaml")

        # --normalise used to be parsed and then dropped on the floor. It lives
        # on ModelConfig: whether patches need stain normalisation is fixed by
        # how the model was trained, not by how the slide is tiled.
        assert cfg.model.stain_normalise is False
        updated = replace(cfg, model=replace(cfg.model, stain_normalise=True))
        assert updated.model.stain_normalise is True

    def test_missing_inputs_are_reported_not_silently_skipped(self, tmp_path: Path, capsys):
        from histocoreml.cli import main_segment

        exit_code = main_segment(["-c", "configs/default.yaml", "-i", str(tmp_path / "absent.svs")])

        err = capsys.readouterr().err
        assert exit_code == 1
        assert "Input not found, skipping" in err
        assert "absent.svs" in err

    def test_all_inputs_missing_is_an_error(self, tmp_path: Path, capsys):
        from histocoreml.cli import main_segment

        exit_code = main_segment(
            ["-c", "configs/default.yaml", "-i", str(tmp_path / "a.svs"), str(tmp_path / "b.svs")]
        )

        assert exit_code == 1
        assert "No valid input files" in capsys.readouterr().err
