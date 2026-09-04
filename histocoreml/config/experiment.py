"""Loader for the nested experiment YAML schema.

The training configs in ``configs/`` describe a whole run in one file::

    experiment: {name, output_dir, seed}
    data:       {train_dir, train_csv, patch_size, augmentation, ...}
    model:      {name, encoder_name, encoder_pretrained}
    training:   {epochs, optimizer, loss, early_stopping, mixed_precision}
    inference:  {patch_size, batch_size, threshold, output_format}
    logging:    {log_dir, tensorboard}

:class:`ExperimentConfig` maps that document onto the typed configs the rest of
the package consumes, so training scripts stay declarative and never reach into
raw dictionaries.

Usage::

    cfg = ExperimentConfig.from_yaml("configs/hubmap_glomeruli.yaml")
    trainer = SegmentationTrainer(cfg.training_config())
    criterion = get_loss(**cfg.loss_spec())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from histocoreml.config import (
    ModelConfig,
    OutputConfig,
    PipelineConfig,
    SegmentationPipelineConfig,
    TilingConfig,
    TrainingConfig,
)

_DEFAULT_TARGET_MPP = 0.5
"""20× objective — the resolution HuBMAP-style datasets are scanned at.

Used when a config omits ``data.target_mpp``, which is the common case for
datasets whose TIFFs carry no MPP metadata at all.
"""


@dataclass
class ExperimentConfig(PipelineConfig):
    """A parsed experiment YAML document.

    Inherits ``output_dir``, ``log_level``, ``num_workers`` and ``device`` from
    :class:`~histocoreml.config.PipelineConfig`, so it can drive a pipeline
    directly.

    Attributes:
        name: Experiment name, used for checkpoints and TensorBoard runs.
        seed: RNG seed applied to random / numpy / torch.
        raw:  The complete document, for keys with no typed home yet.
    """

    name: str = "experiment"
    seed: int = 42
    raw: dict[str, Any] = field(default_factory=dict)

    # ── Loading ───────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Path | str) -> ExperimentConfig:
        """Load and validate an experiment YAML file.

        Args:
            path: Path to the YAML document.

        Returns:
            The parsed configuration.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If a required top-level section is missing.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open() as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}

        # 'training' is optional: an inference-only config has nothing to train.
        missing = [key for key in ("experiment", "data", "model") if key not in raw]
        if missing:
            raise ValueError(f"{path} is missing required section(s): {missing}")

        experiment = raw["experiment"]
        return cls(
            name=experiment.get("name", path.stem),
            output_dir=Path(experiment.get("output_dir", "./outputs")),
            seed=int(experiment.get("seed", 42)),
            num_workers=int(raw.get("data", {}).get("num_workers", 4)),
            log_level=raw.get("logging", {}).get("level", "INFO"),
            raw=raw,
        )

    # ── Section accessors ─────────────────────────────────────────────────────

    @property
    def data(self) -> dict[str, Any]:
        """The ``data`` section."""
        return self.raw.get("data", {})

    @property
    def model(self) -> dict[str, Any]:
        """The ``model`` section."""
        return self.raw.get("model", {})

    @property
    def training(self) -> dict[str, Any]:
        """The ``training`` section."""
        return self.raw.get("training", {})

    @property
    def inference(self) -> dict[str, Any]:
        """The ``inference`` section (absent in training-only configs)."""
        return self.raw.get("inference", {})

    @property
    def logging(self) -> dict[str, Any]:
        """The ``logging`` section."""
        return self.raw.get("logging", {})

    @property
    def patch_size(self) -> int:
        """Training patch side length in pixels."""
        return int(self.data.get("patch_size", 512))

    @property
    def target_mpp(self) -> float:
        """Resolution to tile at, in µm/px."""
        return float(self.data.get("target_mpp", _DEFAULT_TARGET_MPP))

    @property
    def debug(self) -> bool:
        """Whether the config itself requests a debug run."""
        return bool(self.training.get("debug", False))

    @property
    def debug_samples(self) -> int:
        """Patches to keep per split in a debug run."""
        return int(self.training.get("debug_samples", 100))

    # ── Typed config builders ─────────────────────────────────────────────────

    def training_config(self) -> TrainingConfig:
        """Build the :class:`~histocoreml.config.TrainingConfig` for this run."""
        optimizer = self.training.get("optimizer", {})
        early_stopping = self.training.get("early_stopping", {})

        return TrainingConfig(
            architecture=self.model.get("name", "unet"),
            encoder=self.model.get("encoder_name", "resnet50"),
            pretrained=bool(self.model.get("encoder_pretrained", True)),
            loss=self.training.get("loss", {}).get("name", "dice_bce"),
            optimizer=str(optimizer.get("name", "adamw")).lower(),
            learning_rate=float(optimizer.get("lr", 1e-4)),
            weight_decay=float(optimizer.get("weight_decay", 1e-5)),
            epochs=int(self.training.get("epochs", 100)),
            batch_size=int(self.data.get("batch_size", 8)),
            patch_size=self.patch_size,
            target_mpp=self.target_mpp,
            num_workers=int(self.data.get("num_workers", 4)),
            mixed_precision=bool(self.training.get("mixed_precision", True)),
            early_stopping_patience=int(early_stopping.get("patience", 15)),
            checkpoint_dir=self.output_dir / "checkpoints",
            experiment_name=self.name,
        )

    def tiling_config(self) -> TilingConfig:
        """Build the :class:`~histocoreml.config.TilingConfig` for tiling slides.

        ``data.patch_overlap`` is a *fraction* of the patch size in these
        configs; :class:`TilingConfig` wants pixels.
        """
        overlap_fraction = float(self.data.get("patch_overlap", 0.5))
        return TilingConfig(
            overlap=int(self.patch_size * overlap_fraction),
            tissue_threshold=float(self.data.get("tissue_threshold", 0.05)),
            background_value=int(self.data.get("background_value", 230)),
            black_value=int(self.data.get("black_value", 10)),
            num_workers=int(self.data.get("num_workers", 4)),
            prefetch_factor=int(self.data.get("prefetch_factor", 2)),
        )

    def inference_tiling_config(self) -> TilingConfig:
        """Tiling config for inference, honouring ``inference.overlap`` if set."""
        base = self.tiling_config()
        if "overlap" not in self.inference:
            return base

        patch_size = int(self.inference.get("patch_size", self.patch_size))
        overlap = int(patch_size * float(self.inference["overlap"]))
        return TilingConfig(
            overlap=overlap,
            tissue_threshold=base.tissue_threshold,
            background_value=base.background_value,
            black_value=base.black_value,
            num_workers=base.num_workers,
            prefetch_factor=base.prefetch_factor,
        )

    @property
    def checkpoint(self) -> Path | None:
        """Weights named by ``model.checkpoint``, if the config declares any."""
        raw = self.model.get("checkpoint")
        return Path(raw) if raw else None

    def inference_model_config(self, checkpoint: Path | None = None) -> ModelConfig:
        """Build the :class:`~histocoreml.config.ModelConfig` for inference.

        Architecture and encoder are carried over from the ``model`` section so
        a training ``.pth`` checkpoint can be rebuilt by
        :class:`~histocoreml.backends.checkpoint_model.CheckpointModel`.

        Args:
            checkpoint: Weights to run — a TorchScript export, ONNX file, or a
                        training checkpoint. Defaults to ``model.checkpoint``,
                        which is how an inference-only config names its weights.

        Raises:
            ValueError: If no checkpoint is given and none is configured.
        """
        weights = checkpoint or self.checkpoint
        if weights is None:
            raise ValueError(
                f"No model weights for experiment '{self.name}'. Set "
                "'model.checkpoint' in the config, or pass one explicitly."
            )

        return ModelConfig(
            model_path=weights,
            patch_size=int(self.inference.get("patch_size", self.patch_size)),
            target_mpp=self.target_mpp,
            input_channels=int(self.data.get("in_channels", 3)),
            batch_size=int(self.inference.get("batch_size", 4)),
            threshold=float(self.inference.get("threshold", 0.5)),
            # Read from the inference block, not data.num_classes: a training
            # config counts annotation classes, while the network the trainer
            # builds emits a single foreground channel.
            num_classes=int(self.inference.get("num_classes", 1)),
            device=self.inference.get("device", self.device),
            backend=self.inference.get("backend", "auto"),
            architecture=self.model.get("name"),
            encoder=self.model.get("encoder_name"),
            stain_normalise=bool(self.inference.get("stain_normalise", False)),
        )

    def output_config(self, output_dir: Path | None = None) -> OutputConfig:
        """Build the :class:`~histocoreml.config.OutputConfig` for predictions.

        Args:
            output_dir: Where masks are written. Defaults to
                ``inference.output_dir``, else the experiment's ``output_dir``.
        """
        configured = self.inference.get("output_dir")
        return OutputConfig(
            output_dir=output_dir or (Path(configured) if configured else self.output_dir),
            output_format=self.inference.get("output_format", "tiff"),
            downsample_factor=self.inference.get("downsample_factor"),
            compression=self.inference.get("compression", "lzw"),
            save_thumbnail=bool(self.inference.get("save_thumbnail", True)),
            rle_subformat=self.inference.get("rle_subformat", "plain"),
            save_overlay=bool(self.inference.get("save_overlay", True)),
            overlay_alpha=float(self.inference.get("overlay_alpha", 0.40)),
            overlay_max_edge=int(self.inference.get("overlay_max_edge", 2048)),
        )

    def segmentation_config(self, checkpoint: Path | None = None) -> SegmentationPipelineConfig:
        """Assemble the full inference config this experiment describes.

        This is what lets one nested document drive ``histo-segment`` as well as
        training, so the repo needs only a single config schema.

        Args:
            checkpoint: Weights override; defaults to ``model.checkpoint``.
        """
        return SegmentationPipelineConfig(
            model=self.inference_model_config(checkpoint),
            tiling=self.inference_tiling_config(),
            output=self.output_config(),
            log_level=self.log_level,
        )

    def loss_spec(self) -> dict[str, Any]:
        """Return ``{'name': ..., **hyperparameters}`` for :func:`get_loss`.

        Everything in the ``training.loss`` block other than ``name`` is passed
        through, so weights declared in YAML reach the loss constructor::

            criterion = get_loss(**cfg.loss_spec())
        """
        loss = dict(self.training.get("loss", {}))
        name = loss.pop("name", "dice_bce")
        return {"name": name, **loss}

    def log_file(self) -> Path:
        """Path of the run's log file."""
        return self.output_dir / "pipeline.log"
