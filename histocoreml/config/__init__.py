"""HistoCoreML configuration dataclasses.

All configs are frozen dataclasses and can be loaded from YAML::

    from histocoreml.config import PipelineConfig
    cfg = SegmentationPipelineConfig.from_yaml("configs/default.yaml")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

import yaml

# ── Base Pipeline Config ─────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """Base configuration for all pipelines."""

    log_level: str = "INFO"
    output_dir: Path = Path("outputs")
    num_workers: int = 4
    device: str = "auto"

    def __post_init__(self) -> None:
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


# ── Pipeline Results ─────────────────────────────────────────────────────────


@dataclass
class PipelineResult:
    """Base result for all pipeline operations."""

    wsi_path: Path | None = None
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


@dataclass
class InferenceResult(PipelineResult):
    """Base result for inference pipelines."""

    patch_count: int = 0
    model_name: str = ""


@dataclass
class SegmentationInferenceResult(InferenceResult):
    """Result for segmentation inference."""

    write_result: Any | None = None  # WriteResult from output module
    mask_path: Path | None = None


@dataclass
class EmbeddingInferenceResult(InferenceResult):
    """Result for embedding/feature extraction inference."""

    embeddings: Any = None  # np.ndarray (N_patches, embedding_dim)
    coords: list = field(default_factory=list)

    def save(self, output_dir: Path, stem: str | None = None) -> Path:
        """Save embeddings to a .npz file."""
        import numpy as np  # noqa: PLC0415

        output_dir.mkdir(parents=True, exist_ok=True)

        if stem is None and self.wsi_path:
            stem = self.wsi_path.stem
        elif stem is None:
            stem = "embeddings"

        out = output_dir / f"{stem}_embeddings.npz"

        if self.embeddings is not None and len(self.coords) > 0:
            xs = np.array([c.x for c in self.coords])
            ys = np.array([c.y for c in self.coords])
            np.savez_compressed(
                str(out),
                embeddings=self.embeddings,
                coord_x=xs,
                coord_y=ys,
            )
        else:
            np.savez_compressed(
                str(out),
                embeddings=np.empty((0, 0), dtype=np.float32),
                coord_x=np.array([]),
                coord_y=np.array([]),
            )

        return out


@dataclass
class TrainingResult(PipelineResult):
    """Base result for training pipelines."""

    epochs_trained: int = 0
    best_metric: float = 0.0
    final_loss: float = 0.0
    history: dict[str, list] = field(default_factory=dict)
    checkpoint_path: Path | None = None


@runtime_checkable
class PipelineConfigLike(Protocol):
    """The whole contract :class:`BasePipeline` needs from a configuration.

    Bound structurally rather than to :class:`PipelineConfig` so the frozen
    aggregate configs (:class:`SegmentationPipelineConfig`) qualify too — they
    cannot inherit from it, because a frozen dataclass may not subclass a
    non-frozen one.
    """

    @property
    def log_level(self) -> str:  # pragma: no cover - structural declaration
        ...


# Type variables for generic pipeline base classes
C = TypeVar("C", bound=PipelineConfigLike)
R = TypeVar("R", bound=PipelineResult)


# ── Model / Inference ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the segmentation model."""

    model_path: Path
    """Path to the TorchScript (.pt) model file."""

    patch_size: int = 512
    """Spatial size (H=W) of each input patch in pixels."""

    target_mpp: float = 0.88
    """Target microns-per-pixel resolution for inference (10× magnification)."""

    input_channels: int = 3
    """Number of input channels expected by the model (RGB = 3)."""

    device: str = "cpu"
    """Torch device string: 'cpu', 'mps', 'cuda', or 'cuda:N'."""

    batch_size: int = 8
    """Number of patches processed per forward pass."""

    threshold: float = 0.5
    """Sigmoid threshold to binarise logit predictions."""

    num_classes: int = 1
    """Number of output classes (1 = binary segmentation)."""

    backend: str = "auto"
    """Inference backend: 'auto' | 'torchscript' | 'onnx' | 'checkpoint'.

    ``auto`` picks from the file suffix: ``.pt``/``.ts`` → TorchScript,
    ``.onnx`` → ONNX Runtime, ``.pth``/``.ckpt`` → a training checkpoint
    reloaded into ``architecture``/``encoder``.
    """

    architecture: str | None = None
    """Architecture to rebuild for the 'checkpoint' backend, e.g. 'unet++'.

    Ignored by TorchScript and ONNX, which carry their own graph. When None,
    the value stored in the checkpoint's own config is used.
    """

    encoder: str | None = None
    """Encoder backbone to rebuild for the 'checkpoint' backend, e.g. 'resnet50'."""

    stain_normalise: bool = False
    """Apply Macenko H&E normalisation to each patch before inference.

    A property of the *model*, not of the tiling: whether patches must be
    stain-normalised is fixed by how the model was trained. Tiling settings
    (overlap, tissue thresholds, workers) can change freely without affecting it.
    """

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", Path(self.model_path))

        if self.device == "auto":
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                resolved = "cuda"
            elif torch.backends.mps.is_available():
                resolved = "mps"
            else:
                resolved = "cpu"
            object.__setattr__(self, "device", resolved)


@dataclass(frozen=True)
class TilingConfig:
    """Configuration for patch extraction / tiling strategy."""

    overlap: int = 64
    """Overlap in pixels between adjacent patches."""

    tissue_threshold: float = 0.05
    """Minimum foreground tissue fraction to keep a patch."""

    background_value: int = 230
    """Pixel intensity above which a pixel is considered background (white glass)."""

    black_value: int = 10
    """Pixel intensity below which a pixel is considered black scanner background."""

    num_workers: int = 4
    """Number of DataLoader worker processes for parallel patch prefetching."""

    prefetch_factor: int = 2
    """Number of batches prefetched per DataLoader worker."""


# ── Output ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OutputConfig:
    """Configuration for mask output."""

    output_dir: Path = Path("outputs")
    """Directory where result masks are written."""

    output_format: str = "tiff"
    """Output format: 'tiff' | 'npy' | 'rle' | 'geojson' | 'zarr'."""

    downsample_factor: int | None = None
    """If set, the output mask is written at 1/N of the WSI level-0 resolution."""

    compression: str = "lzw"
    """TIFF compression: 'lzw' | 'jpeg' | 'deflate' | 'none'."""

    save_thumbnail: bool = True
    """If True, also save a small PNG overlay for quick visual QC."""

    rle_subformat: str = "plain"
    """RLE sub-format when output_format='rle': 'plain' | 'coco'."""

    save_overlay: bool = False
    """If True, write a PNG that shows the WSI thumbnail with predicted mask."""

    overlay_alpha: float = 0.40
    """Opacity of the red foreground overlay in [0, 1]."""

    overlay_max_edge: int = 2048
    """Maximum edge length (px) of the overlay thumbnail."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))


# ── Foundation Model ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FoundationConfig:
    """Configuration for foundation model feature extraction."""

    model_name: str = "uni"
    """Foundation model name: 'uni' | 'conch' | 'plip' | 'ctranspath' | 'custom'."""

    model_path: Path | None = None
    """Path to local model weights (required for 'custom')."""

    embedding_dim: int = 1024
    """Output embedding dimension."""

    patch_size: int = 224
    """Input patch size expected by the encoder."""

    target_mpp: float = 0.5
    """Target MPP for feature extraction (20×)."""

    device: str = "cpu"
    """Torch device for inference."""

    batch_size: int = 32
    """Patches per forward pass."""

    normalize_embeddings: bool = True
    """L2-normalize embeddings before downstream use."""

    def __post_init__(self) -> None:
        if self.model_path:
            object.__setattr__(self, "model_path", Path(self.model_path))


# ── Trainer State & Config ───────────────────────────────────────────────────


@dataclass(frozen=True)
class TrainingState:
    """Immutable training state snapshot.

    Attributes:
        epoch: Current epoch number (1-indexed)
        global_step: Total number of optimization steps
        best_metric: Best validation metric achieved so far
        patience_counter: Number of epochs without improvement
        history: Dictionary of training history
        is_training: Whether currently in training phase
    """

    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0
    patience_counter: int = 0
    history: dict[str, list[float]] = field(default_factory=dict)
    is_training: bool = False

    def replace(self, **changes: Any) -> TrainingState:
        """Create a new TrainingState with updated fields."""
        from dataclasses import replace  # noqa: PLC0415

        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "patience_counter": self.patience_counter,
            "history": self.history,
        }


@dataclass(frozen=True)
class TrainerConfig:
    """Base configuration for all trainers.

    Attributes:
        output_dir: Directory for checkpoints and logs
        checkpoint_dir: Where checkpoints go. Defaults to ``output_dir/checkpoints``.
        experiment_name: Name of the experiment
        seed: Random seed for reproducibility
        device: Device to use (auto, cuda, cpu, mps)
        mixed_precision: Whether to use automatic mixed precision
        log_level: Logging level
    """

    output_dir: Path = Path("./outputs")
    checkpoint_dir: Path | None = None
    experiment_name: str = "experiment"
    seed: int = 42
    device: str = "auto"
    mixed_precision: bool = True
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self,
            "checkpoint_dir",
            Path(self.checkpoint_dir) if self.checkpoint_dir else self.output_dir / "checkpoints",
        )

        # Auto-select device
        if self.device == "auto":
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                object.__setattr__(self, "device", "cuda")
            elif torch.backends.mps.is_available():
                object.__setattr__(self, "device", "mps")
            else:
                object.__setattr__(self, "device", "cpu")


# Type variable for trainer configs
T = TypeVar("T", bound=TrainerConfig)


# ── Training ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for model training."""

    task: str = "segmentation"
    """Task type: 'segmentation' | 'classification' | 'detection'."""

    architecture: str = "unet"
    """Model architecture: 'unet' | 'unet++' | 'deeplabv3+' | 'segformer'."""

    encoder: str = "resnet50"
    """Encoder backbone name (timm-compatible)."""

    pretrained: bool = True
    """Use ImageNet-pretrained encoder weights."""

    loss: str = "dice_bce"
    """Loss function: 'dice' | 'bce' | 'dice_bce' | 'focal' | 'tversky'."""

    optimizer: str = "adamw"
    """Optimizer: 'adam' | 'adamw' | 'sgd'."""

    learning_rate: float = 1e-4
    """Initial learning rate."""

    weight_decay: float = 1e-5
    """L2 regularisation weight."""

    epochs: int = 100
    """Maximum training epochs."""

    batch_size: int = 8
    """Training batch size."""

    patch_size: int = 512
    """Training patch size (H=W)."""

    target_mpp: float = 0.88
    """Target MPP for training patches."""

    num_workers: int = 4
    """DataLoader workers."""

    device: str = "auto"
    """Device: 'auto', 'cpu', 'cuda', 'mps'."""

    mixed_precision: bool = True
    """Use automatic mixed precision (AMP)."""

    early_stopping_patience: int = 15
    """Stop training if val metric doesn't improve for N epochs."""

    checkpoint_dir: Path = Path("checkpoints")
    """Directory for saving checkpoints."""

    experiment_name: str = "histocoreml_run"
    """MLflow / TensorBoard experiment name."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))


# ── Biomarker ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BiomarkerConfig:
    """Configuration for biomarker extraction."""

    tasks: list[str] = field(default_factory=lambda: ["cell_density"])
    """List of biomarker tasks: 'cell_density' | 'nuclei_morphology' |
    'spatial_graph' | 'tumor_stroma_ratio' | 'ki67_index'."""

    cell_model_path: Path | None = None
    """Path to cell detection model weights."""

    nuclei_model_path: Path | None = None
    """Path to nuclei segmentation model weights."""

    target_mpp: float = 0.25
    """MPP for biomarker extraction (40×)."""

    min_cell_area_px: int = 50
    """Minimum nucleus area in pixels to count as a cell."""

    max_cell_area_px: int = 5000
    """Maximum nucleus area in pixels."""

    output_dir: Path = Path("biomarkers")
    """Directory for biomarker output files."""

    export_geojson: bool = True
    """Export cell coordinates as GeoJSON."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.cell_model_path:
            object.__setattr__(self, "cell_model_path", Path(self.cell_model_path))
        if self.nuclei_model_path:
            object.__setattr__(self, "nuclei_model_path", Path(self.nuclei_model_path))


# ── Segmentation Pipeline Config ─────────────────────────────────────────────


@dataclass(frozen=True)
class SegmentationPipelineConfig:
    """Top-level pipeline configuration for segmentation inference.

    Aggregates model, tiling, and output sub-configs.
    """

    model: ModelConfig
    tiling: TilingConfig
    output: OutputConfig

    log_level: str = "INFO"
    """Python logging level: ``DEBUG`` | ``INFO`` | ``WARNING`` | ``ERROR``."""

    @classmethod
    def from_yaml(cls, path: Path | str) -> SegmentationPipelineConfig:
        """Load a :class:`SegmentationPipelineConfig` from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Fully validated :class:`SegmentationPipelineConfig`.

        Accepts both config schemas: a flat ``model``/``tiling``/``output``
        document, or a nested experiment document, which is converted through
        :meth:`~histocoreml.config.experiment.ExperimentConfig.segmentation_config`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If *path* has neither an ``experiment`` nor a ``model``
                section, or names no weights.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open() as fh:
            raw: dict = yaml.safe_load(fh) or {}

        # Two schemas live in configs/: the nested experiment documents and the
        # flat model/tiling/output form. Accept either so one config file can
        # drive both training and histo-segment.
        if "experiment" in raw:
            from histocoreml.config.experiment import ExperimentConfig  # noqa: PLC0415

            return ExperimentConfig.from_yaml(path).segmentation_config()

        if "model" not in raw:
            raise ValueError(f"{path} is missing the required 'model' section.")

        return cls(
            model=ModelConfig(**raw["model"]),
            tiling=TilingConfig(**raw.get("tiling", {})),
            output=OutputConfig(**raw.get("output", {})),
            log_level=raw.get("log_level", "INFO"),
        )


# Imported last: experiment.py depends on the dataclasses defined above.
from histocoreml.config.experiment import ExperimentConfig  # noqa: E402

__all__ = [
    # Base classes
    "PipelineConfig",
    "PipelineConfigLike",
    "C",
    "R",
    "T",
    # Results
    "PipelineResult",
    "InferenceResult",
    "SegmentationInferenceResult",
    "EmbeddingInferenceResult",
    "TrainingResult",
    # Trainer classes
    "TrainingState",
    "TrainerConfig",
    "T",
    # Configs
    "ModelConfig",
    "TilingConfig",
    "OutputConfig",
    "FoundationConfig",
    "TrainingConfig",
    "BiomarkerConfig",
    "SegmentationPipelineConfig",
    "ExperimentConfig",
]
