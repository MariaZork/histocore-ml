"""HistoCoreML configuration dataclasses.

All configs are frozen dataclasses and can be loaded from YAML::

    from histocoreml.config import PipelineConfig
    cfg = PipelineConfig.from_yaml("configs/default.yaml")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import yaml


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", Path(self.model_path))


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

    downsample_factor: Optional[int] = None
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

    model_path: Optional[Path] = None
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

    tasks: List[str] = field(default_factory=lambda: ["cell_density"])
    """List of biomarker tasks: 'cell_density' | 'nuclei_morphology' |
    'spatial_graph' | 'tumor_stroma_ratio' | 'ki67_index'."""

    cell_model_path: Optional[Path] = None
    """Path to cell detection model weights."""

    nuclei_model_path: Optional[Path] = None
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


# ── Top-level ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline configuration aggregating all sub-configs."""

    model:  ModelConfig
    tiling: TilingConfig
    output: OutputConfig

    log_level: str = "INFO"
    """Python logging level: ``DEBUG`` | ``INFO`` | ``WARNING`` | ``ERROR``."""

    @classmethod
    def from_yaml(cls, path: Path | str) -> "PipelineConfig":
        """Load a :class:`PipelineConfig` from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Fully validated :class:`PipelineConfig`.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open() as fh:
            raw: dict = yaml.safe_load(fh) or {}

        return cls(
            model=ModelConfig(**raw["model"]),
            tiling=TilingConfig(**raw.get("tiling", {})),
            output=OutputConfig(**raw.get("output", {})),
            log_level=raw.get("log_level", "INFO"),
        )


__all__ = [
    "ModelConfig",
    "TilingConfig",
    "OutputConfig",
    "FoundationConfig",
    "TrainingConfig",
    "BiomarkerConfig",
    "PipelineConfig",
]
