"""Factory and pipeline for foundation model feature extraction."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from histocoreml.config import FoundationConfig, ModelConfig, TilingConfig
from histocoreml.foundation.base_encoder import BaseEncoder
from histocoreml.foundation.vit_encoder import UNIEncoder, ViTEncoder
from histocoreml.io.factory import get_reader
from histocoreml.preprocessing.grid_generator import generate_patch_coords
from histocoreml.preprocessing.patch_dataset import build_dataloader

logger = logging.getLogger(__name__)


# ── Factory ───────────────────────────────────────────────────────────────────

def get_encoder(cfg: FoundationConfig) -> BaseEncoder:
    """Return an encoder instance for the specified foundation model.

    Args:
        cfg: Foundation model configuration.

    Returns:
        An uninitialised :class:`BaseEncoder` subclass.

    Raises:
        ValueError: If the model_name is not recognised.
    """
    name = cfg.model_name.lower()

    if name == "uni":
        return UNIEncoder(cfg)

    if name in ("vit", "custom"):
        timm_name = "vit_large_patch16_224"
        return ViTEncoder(cfg, timm_name=timm_name)

    if name == "ctranspath":
        return ViTEncoder(cfg, timm_name="swin_tiny_patch4_window7_224")

    raise ValueError(
        f"Unknown foundation model '{name}'. "
        "Supported: uni, vit, custom, ctranspath. "
        "For CONCH/PLIP, load weights as ViTEncoder with model_path."
    )


# ── Embedding pipeline ────────────────────────────────────────────────────────

@dataclass
class EmbeddingResult:
    """Output of embedding extraction for a single WSI."""

    wsi_path: Path
    embeddings: np.ndarray          # (N_patches, embedding_dim)
    coords: list                    # List[PatchCoord]
    elapsed_seconds: float
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def save(self, output_dir: Path) -> Path:
        """Save embeddings and coords to a .npz file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / f"{self.wsi_path.stem}_embeddings.npz"
        xs = np.array([c.x for c in self.coords])
        ys = np.array([c.y for c in self.coords])
        np.savez_compressed(
            str(out),
            embeddings=self.embeddings,
            coord_x=xs,
            coord_y=ys,
        )
        logger.info("Embeddings saved → %s  (%d patches)", out, len(self.coords))
        return out


class EmbeddingPipeline:
    """WSI-level feature extraction pipeline using a foundation model encoder.

    Tiles the slide at the encoder's target MPP, filters tissue patches, runs
    the encoder in batches, and saves per-patch embeddings as .npz files.

    Usage::

        cfg      = FoundationConfig(model_name="uni", target_mpp=0.5, batch_size=64)
        encoder  = get_encoder(cfg)
        pipeline = EmbeddingPipeline(cfg, encoder)
        results  = pipeline.run([Path("slide.svs")], output_dir=Path("embeddings"))
    """

    def __init__(self, cfg: FoundationConfig, encoder: BaseEncoder) -> None:
        self._cfg = cfg
        self._encoder = encoder

    def run(
        self,
        wsi_paths: list[Path],
        output_dir: Path | None = None,
        save: bool = True,
    ) -> list[EmbeddingResult]:
        """Extract embeddings for a list of WSI files.

        Args:
            wsi_paths:  List of paths to WSI files.
            output_dir: Directory for saving .npz embedding files.
            save:       If True, save embeddings to disk.

        Returns:
            List of :class:`EmbeddingResult` objects.
        """
        results: list[EmbeddingResult] = []
        with self._encoder as enc:
            for path in wsi_paths:
                try:
                    result = self._process_slide(path, enc)
                    if save and output_dir:
                        result.save(output_dir)
                    results.append(result)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Error embedding %s: %s", path, exc, exc_info=True)
                    results.append(EmbeddingResult(
                        wsi_path=path,
                        embeddings=np.empty((0, self._cfg.embedding_dim)),
                        coords=[],
                        elapsed_seconds=0.0,
                        errors=[str(exc)],
                    ))
        return results

    def _process_slide(self, path: Path, enc: BaseEncoder) -> EmbeddingResult:
        t0 = time.perf_counter()
        logger.info("Embedding: %s", path.name)

        # Adapt FoundationConfig → ModelConfig / TilingConfig shapes
        model_cfg = ModelConfig(
            model_path=path,           # placeholder, not used for I/O
            patch_size=self._cfg.patch_size,
            target_mpp=self._cfg.target_mpp,
            batch_size=self._cfg.batch_size,
        )
        tiling_cfg = TilingConfig(overlap=0, tissue_threshold=0.05)

        with get_reader(path) as reader:
            metadata = reader.get_metadata()
            coords = generate_patch_coords(metadata, model_cfg, tiling_cfg, slide_id=path.stem)

        loader = build_dataloader(
            slide_path=path,
            coords=coords,
            tiling_cfg=tiling_cfg,
            model_patch_size=self._cfg.patch_size,
            batch_size=self._cfg.batch_size,
        )

        all_emb: list[np.ndarray] = []
        all_coords = []

        for batch in loader:
            if batch is None:
                continue
            emb = enc.encode_batch_normalised(batch["images"])
            all_emb.append(emb)
            all_coords.extend(batch["coords"])

        embeddings = np.concatenate(all_emb, axis=0) if all_emb else \
            np.empty((0, self._cfg.embedding_dim), dtype=np.float32)

        elapsed = time.perf_counter() - t0
        logger.info("Done: %s | %d patches embedded in %.1fs", path.name, len(all_coords), elapsed)

        return EmbeddingResult(
            wsi_path=path,
            embeddings=embeddings,
            coords=all_coords,
            elapsed_seconds=elapsed,
        )
