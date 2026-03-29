# HistoCoreML

**Production-ready ML framework for computational histology.**

HistoCoreML covers the full spectrum of computational pathology tasks in a single, coherent package:

| Module | What it does |
|--------|-------------|
| `histocoreml.io` | OpenSlide + TiffFile WSI readers, MPP metadata, multi-backend factory |
| `histocoreml.preprocessing` | Patch tiling, MPP enforcement, tissue filtering, Macenko H&E normalisation |
| `histocoreml.inference` | TorchScript & ONNX segmentation backends |
| `histocoreml.postprocessing` | Memory-mapped mask assembly, overlap averaging |
| `histocoreml.output` | TIFF, NumPy, RLE (plain & COCO), Zarr, GeoJSON writers |
| `histocoreml.foundation` | UNI, ViT, CTransPath encoders + WSI embedding pipeline |
| `histocoreml.training` | Dice/Focal/Tversky losses, HistoSegDataset, SegmentationTrainer |
| `histocoreml.biomarkers` | Cell density, nuclei morphology, spatial graphs, Ki-67, H&E/DAB stain separation |

---

## Architecture Overview

```
WSI file (SVS / TIFF / MRXS / NDPI)
        │
        ▼
┌──────────────────────┐
│   BaseWSIReader      │  OpenSlide or TiffFile backed; produces uint8 RGB patches
│   (+ TifffileReader) │  read_region_at_mpp() for physical coordinate queries
└──────────┬───────────┘
           │  read_region(x, y, level, size)
           ▼
┌──────────────────────┐
│  generate_patch_     │  Row-major grid of PatchCoord objects.
│  coords()            │  Selects best pyramid level, stores rescale_factor.
└──────────┬───────────┘
           │  List[PatchCoord]
           ▼
┌──────────────────────────────────────────────────────┐
│  SegmentationPipeline                                │
│    PatchDataset (torch map-style) → DataLoader       │
│    num_workers producers (read/rescale/filter/norm)  │
│    Main process: infer → MaskAssembler               │
└──────────┬───────────────────────────────────────────┘
           │  (N, H, W) uint8 binary masks
           ▼
┌──────────────────────┐
│  MaskAssembler       │  MemmapCanvas: never loads full slide into RAM.
│  (MemmapCanvas)      │  Overlapping patches averaged then binarised.
└──────────┬───────────┘
           │  (H, W) uint8 binary mask
           ▼
┌──────────────────────┐
│  Writer              │  TIFF · NumPy · RLE JSON (plain/COCO) · Zarr · GeoJSON
└──────────────────────┘
           │  (optional parallel paths)
           ├──────────────────────────────────────────────────┐
           ▼                                                  ▼
┌──────────────────────┐                        ┌────────────────────────┐
│  OverlayWriter       │                        │  BiomarkerExtractor    │
│  WSI thumbnail +     │                        │  Cell density · Morph  │
│  mask blend PNG      │                        │  Spatial graph · Ki-67 │
└──────────────────────┘                        └────────────────────────┘

Foundation Model Pipeline (parallel path):
  WSI → patch tiling → EmbeddingPipeline → .npz (N_patches × D)
```

---

## Setup

```bash
conda create -n histocoreml python=3.11
conda activate histocoreml
conda install -c conda-forge openslide

# Core (segmentation only)
pip install -e ".[openslide,dev]"

# Everything
pip install -e ".[all,dev]"
```

---

## Usage

### Segmentation

```bash
# Basic run
histo-segment -c configs/default.yaml -i data/slide.svs --save-overlay

# GPU with RLE output
histo-segment -c configs/gpu.yaml -i data/*.svs --output-format rle --device cuda:0

# GeoJSON contours (compatible with QuPath / ASAP)
histo-segment -c configs/geojson.yaml -i data/slide.svs
```

### Foundation Model Embeddings

```bash
histo-embed --model uni --model-path weights/uni.pth \
            -i data/*.svs -o embeddings/ --device cuda:0
```

### Biomarker Extraction

```bash
histo-extract -i data/slide.svs \
              --mask outputs/slide_mask.npy \
              --tasks cell_density nuclei_morphology spatial_graph ki67_index
```

### Training

```bash
histo-train --images data/images --masks data/masks \
            --val-images data/val_images --val-masks data/val_masks \
            --arch unet --encoder resnet50 --loss dice_bce \
            --epochs 100 --lr 1e-4
```

---

## Python API

### Segmentation pipeline

```python
from histocoreml.config import PipelineConfig
from histocoreml.pipeline import SegmentationPipeline
from pathlib import Path

cfg      = PipelineConfig.from_yaml("configs/default.yaml")
pipeline = SegmentationPipeline(cfg)
results  = pipeline.run([Path("slide.svs")])

for r in results:
    if r.success:
        print(f"Mask: {r.write_result.path}  ({r.elapsed_seconds:.1f}s)")
```

### Foundation model embeddings

```python
from histocoreml.config import FoundationConfig
from histocoreml.foundation import get_encoder, EmbeddingPipeline
from pathlib import Path

cfg      = FoundationConfig(model_name="uni", model_path=Path("uni.pth"),
                             embedding_dim=1024, target_mpp=0.5, device="cuda")
encoder  = get_encoder(cfg)
pipeline = EmbeddingPipeline(cfg, encoder)
results  = pipeline.run([Path("slide.svs")], output_dir=Path("embeddings"))
# → embeddings/slide_embeddings.npz  (N_patches × 1024)
```

### Biomarker extraction

```python
import numpy as np
from histocoreml.config import BiomarkerConfig
from histocoreml.biomarkers import BiomarkerExtractor
from pathlib import Path

mask      = np.load("outputs/slide_mask.npy")
cfg       = BiomarkerConfig(tasks=["cell_density", "nuclei_morphology",
                                    "spatial_graph", "tumor_stroma_ratio"])
extractor = BiomarkerExtractor(cfg)
report    = extractor.run(Path("slide.svs"), mask=mask)
report.save(Path("biomarkers/slide.json"))
print(report.features)
```

### Training

```python
from histocoreml.config import TrainingConfig
from histocoreml.training import SegmentationTrainer, build_train_dataloader
from pathlib import Path

cfg     = TrainingConfig(architecture="unet", encoder="resnet50",
                          loss="dice_bce", epochs=100)
train_loader = build_train_dataloader(Path("data/images"), Path("data/masks"))
val_loader   = build_train_dataloader(Path("data/val_images"), Path("data/val_masks"),
                                       shuffle=False)
trainer = SegmentationTrainer(cfg)
history = trainer.fit(train_loader, val_loader)
```

### Stain normalisation (standalone)

```python
from histocoreml.preprocessing.patch_utils import macenko_normalise
import numpy as np
from PIL import Image

patch     = np.array(Image.open("patch.png").convert("RGB"))
normalised = macenko_normalise(patch)
```

---

## Output Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| TIFF | `_mask.tiff` | Tiled GeoTIFF with MPP resolution tag — compatible with QuPath, ImageJ |
| NumPy | `_mask.npy` | Fast; ideal for downstream Python analysis |
| Plain RLE | `_mask.json` | Row-major (value, count) pairs; ~20× compression |
| COCO RLE | `_mask.json` | Column-major; compatible with pycocotools |
| Zarr | `_mask.zarr` | Chunked array; supports S3/GCS cloud backends |
| GeoJSON | `_mask.geojson` | Polygon contours in physical (µm) coordinates |

---

## Running Tests

```bash
# All tests (no GPU or OpenSlide required)
pytest

# Verbose
pytest -v

# Skip slow tests
pytest -m "not slow"

# With coverage
pytest --cov=histocoreml --cov-report=term-missing
```

---
