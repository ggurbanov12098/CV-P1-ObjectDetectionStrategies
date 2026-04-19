# CV-P1-ObjectDetectionStrategies

This project compares two license-plate detection strategies on CCPD data:

1. A classical computer-vision baseline (OpenCV contours + geometry filters)
2. A YOLO detector trained with Ultralytics

The repository is script-driven and focuses on an end-to-end workflow:

1. Prepare CCPD images/labels in YOLO format
2. Train YOLO on Apple Metal Performance Shaders (MPS)
3. Run classical and YOLO detectors side-by-side on video

## Why This README Calls Out Missing Folders on GitHub

Some important folders are intentionally excluded by [.gitignore](.gitignore):

- `**/data`
- `**/datasets`
- `**/media`
- `**/.venv`
- `**/__pycache__`

This means a GitHub clone may not include raw data, prepared YOLO datasets, or media files, even though those folders exist locally during development.

## Repository Layout

The project currently uses this structure (tracked files plus local generated content):

```text
CV-P1-ObjectDetectionStrategies/
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── yolov8n.pt
├── scripts/
│   ├── prepare_ccpd.py
│   ├── train_yolo.py
│   ├── process_video.py
│   └── classical_detector.py
├── runs/
│   └── detect/runs/train/ccpd_yolo_mps/
│       ├── args.yaml
│       ├── results.csv
│       ├── weights/{best.pt,last.pt}
│       └── training plots/images
├── data/              # ignored; local-only raw CCPD files
├── datasets/          # ignored; local-only YOLO dataset + yaml
└── media/             # ignored; local-only videos
```

### Tracked vs local-only directories

| Path | Purpose | Typical GitHub visibility |
| --- | --- | --- |
| `scripts/` | Main executable pipeline scripts | Tracked |
| `pyproject.toml`, `uv.lock` | Dependencies and lock state | Tracked |
| `runs/` | Training outputs and model artifacts | Tracked if committed |
| `data/` | Raw CCPD source dataset | Ignored by default |
| `datasets/` | Prepared YOLO dataset + `ccpd.yaml` | Ignored by default |
| `media/` | Input/output video assets | Ignored by default |

## Requirements

- Python `>=3.13` (see [.python-version](.python-version) and [pyproject.toml](pyproject.toml))
- macOS + Apple Silicon recommended for YOLO scripts, because training/inference is configured for MPS only
- OpenCV and Ultralytics dependencies (declared in [pyproject.toml](pyproject.toml))

## Environment Setup

### Option A: uv (recommended because `uv.lock` exists)

```bash
uv sync
source .venv/bin/activate
```

### Option B: venv + pip

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Data Preparation (CCPD -> YOLO)

Script: [scripts/prepare_ccpd.py](scripts/prepare_ccpd.py)

This script:

1. Recursively finds images in CCPD directories
2. Parses bounding boxes from CCPD filename metadata (`x1&y1_x2&y2` segment)
3. Converts to YOLO normalized format (`class x_center y_center width height`)
4. Builds train/val splits from a random sample
5. Copies images and writes `.txt` labels

### Expected raw input layout

```text
data/CCPD2020/ccpd_green/
├── train/
├── val/
└── test/
```

### Run preparation

```bash
python scripts/prepare_ccpd.py \
    --source-dir data/CCPD2020/ccpd_green \
    --output-dir datasets/ccpd \
    --sample-size 5000 \
    --train-ratio 0.8 \
    --seed 42
```

### Output layout

```text
datasets/ccpd/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

For the current local workspace snapshot with default sampling:

- `images/train`: 4000 files
- `images/val`: 1000 files
- `labels/train`: 4000 files
- `labels/val`: 1000 files

## Dataset YAML

Training expects [datasets/ccpd.yaml](datasets/ccpd.yaml). Because `datasets/` is ignored, this file may be missing on GitHub and must be created locally.

Use:

```yaml
path: datasets/ccpd
train: images/train
val: images/val

names:
    0: license_plate
```

## YOLO Training

Script: [scripts/train_yolo.py](scripts/train_yolo.py)

### Important design choice

Training is MPS-only by design:

- The script checks that PyTorch was built with MPS support
- The script checks MPS is available at runtime
- CPU fallback is explicitly disabled (`PYTORCH_ENABLE_MPS_FALLBACK=0`)

If MPS is unavailable, the script exits with a clear error.

### Train command

```bash
python scripts/train_yolo.py \
    --data datasets/ccpd.yaml \
    --model yolov8n.pt \
    --epochs 20 \
    --batch 32 \
    --imgsz 640 \
    --workers 8 \
    --project runs/train \
    --name ccpd_yolo_mps
```

### Training outputs

Typical output files include:

- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- PR/F1/confusion matrix plots

An example run in this workspace is stored at:

- [runs/detect/runs/train/ccpd_yolo_mps](runs/detect/runs/train/ccpd_yolo_mps)

From the local `results.csv` snapshot (20 epochs):

- `mAP50(B)`: `0.995`
- `mAP50-95(B)`: `0.89262`

## Classical Baseline (Single Image)

Script: [scripts/classical_detector.py](scripts/classical_detector.py)

Pipeline summary:

1. Grayscale
2. Bilateral filtering
3. Canny edges
4. Contour extraction
5. Quadrilateral + convexity filtering
6. Aspect ratio filtering (`2.0` to `5.0`)
7. Bounding-box visualization

Run:

```bash
python scripts/classical_detector.py \
    --input path/to/image.jpg \
    --output outputs/classical_detection.jpg
```

## Side-by-Side Video Comparison

Script: [scripts/process_video.py](scripts/process_video.py)

This script creates a 2-panel output video:

- Left: Classical detector
- Right: YOLO detector

Run:

```bash
python scripts/process_video.py \
    --input media/chongqing_4k.mp4 \
    --weights runs/detect/runs/train/ccpd_yolo_mps/weights/best.pt \
    --output media/comparison_output.mp4 \
    --max-frames 600 \
    --target-width 1280 \
    --target-height 720 \
    --conf 0.25
```

Note: the script defaults to MPS (`device='mps'`) for YOLO inference.

## Reproducibility Notes

- Data sampling is deterministic with `--seed`
- Train/val split is controlled by `--train-ratio`
- Bounding boxes are clamped to image bounds before YOLO conversion
- Duplicate filenames are avoided by prefixing split name and auto-numbering collisions

## Common Issues and Fixes

### 1) "MPS is not available" or "PyTorch build does not include MPS"

Install a PyTorch build with MPS support for macOS and verify from Python:

```python
import torch
print(torch.backends.mps.is_built())
print(torch.backends.mps.is_available())
```

### 2) "Dataset yaml not found"

Ensure [datasets/ccpd.yaml](datasets/ccpd.yaml) exists locally. It is inside an ignored folder.

### 3) Missing raw dataset or media files after cloning

Populate local `data/`, `datasets/`, and `media/` directories manually. They are intentionally ignored and may not be present on GitHub.

### 4) Weights path mismatch

If your run directory differs from defaults, pass `--weights` explicitly to [scripts/process_video.py](scripts/process_video.py).

## Quick Start (End-to-End)

```bash
# 1) create environment
uv sync && source .venv/bin/activate

# 2) prepare data
python scripts/prepare_ccpd.py --source-dir data/CCPD2020/ccpd_green --output-dir datasets/ccpd

# 3) make sure datasets/ccpd.yaml exists

# 4) train yolo
python scripts/train_yolo.py --data datasets/ccpd.yaml --model yolov8n.pt

# 5) run side-by-side comparison
python scripts/process_video.py --input media/chongqing_4k.mp4 --weights runs/detect/runs/train/ccpd_yolo_mps/weights/best.pt --output media/comparison_output.mp4
```

## Scope and Status

Current implementation is focused on single-class bounding-box detection (`license_plate`) and reproducible experimentation between classical CV and YOLO-based learning.

Potential next extensions:

- Add CPU/CUDA fallback paths for non-Apple machines
- Add automated evaluation script for classical vs YOLO metrics on a shared validation set
- Add experiment tracking and model versioning
