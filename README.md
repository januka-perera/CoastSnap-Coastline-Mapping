# CoastSnap Coastline Mapping

Automated beach segmentation and shoreline extraction from CoastSnap imagery using SAM2 (Segment Anything Model 2).

## Overview

**Phase 1:** Beach segmentation using SAM2 with positive/negative point prompts from a reference image.
**Phase 2:** Shoreline extraction from the segmentation mask boundary.

## Requirements

- NVIDIA GPU with CUDA 12.1+
- [Mamba](https://github.com/conda-forge/miniforge) (recommended) or Conda

Check your maximum supported CUDA version before setup:
```bash
nvidia-smi
```

## Setup

### 1. Install Mamba (once, into base environment)

```bash
conda install -n base -c conda-forge mamba
```

### 2. Create the environment

```bash
mamba env create -f environment.yml
```

### 3. Activate the environment

```bash
mamba activate coastsnap
```

### 4. Install SAM2 and download model checkpoints

SAM2 must be installed as an editable install from a local clone. The clone **must not** be named `sam2` — it would shadow the installed Python package. Clone it as `sam2-source`:

```bash
git clone https://github.com/facebookresearch/sam2.git sam2-source
pip install -e sam2-source
```

> **Why editable?** Installing SAM2 via `pip install git+...` does not bundle the Hydra config YAML files needed at runtime. An editable install from a local clone makes them available in-place.

Then download the model checkpoints:

```bash
cd sam2-source/checkpoints
bash download_ckpts.sh
cd ../..
```

Move the downloaded `.pt` files into `models/checkpoints/`. The recommended checkpoint for this project is `sam2.1_hiera_base_plus.pt`.

## Project Structure

```
CoastSnap-Coastline-Mapping/
├── sam2-source/                    # SAM2 repo clone (gitignored)
├── data/
│   ├── raw/                        # Original, unmodified images (never edited)
│   │   └── <site_name>/
│   ├── processed/                  # Resized/normalized images if needed
│   └── reference/                  # Reference images + point annotations
│       └── <site_name>/
│           ├── reference.jpg
│           └── annotations.json
├── outputs/
│   ├── masks/                      # Binary segmentation masks
│   ├── visualizations/             # Mask overlays for QC
│   └── shorelines/                 # Phase 2: extracted shoreline coordinates
├── models/
│   └── checkpoints/                # SAM2 .pt weight files (gitignored)
├── src/
│   ├── segmentation/
│   │   ├── predictor.py            # SAM2 wrapper
│   │   └── postprocess.py          # Mask cleanup
│   ├── shoreline/                  # Phase 2
│   │   └── extractor.py
│   └── utils/
│       ├── io.py                   # Image and annotation I/O
│       └── visualization.py        # QC overlays
├── notebooks/
│   ├── 01_sam2_exploration.ipynb
│   └── 02_beach_segmentation.ipynb
├── configs/
│   └── config.yaml
├── environment.yml
├── requirements.txt
└── README.md
```

## Annotation Format

Point annotations are stored as JSON alongside the reference image:

```json
{
  "image": "reference.jpg",
  "positive_points": [[x1, y1], [x2, y2]],
  "negative_points": [[x3, y3], [x4, y4]],
  "notes": "pos = dry sand, neg = waterline and sky"
}
```

## Updating the Environment

If `environment.yml` changes:

```bash
mamba env update -f environment.yml --prune
```
