# EchoNet-Dynamic Demo

Echocardiogram video analysis using deep learning. Trains a **ResNet18** for ejection fraction (EF) regression and a **U-Net** for left ventricle (LV) segmentation on a subset of the [EchoNet-Dynamic](https://echonet.github.io/dynamic/) dataset.

## What it does

1. **Extract frames** from echo videos (from EchoNet-Dynamic.zip)
2. **Build segmentation masks** from volume tracings
3. **Train models**:
   - `ResNet18` — predicts ejection fraction from video frames
   - `U-Net` — segments the left ventricle in individual frames
4. **Streamlit UI** — upload an echo video and see EF prediction + LV segmentation with contour overlay

## Getting started

### Prerequisites

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### 1. Download the dataset

Place `EchoNet-Dynamic.zip` in the project root. You can get it from the [EchoNet-Dynamic website](https://echonet.github.io/dynamic/).

### 2. Install dependencies

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

### 3. Run the pipeline

```bash
uv run python run_pipeline.py
```

This will:
- Extract frames from the first 12 videos in the zip
- Build masks from tracings
- Train ResNet18 (EF regression) and U-Net (LV segmentation)
- Save models to `models_saved/`

### 4. Launch the UI

```bash
uv run streamlit run ui/app.py
```

Upload an echocardiogram video (AVI, MP4) to see EF predictions and LV segmentation overlays.

## Docker (GPU)

```bash
docker compose up --build
```

Requires NVIDIA Container Toolkit. The UI runs on port 8501.

## Project structure

```
├── run_pipeline.py          # End-to-end pipeline
├── pyproject.toml           # Dependencies
├── Dockerfile / docker-compose.yml
├── src/
│   ├── config.py            # Paths & hyperparameters
│   ├── preprocessing/       # Frame extraction, masks, dataset
│   ├── models/              # ResNet18 (regression), U-Net (segmentation)
│   ├── training/            # Training loops, early stopping
│   ├── postprocessing/      # LV contour extraction
│   └── visualization/       # Mask overlay, comparison grids
├── ui/
│   └── app.py               # Streamlit app
├── models_saved/            # Trained .pth weights (gitignored)
└── data/
    └── processed/           # Extracted frames, masks (gitignored)
```

## Notes

- Trained on a small subset (12 videos) for demo purposes only — not for clinical use.
- Re-running the pipeline automatically cleans and regenerates all outputs.
