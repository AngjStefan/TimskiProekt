# EchoNet-Dynamic Demo

Echocardiogram video analysis using deep learning and AI. Trains **ResNet18 / ResNet34** for ejection fraction (EF) regression and a **U-Net** for left ventricle (LV) segmentation on a subset of the [EchoNet-Dynamic](https://echonet.github.io/dynamic/) dataset. Includes **Gemini 3 Flash / Pro** integration for AI-powered structural overlays and clinical assessments.

## What it does

1. **Extract frames** from echo videos (from EchoNet-Dynamic.zip)
2. **Build segmentation masks** from volume tracings
3. **Train models**:
   - `ResNet18 / ResNet34` — predicts ejection fraction from video frames
   - `U-Net` — segments the left ventricle in individual frames
4. **Streamlit UI** — upload an echo video and see:
   - EF prediction with HF clinical classification (HFrEF / HFmrEF / HFpEF)
   - LV segmentation with contour overlay and anatomical key points (apex, basal, mid)
   - AI-generated structural overlays and clinical opinion (Gemini 3 Flash / Pro)

## Getting started

### Prerequisites

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Google Gemini API key (for AI features)

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

### 3. Set up Gemini API key

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY="your-api-key-here"
```

### 4. Run the pipeline

```bash
uv run python run_pipeline.py
uv run python run_pipeline.py --subset 500
```

This will:
- Extract frames from the first 25 videos in the zip (configurable via `SUBSET_SIZE`)
- Build masks from tracings
- Train ResNet18 (EF regression) and U-Net (LV segmentation)
- Save models to `models_saved/`

### 5. Launch the UI

```bash
uv run streamlit run ui/app.py
```

Upload an echocardiogram video (AVI, MP4, MPEG) to see:
- EF prediction with clinical classification
- LV segmentation with contour overlays and key points
- AI-generated structural overlays and clinical opinion via Gemini

## Docker (GPU)

```bash
docker compose up --build
```

Requires NVIDIA Container Toolkit. The UI runs on port 8501.

## Project structure

```
├── run_pipeline.py          # End-to-end pipeline
├── pyproject.toml           # Dependencies
├── .env                     # Gemini API key (gitignored)
├── Dockerfile / docker-compose.yml
├── src/
│   ├── config.py            # Paths & hyperparameters
│   ├── preprocessing/       # Frame extraction, masks, dataset
│   ├── models/
│   │   ├── regression.py    # ResNet18 / ResNet34 (EF regression)
│   │   ├── segmentation.py  # U-Net (LV segmentation)
│   │   └── gemini3_analyzer.py  # Gemini AI structural & clinical analysis
│   ├── training/            # Training loops, early stopping
│   ├── postprocessing/      # LV contour extraction, key points
│   └── visualization/       # Mask overlay, comparison grids
├── ui/
│   └── app.py               # Streamlit app
├── models_saved/            # Trained .pth weights (gitignored)
└── data/
    └── processed/           # Extracted frames, masks (gitignored)
```

## Notes

- Trained on a small subset (25 videos by default) for demo purposes only — not for clinical use.
- Re-running the pipeline automatically cleans and regenerates all outputs.
- The Gemini AI analysis requires a valid API key in `.env` and internet connectivity.
