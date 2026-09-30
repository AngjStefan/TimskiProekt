# EchoNet-Dynamic Demo

Echocardiogram video analysis using deep learning and AI. Trains **ResNet18 / ResNet34** for ejection fraction (EF) regression and a **U-Net** for left ventricle (LV) segmentation on a subset of the [EchoNet-Dynamic](https://echonet.github.io/dynamic/) dataset. The segmentation is turned into clinical measurements (LV area, long axis, method-of-disks volume, EF) and evaluated against the dataset's expert tracings. Includes **Gemini 3 Flash / Pro** integration for AI-powered structural overlays and clinical assessments.

## What it does

1. **Extract frames** from echo videos (from EchoNet-Dynamic.zip)
2. **Build segmentation masks** from the expert volume tracings
3. **Train models**:
   - `ResNet18 / ResNet34` — predicts ejection fraction from video frames
   - `U-Net` — segments the left ventricle in individual frames
4. **Measure the LV** from each segmented frame:
   - landmarks: apex, mitral annulus hinges (basal septal / lateral), mid-cavity walls
   - long axis (apex → annulus midpoint), area, and single-plane volume with 20 disks (Simpson's rule, the method used for the EchoNet labels)
   - EF from the per-frame volume curve, computed per heartbeat (median over beats)
5. **Evaluate** on held-out TEST videos against the expert EF and tracings
6. **Streamlit UI** — upload an echo video and see:
   - EF from the ResNet and from the segmentation, with HF clinical classification (HFrEF / HFmrEF / HFpEF)
   - LV segmentation with contour, long axis, disks and landmarks, plus the measurements and an LV-area-per-frame chart
   - for EchoNet videos: the expert EF and, on expert-traced frames, the expert contour with Dice
   - AI-generated structural overlays and clinical opinion (Gemini 3 Flash / Pro)

## Getting started

### Prerequisites

- Python >= 3.10 (3.12 recommended)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- NVIDIA GPU (CUDA), Apple Silicon (MPS) or CPU — the device is picked automatically
- Google Gemini API key (only for the AI section)

### 1. Download the dataset

Place `EchoNet-Dynamic.zip` in the project root. You can get it from the [EchoNet-Dynamic website](https://echonet.github.io/dynamic/).

### 2. Install dependencies

```bash
uv sync --python 3.12
```

On Linux / Windows this installs the CUDA 12.8 PyTorch wheels; on macOS it installs the standard PyPI wheels with Apple MPS support.

Or with pip:

```bash
pip install -e .
```

### 3. Set up Gemini API key (optional)

Create a `.env` file in the project root (see `.env.example`):

```bash
GEMINI_API_KEY="your-api-key-here"
```

Without a key the app still runs; only "Run AI Analysis" reports an error.

### 4. Run the pipeline

```bash
uv run python run_pipeline.py                  # first 25 videos (smoke run, ~6 min on an M1 Pro)
uv run python run_pipeline.py --subset 300     # first 300 videos
```

This will:
- Extract frames from the first N videos of `FileList.csv` (default `SUBSET_SIZE = 25`); about 75% of them are TRAIN
- Build masks from the expert tracings
- Train ResNet18 (EF regression) and U-Net (LV segmentation); VAL videos are used for early stopping
- Save models to `models_saved/`

Options: `--skip-train` (only re-extract data), `--batch-size N` (ResNet batch; default 16 on CUDA, 2 on MPS / CPU to fit in memory).

### 5. Evaluate on held-out videos

```bash
uv run python -m src.training.evaluate_test_set --n 200
```

Runs both models on the first N TEST-split videos (never used for training) and compares them with the expert labels. Output goes to `reports/eval_<timestamp>/` (gitignored):

- `summary.txt` / `results.csv` — Dice, area and long-axis error, landmark error, EF error, per video
- `<video>_ed_es.png` — expert contour vs U-Net contour with measurements, at the expert ED and ES frames
- `<video>_curve.png` — LV area over the clip with expert and detected ED/ES frames
- `area.png`, `ef_seg.png`, `ef_resnet.png`, `dice.png`, `mask_check.png`
- `videos/` — the evaluated clips, ready to upload to the app

Options: `--split VAL`, `--threshold`, `--backbone`, `--no-figures`.

### 6. Launch the UI

```bash
uv run streamlit run ui/app.py
```

Upload an echocardiogram video (AVI, MP4, MPEG). Clips from `reports/eval_*/videos/` also show the expert comparison.

## Results

U-Net and ResNet18 trained with `--subset 300` (238 TRAIN videos), evaluated on 200 TEST videos:

| Measure | Result |
|---|---|
| Dice vs expert tracing | 0.908 mean, 0.923 median (ED 0.925, ES 0.891) |
| LV area at ED | −1.1% bias (30.2 vs 29.4 cm², mean) |
| Long axis | −6% bias (U-Net stops short of the apex) |
| Landmark error | apex 5.1 mm, basal septal 3.1 mm, basal lateral 4.4 mm, mid 3.1–4.1 mm |
| EF, ResNet18 | MAE 7.4 percentage points (r = 0.62) |
| EF, from segmentation (median over beats) | MAE 6.7 (r = 0.76) |
| EF, predict the training mean | MAE 8.6 (baseline) |

Notes on the measurements:
- Lengths and areas are in pixels of the native 112×112 EchoNet grid; the videos carry no pixel spacing. The evaluation derives cm per pixel for each video from the expert EDV (1 mL = 1 cm³) and reports cm only when ED and ES agree within 5% (198/200 videos).
- Applying the disks formula to the expert tracings reproduces the FileList EF (MAE 1.2), which validates the formula and the tracing format.
- EF from segmentation is computed per heartbeat (ED peak to the following ES minimum, median over beats; 4 beats per clip typically). Reading the clip's overall max/min instead let single glitches inflate EF (MAE 7.7, bias +4.6); the per-beat version has a −2.4 point bias, consistent with slight over-segmentation at ES.

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
│   ├── config.py            # Paths, hyperparameters, device selection
│   ├── preprocessing/       # Frame extraction, masks, dataset
│   ├── models/
│   │   ├── regression.py    # ResNet18 / ResNet34 (EF regression)
│   │   ├── segmentation.py  # U-Net (LV segmentation)
│   │   ├── inference.py     # Shared model loading + inference (app and evaluation)
│   │   └── gemini3_analyzer.py  # Gemini AI structural & clinical analysis
│   ├── training/
│   │   ├── train_*.py       # Training loops, early stopping
│   │   ├── evaluate.py      # Legacy metrics script (expects data/raw/FileList.csv; use evaluate_test_set.py)
│   │   └── evaluate_test_set.py  # Held-out evaluation against expert labels
│   ├── postprocessing/
│   │   ├── contours.py      # LV contour, key points, overlay drawing
│   │   └── measurements.py  # Landmarks, long axis, method of disks, EF
│   └── visualization/       # Mask overlay, metric plots
├── ui/
│   └── app.py               # Streamlit app
├── models_saved/            # Trained .pth weights (gitignored)
├── reports/                 # Evaluation output (gitignored)
└── data/
    └── processed/           # Extracted frames, masks (gitignored)
```

## Notes

- Trained on a subset of EchoNet-Dynamic for demo purposes only — not for clinical use.
- Expert masks are built from the tracing chords only; row 0 of each tracing is the long axis, not a wall point (same as the official EchoNet loader).
- The segmentation threshold is 0.5 (`SEG_THRESHOLD` in `src/config.py`), chosen on VAL videos.
- The app warns when a checkpoint is missing instead of running untrained weights.
- Re-running the pipeline automatically cleans and regenerates all outputs.
- The Gemini AI analysis requires a valid API key in `.env` and internet connectivity.
