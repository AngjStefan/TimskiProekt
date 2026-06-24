import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = ROOT / "models_saved"
UI_DIR = ROOT / "ui"

# subset (брз демо тренинг)
SUBSET_SIZE = 25

# preprocessing
FRAME_SIZE = 224
FRAME_SIZE_SEG = 224
N_FRAMES = 48
TARGET_FPS = 50

# regression
REGRESSION_BACKBONE = "resnet34"
REGRESSION_IN_CHANNELS = 3
REGRESSION_LR = 1e-4
REGRESSION_WEIGHT_DECAY = 1e-5
REGRESSION_BATCH_SIZE = 16
REGRESSION_EPOCHS = 35
REGRESSION_PATIENCE = 10


# segmentation
SEG_LR = 1e-4
SEG_WEIGHT_DECAY = 1e-4
SEG_BATCH_SIZE = 16
SEG_EPOCHS = 50
SEG_PATIENCE = 12
SEG_N_CLASSES = 1

# paths
RAW_VIDEOS = RAW_DIR / "videos"
FILE_LIST = RAW_DIR / "FileList.csv"
VOLUME_TRACINGS = RAW_DIR / "VolumeTracings.csv"
ZIP_PATH = ROOT / "EchoNet-Dynamic.zip"
SUBSET_FILE_LIST = PROCESSED_DIR / "filelist_subset.csv"

# device
DEVICE = "cuda"

# split seed
SEED = 42

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
