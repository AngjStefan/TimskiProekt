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
# Sigmoid cut-off used by the app and evaluation. 0.5 beat 0.8 on 30 VAL
# videos (Dice 0.855 vs 0.836, ED area bias -4% vs -13%); 0.8 under-segments.
SEG_THRESHOLD = 0.5

# EchoNet-Dynamic videos are 112x112; measurements are reported in this grid
NATIVE_SIZE = 112

# paths
RAW_VIDEOS = RAW_DIR / "videos"
FILE_LIST = RAW_DIR / "FileList.csv"
VOLUME_TRACINGS = RAW_DIR / "VolumeTracings.csv"
ZIP_PATH = ROOT / "EchoNet-Dynamic.zip"
SUBSET_FILE_LIST = PROCESSED_DIR / "filelist_subset.csv"

# device: CUDA (Linux/Windows GPU) -> Apple MPS -> CPU
def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Regression trains on B x N_FRAMES images per step; 16 x 48 doesn't fit in
# 16 GB of Apple unified memory, so MPS/CPU use a smaller default batch.
def default_regression_batch_size(device) -> int:
    return REGRESSION_BATCH_SIZE if device.type == "cuda" else 2


# split seed
SEED = 42

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
