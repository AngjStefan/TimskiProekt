from pathlib import Path
from src.config import ROOT, RAW_DIR


def find_local_zip() -> Path | None:
    candidates = [
        ROOT / "EchoNet-Dynamic.zip",
        RAW_DIR / "EchoNet-Dynamic.zip",
        Path.home() / "Downloads" / "EchoNet-Dynamic.zip",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def ensure_data() -> Path:
    zip_path = find_local_zip()
    if zip_path is not None:
        print(f"Found zip: {zip_path}")
        return zip_path
    raise FileNotFoundError(
        "EchoNet-Dynamic.zip not found. "
        "Place it in the project root directory."
    )
