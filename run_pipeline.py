#!/usr/bin/env python3
"""
Demo pipeline:
  1. Find EchoNet-Dynamic.zip in project root
  2. Extract N videos (subset) → data/processed/frames/
  3. Build masks for those N videos → data/processed/masks/
  4. Train ResNet18 + U-Net on subset (~15-20 min on CPU)
  5. Done — start UI separately
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import ROOT, PROCESSED_DIR, MODELS_DIR, SUBSET_SIZE, SUBSET_FILE_LIST
from src.preprocessing.extract_frames import preprocess_all
from src.preprocessing.build_masks import build_all_masks


def find_zip() -> Path:
    candidates = [
        ROOT / "EchoNet-Dynamic.zip",
        ROOT / "data" / "raw" / "EchoNet-Dynamic.zip",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "EchoNet-Dynamic.zip not found. Place it in the project root."
    )


# def get_subset_video_ids(zip_path: Path, n: int = SUBSET_SIZE) -> list[str]:
#     z = zipfile.ZipFile(str(zip_path), "r")
#     videos = sorted(v for v in z.namelist() if v.endswith(".avi"))[:n]
#     z.close()
#     ids = [Path(v).stem for v in videos]
#     print(f"Using {len(ids)} videos for demo: {ids[:3]} ... {ids[-1]}")
#     return ids
def get_subset_video_ids(zip_path: Path, n: int):
    import pandas as pd
    import io

    z = zipfile.ZipFile(str(zip_path), "r")

    filelist_entry = next(
        x for x in z.namelist()
        if x.endswith("FileList.csv")
    )

    df = pd.read_csv(io.BytesIO(z.read(filelist_entry)))

    subset = df.iloc[:n].copy()

    ids = (
        subset["FileName"]
        .astype(str)
        .str.replace(".avi", "", regex=False)
        .tolist()
    )

    z.close()

    print(f"\nSelected first {len(ids)} videos")

    print(
        subset["Split"]
        .value_counts()
        .to_dict()
    )

    return ids


def step_extract(zip_path: Path, video_ids: list[str]):
    print("\n" + "=" * 50)
    print(f"Step 1: Extract {len(video_ids)} videos → frames")
    print("=" * 50)
    dest = PROCESSED_DIR / "frames"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    preprocess_all(zip_path, dest, video_ids=video_ids)


# def create_subset_filelist(zip_path: Path, video_ids: list[str]):
#     import pandas as pd
#     import io
#     z = zipfile.ZipFile(str(zip_path), "r")
#     entry = next(n for n in z.namelist() if n.endswith("FileList.csv"))
#     df = pd.read_csv(io.BytesIO(z.read(entry)))
#     z.close()
#     df = df[df["FileName"].isin(video_ids)].copy()
#     df["Split"] = "TRAIN"
#     df.to_csv(SUBSET_FILE_LIST, index=False)
#     print(f"Created subset filelist: {len(df)} videos → {SUBSET_FILE_LIST}")
def create_subset_filelist(zip_path, video_ids):
    import pandas as pd
    import io

    z = zipfile.ZipFile(str(zip_path))

    entry = next(
        x for x in z.namelist()
        if x.endswith("FileList.csv")
    )

    df = pd.read_csv(
        io.BytesIO(z.read(entry))
    )

    z.close()

    subset = df[
        df["FileName"]
        .astype(str)
        .str.replace(".avi", "", regex=False)
        .isin(video_ids)
    ].copy()

    subset.to_csv(
        SUBSET_FILE_LIST,
        index=False,
    )

    print(
        "\nSaved split distribution:"
    )

    print(
        subset["Split"]
        .value_counts()
    )

def step_masks(zip_path: Path, video_ids: list[str]):
    print("\n" + "=" * 50)
    print(f"Step 2: Build masks for {len(video_ids)} videos")
    print("=" * 50)
    masks_dir = PROCESSED_DIR / "masks"
    if masks_dir.exists():
        shutil.rmtree(masks_dir)
    masks_dir.mkdir(parents=True)
    z = zipfile.ZipFile(str(zip_path), "r")
    entry_name = None
    for n in z.namelist():
        if n.endswith("VolumeTracings.csv"):
            entry_name = n
            break
    if entry_name:
        import pandas as pd
        import io
        df = pd.read_csv(io.BytesIO(z.read(entry_name)))
        z.close()
        video_set = set(video_ids)
        df = df[df["FileName"].str.replace(".avi", "").isin(video_set)]
        tmp_csv = PROCESSED_DIR / "tracings_subset.csv"
        df.to_csv(tmp_csv, index=False)
        build_all_masks(tracings_csv=tmp_csv)
        tmp_csv.unlink()
    else:
        z.close()
        print("VolumeTracings.csv not found in zip, skipping masks")


def step_train_regression(backbone: str, subset_size: int):
    print(f"\n" + "=" * 50)
    print(f"Step 3: Train {backbone} (subset={subset_size})")
    print("=" * 50)
    ckpt = MODELS_DIR / f"{backbone}_ef.pth"
    ckpt.unlink(missing_ok=True)
    from src.training.train_regression import main as train_reg
    train_reg(backbone=backbone, subset_size=subset_size,
              file_list=str(SUBSET_FILE_LIST))


def step_train_segmentation(subset_size: int):
    print(f"\n" + "=" * 50)
    print(f"Step 4: Train U-Net (subset={subset_size})")
    print("=" * 50)
    ckpt = MODELS_DIR / "unet_lv.pth"
    ckpt.unlink(missing_ok=True)
    from src.training.train_segmentation import main as train_seg
    train_seg(subset_size=subset_size, file_list=str(SUBSET_FILE_LIST))


def main():
    parser = argparse.ArgumentParser(description="EchoNet-Dynamic Demo Pipeline")
    parser.add_argument("--subset", type=int, default=None,
                        help=f"Number of videos to use (default: {None})")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training if models already exist")
    args = parser.parse_args()

    zip_path = find_zip()
    print(f"Found zip: {zip_path}")

    video_ids = get_subset_video_ids(
        zip_path,
        args.subset or SUBSET_SIZE
    )

    step_extract(zip_path, video_ids)
    create_subset_filelist(zip_path, video_ids)
    step_masks(zip_path, video_ids)

    if not args.skip_train:
        step_train_regression("resnet18", args.subset)
        step_train_segmentation(args.subset)
    else:
        print("\nSkipping training (--skip-train)")
        for name in ["resnet18_ef.pth", "unet_lv.pth"]:
            p = MODELS_DIR / name
            print(f"  {'✓' if p.exists() else '✗'} {name}")

    print("\n✓ Pipeline complete.")
    print("  Start the UI separately:  uv run streamlit run ui/app.py")


if __name__ == "__main__":
    main()
