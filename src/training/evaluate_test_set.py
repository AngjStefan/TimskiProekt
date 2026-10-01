"""
Evaluate the trained models on held-out EchoNet-Dynamic videos against the
expert labels (FileList.csv EF/EDV/ESV + VolumeTracings.csv contours).

For each video it segments every frame, measures the LV (area, long axis,
method-of-disks volume, landmarks), derives EF from the volume curve, runs the
EF regressor, and compares everything with the expert ED/ES tracings.

Usage:
    uv run python -m src.training.evaluate_test_set --n 50

Outputs (reports/ is gitignored):
    reports/eval_<timestamp>/results.csv      one row per video
    reports/eval_<timestamp>/summary.txt      aggregate metrics
    reports/eval_<timestamp>/<video>_ed_es.png expert vs predicted contour + measurements
    reports/eval_<timestamp>/<video>_curve.png LV area over the clip, expert ED/ES marked
    reports/eval_<timestamp>/ef_*.png, dice.png, area.png, mask_check.png
    reports/eval_<timestamp>/videos/*.avi     the evaluated clips, for the app

Measurements are in native EchoNet pixels (112x112 grid). Optional cm units
use a per-video scale derived from the expert EDV (see `calibration` below).
"""
import argparse
import io
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    ROOT, ZIP_PATH, FRAME_SIZE_SEG, NATIVE_SIZE, SEG_THRESHOLD, SUBSET_FILE_LIST, get_device,
)
from src.models.inference import load_unet, load_regression, segment_frames, predict_ef
from src.postprocessing.contours import draw_contour_and_points
from src.postprocessing.measurements import (
    measure_mask, gt_from_tracing, ef_from_volume_curve, dice,
)
from src.preprocessing.extract_frames import extract_frames_from_avi
from src.visualization.metrics_plot import plot_regression_results, plot_segmentation_results

S = FRAME_SIZE_SEG / NATIVE_SIZE          # analysis grid -> native pixels
LANDMARKS = ("apex", "basal_septal", "basal_lateral", "mid_septal", "mid_lateral")
CALIBRATION_TOL = 0.05                    # ED/ES cm-per-px must agree within 5%


def _stem(name) -> str:
    return Path(str(name)).name.replace(".avi", "")


def load_labels(z: zipfile.ZipFile):
    names = z.namelist()
    fl_entry = next(n for n in names if n.endswith("FileList.csv"))
    vt_entry = next(n for n in names if n.endswith("VolumeTracings.csv"))
    filelist = pd.read_csv(io.BytesIO(z.read(fl_entry)))
    tracings = pd.read_csv(io.BytesIO(z.read(vt_entry)))
    filelist["stem"] = filelist["FileName"].map(_stem)
    tracings["stem"] = tracings["FileName"].map(_stem)
    videos = {_stem(n): n for n in names if n.endswith(".avi")}
    return filelist, tracings, videos


def expert_frames(tracings: pd.DataFrame, stem: str, T: int) -> dict | None:
    """{frame: (K,4) rows} for the two traced frames, in CSV order."""
    t = tracings[tracings["stem"] == stem]
    groups = {int(f): g[["X1", "Y1", "X2", "Y2"]].to_numpy(float)
              for f, g in t.groupby("Frame", sort=False)}
    groups = {f: r for f, r in groups.items() if f < T and len(r) >= 3}
    return groups if len(groups) == 2 else None


def calibration(edv_ml: float, esv_ml: float, v_ed_px: float, v_es_px: float):
    """cm per native pixel from the expert volumes (1 mL = 1 cm^3).

    The FileList volumes were measured on the original-resolution images; the
    tracings were rescaled to 112x112. If both frames give the same scale, the
    scale is trusted and areas/lengths can be reported in cm.
    """
    if min(edv_ml, esv_ml, v_ed_px, v_es_px) <= 0:
        return np.nan, np.nan, False
    c_ed = (edv_ml / v_ed_px) ** (1 / 3)
    c_es = (esv_ml / v_es_px) ** (1 / 3)
    return c_ed, c_es, abs(c_es / c_ed - 1) <= CALIBRATION_TOL


def _rgb(frame_gray: np.ndarray) -> np.ndarray:
    return np.stack([frame_gray] * 3, axis=-1).astype(np.uint8)


def _poly(pts) -> np.ndarray:
    return np.rint(pts).astype(np.int32).reshape(-1, 1, 2)


def _draw_gt(image, gt) -> np.ndarray:
    kp = {k: gt[k] for k in (*LANDMARKS, "base_mid")}
    return draw_contour_and_points(image, _poly(gt["polygon"]), kp, color=(0, 255, 0), chords=gt["chords"])


def _draw_pred(image, m) -> np.ndarray:
    if m is None:
        return image
    kp = {k: m[k] for k in (*LANDMARKS, "base_mid")}
    return draw_contour_and_points(image, m["contour"], kp, color=(255, 60, 60), chords=m["chords"])


def figure_ed_es(stem, frames224, per_frame, gt, rec, out: Path):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 8.2))
    for row, phase in enumerate(("ed", "es")):
        f = rec[f"{phase}_frame_gt"]
        img = _rgb(frames224[f])
        g, m = gt[phase], per_frame[f]

        both = img.copy()
        cv2.drawContours(both, [_poly(g["polygon"])], -1, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        if m is not None:
            cv2.drawContours(both, [m["contour"]], -1, (255, 60, 60), 1, lineType=cv2.LINE_AA)

        panels = [
            (both, f"{phase.upper()} frame {f}: expert (green) vs U-Net (red)\n"
                   f"Dice {rec[f'dice_{phase}']:.3f}"),
            (_draw_pred(img, m), f"U-Net measurement\n"
                                 f"area {rec[f'area_pred_{phase}']:.0f} px², L {rec[f'L_pred_{phase}']:.1f} px"),
            (_draw_gt(img, g), f"Expert tracing\n"
                               f"area {rec[f'area_gt_{phase}']:.0f} px², L {rec[f'L_gt_{phase}']:.1f} px"),
        ]
        for ax, (im, title) in zip(axes[row], panels):
            ax.imshow(im, interpolation="lanczos")
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    errs = ", ".join(f"{k.replace('_', ' ')} {np.nanmean([rec[f'err_{k}_ed'], rec[f'err_{k}_es']]):.1f}"
                     for k in LANDMARKS)
    fig.suptitle(
        f"{stem}  |  EF expert {rec['ef_gt']:.1f}%  ·  from segmentation {rec['ef_seg']:.1f}%  ·  "
        f"ResNet {rec['ef_resnet']:.1f}%\nlandmark error (px, mean ED/ES): {errs}",
        fontsize=9.5,
    )
    fig.tight_layout(h_pad=2.5)
    fig.savefig(out / f"{stem}_ed_es.png", dpi=140)
    plt.close(fig)


def figure_curve(stem, areas, ef_seg, rec, fps, out: Path):
    t = np.arange(len(areas)) / fps if fps else np.arange(len(areas))
    fig, ax = plt.subplots(figsize=(9, 3.9))
    ax.plot(t, areas, color="#2563eb", lw=1.2, label="U-Net LV area (per frame)")
    for key, color, label in (("ed_frame_gt", "#16a34a", "expert ED frame"),
                              ("es_frame_gt", "#ea580c", "expert ES frame")):
        ax.axvline(t[rec[key]], color=color, ls="--", lw=1.2, label=label)
    if ef_seg is not None:
        beats = ef_seg["beats"] or [(ef_seg["ed_frame"], ef_seg["es_frame"], ef_seg["ef"])]
        eds, ess = [b[0] for b in beats], [b[1] for b in beats]
        ax.plot(t[eds], areas[eds], "v", color="#16a34a", ms=8, label="detected ED (per beat)")
        ax.plot(t[ess], areas[ess], "^", color="#ea580c", ms=8, label="detected ES (per beat)")
    ax.set_xlabel("time (s)" if fps else "frame")
    ax.set_ylabel("LV area (px²)")
    ax.set_title(f"{stem} — EF expert {rec['ef_gt']:.1f}%, from segmentation {rec['ef_seg']:.1f}% "
                 f"(median of {rec['n_beats']} beats)", fontsize=10)
    ax.legend(fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"{stem}_curve.png", dpi=140)
    plt.close(fig)


def figure_mask_check(stem, frame224, rows, out: Path):
    """Old mask (long-axis row included) vs fixed mask for one expert tracing."""
    r = rows * S
    old = np.concatenate([r[:, :2], r[::-1, 2:]])
    new = np.concatenate([r[1:, :2], r[:0:-1, 2:]])
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.8))
    for ax, poly, title in ((axes[0], old, "before fix: row 0 (long axis) in polygon"),
                            (axes[1], new, "after fix: chords only")):
        m = np.zeros(frame224.shape, np.uint8)
        cv2.fillPoly(m, [_poly(poly)], 1)
        img = _rgb(frame224)
        img[m > 0] = (0.55 * img[m > 0] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Expert mask construction — {stem}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "mask_check.png", dpi=140)
    plt.close(fig)


def evaluate_video(stem, row, masks, gt_rows, fps, ef_resnet):
    per_frame = [measure_mask(m) if m.any() else None for m in masks]
    areas = np.array([p["area"] / S**2 if p else np.nan for p in per_frame])
    vols = np.array([p["volume"] / S**3 if p else np.nan for p in per_frame])
    ef_seg = ef_from_volume_curve(vols, fps=fps)

    gt_by_frame = {f: gt_from_tracing(r, scale=S, shape=masks.shape[1:]) for f, r in gt_rows.items()}
    ed_f, es_f = sorted(gt_by_frame, key=lambda f: gt_by_frame[f]["volume"], reverse=True)
    gt = {"ed": gt_by_frame[ed_f], "es": gt_by_frame[es_f]}

    rec = {
        "file": stem, "n_frames": len(masks), "fps": fps,
        "ef_gt": float(row["EF"]), "edv_ml": float(row["EDV"]), "esv_ml": float(row["ESV"]),
        "ed_frame_gt": ed_f, "es_frame_gt": es_f,
        "frames_without_seg": int(np.sum(~np.isfinite(vols))),
        "ef_resnet": ef_resnet,
        "ef_seg": ef_seg["ef"] if ef_seg else np.nan,
        "ed_frame_pred": ef_seg["ed_frame"] if ef_seg else -1,
        "es_frame_pred": ef_seg["es_frame"] if ef_seg else -1,
        "n_beats": len(ef_seg["beats"]) if ef_seg else 0,
        "ef_method": ef_seg["method"] if ef_seg else "",
    }
    v_gt = {ph: gt[ph]["volume"] / S**3 for ph in gt}
    rec["ef_trace"] = 100 * (v_gt["ed"] - v_gt["es"]) / v_gt["ed"]

    v_pred = {}
    for ph, f in (("ed", ed_f), ("es", es_f)):
        g, m = gt[ph], per_frame[f]
        rec[f"area_gt_{ph}"] = g["area"] / S**2
        rec[f"L_gt_{ph}"] = g["length"] / S
        if m is None:
            rec[f"dice_{ph}"] = 0.0
            rec[f"iou_{ph}"] = 0.0
            rec[f"area_pred_{ph}"] = rec[f"L_pred_{ph}"] = np.nan
            for k in LANDMARKS:
                rec[f"err_{k}_{ph}"] = np.nan
            continue
        d = dice(m["region"], g["region"])
        rec[f"dice_{ph}"] = d
        rec[f"iou_{ph}"] = d / (2 - d)
        rec[f"area_pred_{ph}"] = m["area"] / S**2
        rec[f"L_pred_{ph}"] = m["length"] / S
        v_pred[ph] = m["volume"] / S**3
        for k in LANDMARKS:
            rec[f"err_{k}_{ph}"] = float(np.linalg.norm(np.asarray(m[k]) - g[k]) / S)
    rec["ef_seg_at_expert_frames"] = (
        100 * (v_pred["ed"] - v_pred["es"]) / v_pred["ed"] if len(v_pred) == 2 and v_pred["ed"] > 0 else np.nan
    )

    c_ed, c_es, ok = calibration(rec["edv_ml"], rec["esv_ml"], v_gt["ed"], v_gt["es"])
    rec["cm_per_px_ed"], rec["cm_per_px_es"], rec["calibrated"] = c_ed, c_es, ok
    c = (c_ed + c_es) / 2 if ok else np.nan
    for ph in ("ed", "es"):
        rec[f"area_pred_{ph}_cm2"] = rec[f"area_pred_{ph}"] * c**2
        rec[f"area_gt_{ph}_cm2"] = rec[f"area_gt_{ph}"] * c**2
        rec[f"L_pred_{ph}_cm"] = rec[f"L_pred_{ph}"] * c
        rec[f"L_gt_{ph}_cm"] = rec[f"L_gt_{ph}"] * c

    return rec, per_frame, areas, ef_seg, gt


def summarize(df: pd.DataFrame, baseline_ef: float, baseline_src: str) -> str:
    def mae(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return np.abs(a[ok] - b[ok]).mean(), ok.sum()

    def corr(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 2 else np.nan

    gt = df["ef_gt"].to_numpy()
    lines = [f"Videos evaluated: {len(df)} (held-out split, never used for training)", ""]

    dices = np.concatenate([df["dice_ed"], df["dice_es"]])
    lines += [
        "Segmentation vs expert tracing (ED + ES frames)",
        f"  Dice  mean {dices.mean():.3f}  median {np.median(dices):.3f}  (ED {df['dice_ed'].mean():.3f}, ES {df['dice_es'].mean():.3f})",
        f"  IoU   mean {np.concatenate([df['iou_ed'], df['iou_es']]).mean():.3f}",
    ]
    for ph in ("ed", "es"):
        rel = (df[f"area_pred_{ph}"] - df[f"area_gt_{ph}"]) / df[f"area_gt_{ph}"] * 100
        relL = (df[f"L_pred_{ph}"] - df[f"L_gt_{ph}"]) / df[f"L_gt_{ph}"] * 100
        lines.append(f"  {ph.upper()} area error {rel.mean():+.1f}% (mean abs {rel.abs().mean():.1f}%), "
                     f"long-axis error {relL.mean():+.1f}% (mean abs {relL.abs().mean():.1f}%)")
    lines += ["", "Landmark error vs expert (native px, mean over ED+ES)"]
    for k in LANDMARKS:
        e = np.concatenate([df[f"err_{k}_ed"], df[f"err_{k}_es"]])
        lines.append(f"  {k:14s} {np.nanmean(e):5.2f} px  (median {np.nanmedian(e):.2f})")

    lines += ["", "Ejection fraction vs expert FileList EF (percentage points)"]
    for col, name in (("ef_trace", "disks formula on expert tracing (sanity check)"),
                      ("ef_seg", "from segmentation (median over beats)"),
                      ("ef_seg_at_expert_frames", "from segmentation at expert ED/ES frames"),
                      ("ef_resnet", "ResNet regression")):
        m, n = mae(df[col].to_numpy(), gt)
        lines.append(f"  {name:48s} MAE {m:5.2f}  r={corr(df[col].to_numpy(), gt):.2f}  (n={n})")
    lines.append(f"  {'baseline: always predict ' + f'{baseline_ef:.1f}% ({baseline_src})':48s} "
                 f"MAE {np.abs(gt - baseline_ef).mean():5.2f}")
    beats = df.loc[df["ef_method"] == "beats", "n_beats"]
    lines.append(f"  beats per video (segmentation EF): median {beats.median():.0f}; "
                 f"max/min fallback used for {(df['ef_method'] != 'beats').sum()} videos")

    cal = df["calibrated"].astype(bool)
    lines += ["", f"cm calibration consistent (ED vs ES within {CALIBRATION_TOL:.0%}) for {cal.sum()}/{len(df)} videos"]
    if cal.any():
        c = df.loc[cal]
        lines.append(f"  ED area expert {c['area_gt_ed_cm2'].mean():.1f} cm² vs U-Net {c['area_pred_ed_cm2'].mean():.1f} cm² (mean)")
        lines.append(f"  ED long axis expert {c['L_gt_ed_cm'].mean():.2f} cm vs U-Net {c['L_pred_ed_cm'].mean():.2f} cm (mean)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50, help="number of videos to evaluate")
    ap.add_argument("--split", default="TEST", choices=["TEST", "VAL"])
    ap.add_argument("--zip", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--threshold", type=float, default=SEG_THRESHOLD)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    zip_path = args.zip or next((p for p in (ZIP_PATH, ROOT / "data" / "raw" / "EchoNet-Dynamic.zip") if p.exists()), None)
    if zip_path is None or not zip_path.exists():
        sys.exit("EchoNet-Dynamic.zip not found. Place it in the project root or pass --zip.")

    device = get_device()
    unet, unet_ok = load_unet(device)
    reg, reg_ok = load_regression(device, args.backbone)
    if not unet_ok or not reg_ok:
        sys.exit("Missing checkpoint(s) in models_saved/ (unet_lv.pth and "
                 f"{args.backbone}_ef.pth). Train first: uv run python run_pipeline.py")

    out = args.out or ROOT / "reports" / f"eval_{datetime.now():%Y%m%d_%H%M%S}"
    (out / "videos").mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  |  output: {out}")

    z = zipfile.ZipFile(zip_path)
    filelist, tracings, videos = load_labels(z)
    traced = set(tracings["stem"])
    cand = filelist[(filelist["Split"] == args.split) & filelist["stem"].isin(traced) & filelist["stem"].isin(videos)]

    if SUBSET_FILE_LIST.exists():
        sub = pd.read_csv(SUBSET_FILE_LIST)
        baseline_ef, baseline_src = sub.loc[sub["Split"] == "TRAIN", "EF"].mean(), "mean EF of training subset"
    else:
        baseline_ef, baseline_src = filelist.loc[filelist["Split"] == "TRAIN", "EF"].mean(), "mean EF of TRAIN split"

    records, mask_check_done = [], False
    for _, row in tqdm(cand.iterrows(), total=min(args.n, len(cand)), desc=f"Evaluating {args.split}"):
        if len(records) >= args.n:
            break
        stem = row["stem"]
        raw = z.read(videos[stem])
        frames224 = extract_frames_from_avi(raw, FRAME_SIZE_SEG)
        gt_rows = expert_frames(tracings, stem, len(frames224))
        if gt_rows is None:
            continue

        masks, _ = segment_frames(unet, frames224, device, threshold=args.threshold)
        ef_resnet = predict_ef(reg, frames224, device)
        fps = float(row["FPS"]) if "FPS" in row and pd.notna(row["FPS"]) else None
        rec, per_frame, areas, ef_seg, gt = evaluate_video(stem, row, masks, gt_rows, fps, ef_resnet)
        records.append(rec)
        (out / "videos" / f"{stem}.avi").write_bytes(raw)

        if not args.no_figures:
            figure_ed_es(stem, frames224, per_frame, gt, rec, out)
            figure_curve(stem, areas, ef_seg, rec, fps, out)
            if not mask_check_done:
                figure_mask_check(stem, frames224[rec["ed_frame_gt"]], gt_rows[rec["ed_frame_gt"]], out)
                mask_check_done = True
    z.close()

    if not records:
        sys.exit("No evaluable videos found.")
    df = pd.DataFrame(records)
    df.to_csv(out / "results.csv", index=False)

    ok = df["ef_seg"].notna()
    plot_regression_results(df["ef_gt"].to_numpy(), df["ef_resnet"].to_numpy(), "ResNet18 EF",
                            save_path=str(out / "ef_resnet.png"))
    if ok.sum() > 1:
        plot_regression_results(df.loc[ok, "ef_gt"].to_numpy(), df.loc[ok, "ef_seg"].to_numpy(),
                                "EF from segmentation", save_path=str(out / "ef_seg.png"))
    plot_segmentation_results(np.concatenate([df["dice_ed"], df["dice_es"]]),
                              np.concatenate([df["iou_ed"], df["iou_es"]]), save_path=str(out / "dice.png"))

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for ph, color in (("ed", "#16a34a"), ("es", "#ea580c")):
        ax.scatter(df[f"area_gt_{ph}"], df[f"area_pred_{ph}"], s=14, alpha=0.7, color=color, label=ph.upper())
    lim = [0, np.nanmax(df[["area_gt_ed", "area_pred_ed"]].to_numpy()) * 1.08]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("expert LV area (px²)"); ax.set_ylabel("U-Net LV area (px²)")
    ax.set_title("LV area: U-Net vs expert tracing", fontsize=10)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "area.png", dpi=140); plt.close(fig)

    summary = summarize(df, baseline_ef, baseline_src)
    (out / "summary.txt").write_text(summary + "\n")
    print("\n" + summary)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
