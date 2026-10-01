import sys
import io
import base64
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import zipfile
import plotly.graph_objects as go

from PIL import Image

from src.config import (
    ROOT, ZIP_PATH, MODELS_DIR, FRAME_SIZE_SEG, NATIVE_SIZE, get_device,
)
from src.models import inference
from src.preprocessing.extract_frames import extract_single_video
from src.postprocessing.contours import process_mask
from src.postprocessing.measurements import measure_mask, ef_from_volume_curve, gt_from_tracing, dice
from src.visualization.overlay import overlay_mask
from src.models.gemini3_analyzer import generate_image, generate_medical_opinion

st.set_page_config(page_title="Echocardiogram Analysis with AI", layout="wide")

st.markdown("""
<style>
    .main .block-container {
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        max-width: 1400px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Echocardiogram Analysis with AI")
st.markdown(
    "Upload an echocardiogram video to analyze ejection fraction, LV segmentation key points, and AI-powered clinical insights."
)
st.caption(
    "⚠️ **Demo mode**: models trained on a small subset of EchoNet-Dynamic "
    "for demonstration purposes only. Not for clinical use."
)

device = get_device()
S = FRAME_SIZE_SEG / NATIVE_SIZE   # analysis grid (224) -> EchoNet pixels (112)


@st.cache_resource
def load_regression(backbone: str):
    return inference.load_regression(device, backbone)


@st.cache_resource
def load_unet():
    return inference.load_unet(device)


@st.cache_resource
def load_expert_labels():
    """EchoNet FileList + VolumeTracings from the dataset zip, if present."""
    from src.training.evaluate_test_set import load_labels
    zip_path = next((p for p in (ZIP_PATH, ROOT / "data" / "raw" / "EchoNet-Dynamic.zip") if p.exists()), None)
    if zip_path is None:
        return None
    with zipfile.ZipFile(zip_path) as z:
        filelist, tracings, _ = load_labels(z)
    return filelist, tracings


def expert_for_upload(filename: str, n_frames: int):
    """Expert EF and ED/ES tracings when the upload is an EchoNet-Dynamic video."""
    labels = load_expert_labels()
    if labels is None:
        return None
    from src.training.evaluate_test_set import expert_frames
    filelist, tracings = labels
    stem = Path(filename).stem
    row = filelist[filelist["stem"] == stem]
    if row.empty:
        return None
    row = row.iloc[0]
    frames = expert_frames(tracings, stem, n_frames) or {}
    gts = {f: gt_from_tracing(r, scale=S, shape=(FRAME_SIZE_SEG, FRAME_SIZE_SEG)) for f, r in frames.items()}
    phases = {}
    if len(gts) == 2:
        ed, es = sorted(gts, key=lambda f: gts[f]["volume"], reverse=True)
        phases = {ed: "ED", es: "ES"}
    return {"ef": float(row["EF"]), "split": row["Split"], "gt": gts, "phase": phases}


def render_overlay(frame_gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Mask tint + contour, long axis, disks and landmarks for one frame."""
    frame_rgb = np.stack([frame_gray] * 3, axis=-1).astype(np.uint8)
    tinted = overlay_mask(frame_rgb, mask, color=(0, 255, 0), alpha=0.25)
    data = process_mask(mask, tinted)
    return data["overlay"], data


def get_ef_classification(ef: float) -> dict:
    """Return classification dict based on EF percentage."""
    if ef <= 40:
        return {
            "label": "Heart Failure with Reduced Ejection Fraction (HFrEF)",
            "badge": "HFrEF",
            "color": "#dc3545",
            "bg": "#fce4e4",
            "description": "Also known as systolic heart failure, the heart muscle is weakened and cannot pump enough blood to the body.",
            "treatment": "Standard therapies include ACE inhibitors, ARBs, beta-blockers, and mineralocorticoid receptor antagonists (MRAs) to reduce hospitalizations and mortality.",
        }
    elif ef <= 49:
        return {
            "label": "Heart Failure with Mildly Reduced Ejection Fraction (HFmrEF)",
            "badge": "HFmrEF",
            "color": "#e67e22",
            "bg": "#fef5e7",
            "description": "The heart's pumping function is borderline. Patients in this category often have signs of high filling pressures and structural heart changes.",
            "treatment": "Management relies on treating underlying conditions (like high blood pressure) and careful use of HFrEF medications.",
        }
    else:
        return {
            "label": "Heart Failure with Preserved Ejection Fraction (HFpEF)",
            "badge": "HFpEF",
            "color": "#27ae60",
            "bg": "#e8f8f0",
            "description": "Also known as diastolic heart failure, the heart pumps out a normal percentage of blood. However, the muscle is stiff or thickened, meaning it cannot relax enough to fill with an adequate amount of blood between beats.",
            "treatment": "Focuses on lifestyle changes, managing blood pressure, and medications aimed at relieving symptoms and improving exercise capacity.",
        }


def _show_gif(gif_path, placeholder):
    if not gif_path or not os.path.exists(gif_path):
        placeholder.info("Unavailable")
        return
    with open(gif_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    placeholder.markdown(
        f'<div style="max-width:600px;max-height:600px;margin:0 auto;text-align:center;padding-bottom:1rem;">'
        f'<img src="data:image/gif;base64,{b64}" '
        f'style="width:100%;height:100%;object-fit:contain;display:block;">'
        f'</div>',
        unsafe_allow_html=True,
    )


def _create_gif(frames_rgb, fps_out=20, max_frames=60, loop=0, target_size=None):
    step = max(1, len(frames_rgb) // max_frames)
    selected = [frames_rgb[i] for i in range(0, len(frames_rgb), step)]
    if target_size:
        selected = [cv2.resize(f, target_size, interpolation=cv2.INTER_AREA) for f in selected]
    duration = int(1000 / fps_out * step)
    pil_frames = [Image.fromarray(f).convert("P", palette=Image.Palette.ADAPTIVE) for f in selected]
    path = tempfile.NamedTemporaryFile(delete=False, suffix=".gif").name
    pil_frames[0].save(
        path, save_all=True, append_images=pil_frames[1:],
        loop=loop, duration=duration,
    )
    return path


uploaded_file = st.file_uploader("Choose an echocardiogram video", type=["avi", "mp4", "mpeg"])

if uploaded_file is not None:
    ext = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Extracting frames ..."):
        frames_gray = extract_single_video(tmp_path, target_size=FRAME_SIZE_SEG)  # (T, H, W)
    T = frames_gray.shape[0]

    model_unet, unet_loaded = load_unet()
    model18, reg_loaded = load_regression("resnet18")
    missing = [n for n, ok in (("unet_lv.pth", unet_loaded), ("resnet18_ef.pth", reg_loaded)) if not ok]
    if missing:
        st.warning(
            f"Checkpoint(s) not found in `{MODELS_DIR.name}/`: {', '.join(missing)}. "
            "Results below come from **untrained** weights. Train first: `uv run python run_pipeline.py`."
        )

    original_gif_path = None
    processed_gif_path = None
    processed_video_path = None
    ef_predicted = None

    upload_id = f"{uploaded_file.name}_{uploaded_file.size}"
    need_processing = st.session_state.get("_upload_id") != upload_id

    with st.spinner("Loading analysis ..."):
        # --- Clean up old files from previous upload ---
        if need_processing:
            st.session_state._upload_id = upload_id
            for old_key in ["_orig_gif", "_proc_gif", "_proc_mp4"]:
                old_path = st.session_state.get(old_key)
                if old_path and os.path.exists(old_path):
                    try:
                        os.unlink(old_path)
                    except Exception:
                        pass
                    st.session_state[old_key] = None
            for key in ["_masks", "_areas", "_volumes", "_ef_seg", "_ef_resnet"]:
                st.session_state.pop(key, None)

        # --- EF prediction (regression) ---
        if "_ef_resnet" not in st.session_state:
            try:
                st.session_state["_ef_resnet"] = inference.predict_ef(model18, frames_gray, device)
            except Exception as e:
                st.error(f"ResNet18: {e}")
                st.session_state["_ef_resnet"] = None
        ef_predicted = st.session_state["_ef_resnet"]

        # --- U-Net on every frame + measurements (cached per upload) ---
        if "_masks" not in st.session_state:
            try:
                masks, _ = inference.segment_frames(model_unet, frames_gray, device)
                meas = [measure_mask(m) if m.any() else None for m in masks]
                st.session_state["_masks"] = masks
                st.session_state["_areas"] = np.array([m["area"] / S**2 if m else np.nan for m in meas])
                st.session_state["_volumes"] = np.array([m["volume"] / S**3 if m else np.nan for m in meas])
                cap = cv2.VideoCapture(tmp_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or None
                cap.release()
                st.session_state["_ef_seg"] = ef_from_volume_curve(st.session_state["_volumes"], fps=fps)
            except Exception as e:
                st.error(f"U-Net segmentation: {e}")
        masks = st.session_state.get("_masks")
        areas = st.session_state.get("_areas")
        ef_seg = st.session_state.get("_ef_seg")

        # --- GIFs / MP4 of the overlay (cached per upload) ---
        cached_path = st.session_state.get("_orig_gif")
        if cached_path and os.path.exists(cached_path):
            original_gif_path = cached_path
            processed_gif_path = st.session_state["_proc_gif"]
            processed_video_path = st.session_state.get("_proc_mp4")
        elif masks is not None:
            try:
                original_rgb = [np.stack([f] * 3, axis=-1).astype(np.uint8) for f in frames_gray]
                processed_frames = [render_overlay(f, m)[0] for f, m in zip(frames_gray, masks)]

                original_gif_path = _create_gif(original_rgb)
                processed_gif_path = _create_gif(processed_frames)

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vid_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                out = cv2.VideoWriter(vid_path, fourcc, 30, (FRAME_SIZE_SEG, FRAME_SIZE_SEG))
                if out.isOpened():
                    for f in processed_frames:
                        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                    out.release()
                    processed_video_path = vid_path
                else:
                    out.release()
                    try:
                        os.unlink(vid_path)
                    except Exception:
                        pass

                st.session_state["_orig_gif"] = original_gif_path
                st.session_state["_proc_gif"] = processed_gif_path
                st.session_state["_proc_mp4"] = processed_video_path
            except Exception as e:
                st.error(f"U-Net video processing: {e}")

        try:
            expert = expert_for_upload(uploaded_file.name, T)
        except Exception as e:
            expert = None
            st.caption(f"Expert labels unavailable: {e}")

    # ============================
    # VIDEO COMPARISON SECTION
    # ============================
    st.markdown("---")
    st.header("Echocardiogram Video Analysis")
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.markdown("<h3 style='text-align:center;'>Original Video</h3>", unsafe_allow_html=True)
        _show_gif(original_gif_path, st)

    with col_v2:
        st.markdown("<h3 style='text-align:center;'>Segmentation Overlay</h3>", unsafe_allow_html=True)
        _show_gif(processed_gif_path, st)

    # Download button centered
    if processed_video_path and os.path.exists(processed_video_path):
        st.markdown("<div style='padding-top:1.5rem;'>&nbsp;</div>", unsafe_allow_html=True)
        _, col_center, _ = st.columns([1, 2, 1])
        with col_center:
            st.download_button(
                "Download Processed Video (MP4)",
                data=open(processed_video_path, "rb").read(),
                file_name="segmentation_overlay.mp4",
                mime="video/mp4",
                width='stretch',
            )
    st.markdown("---")

    # ============================
    # EF PREDICTION & CLASSIFICATION SECTION
    # ============================
    st.header("Ejection Fraction Prediction")

    if ef_predicted is not None:
        cls = get_ef_classification(ef_predicted)

        col_ef1, col_ef2 = st.columns([1, 2])

        with col_ef1:
            st.markdown(
                f"""
                <div style="
                    background: {cls['bg']};
                    border-radius: 16px;
                    padding: 32px 24px;
                    text-align: center;
                    border: 2px solid {cls['color']};
                    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                ">
                    <div style="font-size: 14px; color: #666; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 1px;">
                        Left Ventricular Ejection Fraction
                    </div>
                    <div style="font-size: 72px; font-weight: 800; color: {cls['color']}; line-height: 1.1;">
                        {ef_predicted:.1f}%
                    </div>
                    <div style="
                        display: inline-block;
                        margin-top: 12px;
                        background: {cls['color']};
                        color: white;
                        padding: 4px 16px;
                        border-radius: 20px;
                        font-size: 13px;
                        font-weight: 600;
                        letter-spacing: 0.5px;
                    ">
                        {cls['badge']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_ef2:
            st.markdown(
                f"""
                <div style="
                    background: #fafafa;
                    border-radius: 16px;
                    padding: 24px;
                    border: 1px solid #e0e0e0;
                    min-height: 200px;
                ">
                    <div style="font-size: 20px; font-weight: 700; color: {cls['color']}; margin-bottom: 12px;">
                        {cls['label']}
                    </div>
                    <div style="font-size: 14px; color: #444; line-height: 1.6; margin-bottom: 16px;">
                        {cls['description']}
                    </div>
                    <div style="font-size: 13px; color: #666; line-height: 1.5; padding: 12px; background: #fff; border-radius: 8px; border-left: 3px solid {cls['color']};">
                        <strong style="color: {cls['color']};">Treatment Focus:</strong> {cls['treatment']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='padding-top:1rem;'></div>", unsafe_allow_html=True)
    ef_cols = st.columns(3 if expert else 2)
    ef_cols[0].metric("ResNet18 regression", f"{ef_predicted:.1f}%" if ef_predicted is not None else "—")
    ef_cols[1].metric(
        "From segmentation (method of disks)",
        f"{ef_seg['ef']:.1f}%" if ef_seg else "—",
        help="20-disk Simpson's rule on the U-Net mask for every frame; EF is computed for "
             "each detected heartbeat (ED peak to the following ES minimum) and the median "
             "over beats is shown.",
    )
    if expert:
        ef_cols[2].metric(f"Expert (EchoNet FileList, {expert['split']} split)", f"{expert['ef']:.1f}%")

    st.markdown("---")

    # --- Segmentation ---
    st.header("Left Ventricle Segmentation")

    if masks is not None:
        if expert and expert["phase"]:
            st.caption("Expert-traced frames: " + ", ".join(f"{p} = frame {f}" for f, p in sorted(expert["phase"].items())))
        seg_frame = st.slider("Select frame", 0, T - 1, 0)

        frame = frames_gray[seg_frame]
        frame_rgb = np.stack([frame] * 3, axis=-1).astype(np.uint8)
        mask = masks[seg_frame]
        overlay_img, contour_data = render_overlay(frame, mask)
        m = contour_data["measurement"]

        st.markdown("<h3 style='text-align:center;'>U-Net Segmentation Results</h3>", unsafe_allow_html=True)

        _, col3, _, col4, _, col5, _ = st.columns([0.5, 2, 0.5, 2, 0.5, 2, 0.5])
        with col3:
            st.image(frame_rgb, caption="Original Frame", width='stretch')
        with col4:
            st.image(overlay_mask(frame_rgb, mask), caption="Mask Overlay", width='stretch')
        with col5:
            st.image(overlay_img, caption="Long axis, 20 disks & landmarks", width='stretch')

        mcols = st.columns(4)
        if m:
            mcols[0].metric("LV area", f"{m['area'] / S**2:.0f} px²")
            mcols[1].metric("Long axis (apex → annulus)", f"{m['length'] / S:.1f} px")
            mcols[2].metric("Disk volume (single plane)", f"{m['volume'] / S**3:,.0f} px³")
            mcols[3].metric("Mid-cavity width", f"{m['diameters'][len(m['diameters']) // 2] / S:.1f} px")
            st.caption("Measured on the 112×112 EchoNet pixel grid. The videos carry no pixel spacing, "
                       "so values are in pixels; EF is a ratio and needs no calibration.")
        else:
            st.info("No left ventricle segmented on this frame.")

        gt = expert["gt"].get(seg_frame) if expert else None
        if gt is not None:
            both = frame_rgb.copy()
            cv2.drawContours(both, [np.rint(gt["polygon"]).astype(np.int32).reshape(-1, 1, 2)], -1, (0, 255, 0), 1, cv2.LINE_AA)
            if m:
                cv2.drawContours(both, [m["contour"]], -1, (255, 60, 60), 1, cv2.LINE_AA)
            _, cg, _, ct, _ = st.columns([0.5, 2, 0.5, 4.5, 0.5])
            with cg:
                st.image(both, caption=f"{expert['phase'].get(seg_frame, '')} frame: expert (green) vs U-Net (red)", width='stretch')
            with ct:
                rows = {
                    "LV area (px²)": (gt["area"] / S**2, m["area"] / S**2 if m else None),
                    "Long axis (px)": (gt["length"] / S, m["length"] / S if m else None),
                    "Disk volume (px³)": (gt["volume"] / S**3, m["volume"] / S**3 if m else None),
                }
                st.markdown(f"**Against the expert tracing** — Dice {dice(m['region'], gt['region']) if m else 0:.3f}")
                st.table({
                    "measure": list(rows),
                    "expert": [f"{e:,.1f}" for e, _ in rows.values()],
                    "U-Net": [f"{p:,.1f}" if p is not None else "—" for _, p in rows.values()],
                })

        # LV area over the clip
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=areas, mode="lines", name="LV area", line=dict(color="#2563eb", width=2)))
        if ef_seg:
            beats = ef_seg["beats"] or [(ef_seg["ed_frame"], ef_seg["es_frame"], ef_seg["ef"])]
            for pos, name, color, symbol in ((0, "detected ED", "#16a34a", "triangle-down"),
                                             (1, "detected ES", "#ea580c", "triangle-up")):
                fs = [b[pos] for b in beats]
                fig.add_trace(go.Scatter(x=fs, y=areas[fs], mode="markers", name=name,
                                         marker=dict(color=color, size=12, symbol=symbol)))
        for f, p in (expert["phase"].items() if expert else []):
            fig.add_vline(x=f, line_dash="dash", line_color="#16a34a" if p == "ED" else "#ea580c",
                          annotation_text=f"expert {p}", annotation_position="top")
        fig.add_vline(x=seg_frame, line_color="#94a3b8", line_width=1)
        fig.update_layout(title="LV area per frame (U-Net)", xaxis_title="frame", yaxis_title="area (px²)",
                          height=320, margin=dict(l=40, r=20, t=50, b=40), legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, width='stretch')

    # --- Google Gemini AI Section ---
    st.header("AI-Powered Analysis (Gemini)")
    st.markdown(
        "Generate structural contour overlays and clinical assessments using Google's Gemini models."
    )

    model_tier = st.selectbox(
        "Select Gemini Model",
        options=["Gemini 3 Flash", "Gemini 3 Pro"],
        index=0,
        help="Gemini 3 Flash: faster results. Gemini 3 Pro: deeper clinical reasoning.",
    )

    if model_tier == "Gemini 3 Flash":
        target_image_model = "gemini-3.1-flash-image"
        target_opinion_model = "gemini-3.1-flash-lite"
    else:
        target_image_model = "gemini-3-pro-image"
        target_opinion_model = "gemini-3.1-pro-preview"

    if "gemini_image" not in st.session_state:
        st.session_state.gemini_image = None
    if "gemini_opinion_text" not in st.session_state:
        st.session_state.gemini_opinion_text = None

    if st.button("Run AI Analysis", type="primary"):
        with st.spinner(f"Analyzing with {model_tier}..."):
            try:
                st.session_state.gemini_image = generate_image(tmp_path, model_name=target_image_model)
                st.session_state.gemini_opinion_text = generate_medical_opinion(
                    tmp_path, model_name=target_opinion_model
                )
                st.success("Analysis complete!")
            except Exception as e:
                st.error(f"API Error: {e}")

    if st.session_state.gemini_image or st.session_state.gemini_opinion_text:
        col_image, col_gemini = st.columns(2)

        with col_image:
            if st.session_state.gemini_image:
                st.subheader("Contour Overlay")
                st.image(
                    st.session_state.gemini_image,
                    caption=f"AI-generated structural overlay ({model_tier})",
                )
                buf = io.BytesIO()
                st.session_state.gemini_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Processed Image",
                    data=byte_im,
                    file_name="gemini_structural_contours.png",
                    mime="image/png",
                )

        with col_gemini:
            if st.session_state.gemini_opinion_text:
                st.subheader("Gemini Opinion")
                st.markdown("### Clinical Assessment")
                st.write(st.session_state.gemini_opinion_text)
                st.download_button(
                    label="Download Opinion (.txt)",
                    data=st.session_state.gemini_opinion_text.encode("utf-8"),
                    file_name="gemini_clinical_assessment.txt",
                    mime="text/plain",
                )

    Path(tmp_path).unlink(missing_ok=True)

else:
    st.info("Upload an echocardiogram video to start analysis.")
    st.markdown("""
    ### Models:
    - **ResNet18 / ResNet34**: Ejection fraction regression
    - **U-Net**: Left ventricle segmentation with contour key points
    - **Gemini 3 Flash / Pro**: AI-powered structural overlay & clinical assessment

    ### Features:
    - EF prediction with HF classification (HFrEF / HFmrEF / HFpEF)
    - LV segmentation with anatomical key points (apex, basal, mid)
    - AI-generated structural overlays and clinical opinion
    """)
