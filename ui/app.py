import sys
import io
import base64
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
import torch
import cv2
import tempfile
import os

from PIL import Image

from src.config import MODELS_DIR, FRAME_SIZE, FRAME_SIZE_SEG, DEVICE
from src.models.regression import EchoResNet
from src.models.segmentation import UNet
from src.preprocessing.extract_frames import extract_single_video
from src.preprocessing.normalize import normalize_frames, normalize_image
from src.postprocessing.contours import process_mask, get_lv_contour, get_key_points, draw_contour_and_points
from src.visualization.overlay import overlay_mask
from src.models.gemini3_analyzer import generate_image, generate_medical_opinion

st.set_page_config(page_title="EchoNet-Dynamic Analysis", layout="wide")

st.markdown("""
<style>
    .main .block-container {
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        max-width: 1400px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Echocardiogram Analysis — Model Comparison")
st.markdown(
    "Upload an echocardiogram video and compare predictions from ResNet18, ResNet34, and U-Net."
)
st.caption(
    "⚠️ **Demo mode**: models trained on a small subset of EchoNet-Dynamic "
    "for demonstration purposes only. Not for clinical use."
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_regression(backbone: str):
    model = EchoResNet(backbone=backbone).to(device)
    ckpt = MODELS_DIR / f"{backbone}_ef.pth"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


@st.cache_resource
def load_unet():
    model = UNet().to(device)
    ckpt = MODELS_DIR / "unet_lv.pth"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def process_frame_with_unet(frame_gray, model_unet, device):
    """Process a single grayscale frame through U-Net and return overlay with contours."""
    frame = cv2.resize(frame_gray, (FRAME_SIZE_SEG, FRAME_SIZE_SEG))
    frame_3c = np.stack([frame.astype(np.float32)] * 3, axis=-1)
    frame_norm = normalize_image(frame_3c.copy())
    frame_t = torch.from_numpy(frame_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model_unet(frame_t)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        mask = (probs > 0.8).astype(np.uint8)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest).astype(np.uint8)

    frame_rgb = np.stack([frame] * 3, axis=-1).astype(np.uint8)

    overlay = overlay_mask(frame_rgb, mask, color=(0, 255, 0), alpha=0.4)

    contour, area = get_lv_contour(mask)
    keypoints = get_key_points(contour)
    overlay = draw_contour_and_points(overlay, contour, keypoints)

    return overlay  # (H, W, 3) uint8


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


uploaded_file = st.file_uploader("Choose an AVI video", type=["avi", "mp4", "mpeg"])

if uploaded_file is not None:
    ext = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Extracting frames ..."):
        frames = extract_single_video(tmp_path, target_size=FRAME_SIZE)
        frames_gray = frames  # (T, H, W)

    T = frames_gray.shape[0]
    idxs = np.linspace(0, T - 1, 32, dtype=int)
    sampled = frames_gray[idxs]
    sampled = normalize_frames(sampled)
    sampled = np.stack([sampled, sampled, sampled], axis=-1)
    video_t = torch.from_numpy(sampled).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)

    model_unet = load_unet()

    original_gif_path = None
    processed_gif_path = None
    processed_video_path = None
    ef_predicted = None

    upload_id = f"{uploaded_file.name}_{uploaded_file.size}"
    need_processing = st.session_state.get("_upload_id") != upload_id

    with st.spinner("Loading analysis ..."):
        # --- EF Prediction ---
        try:
            model18 = load_regression("resnet18")
            with torch.no_grad():
                ef_val = model18(video_t).item()
                ef_val = max(0.0, min(1.0, ef_val))
                ef_predicted = ef_val * 100
        except Exception as e:
            st.error(f"ResNet18: {e}")

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

        # --- Process all frames through U-Net (cached per file) ---
        cached_path = st.session_state.get("_orig_gif")
        if cached_path and os.path.exists(cached_path):
            original_gif_path = cached_path
            processed_gif_path = st.session_state["_proc_gif"]
            processed_video_path = st.session_state.get("_proc_mp4")
        else:
            try:
                original_rgb = []
                processed_frames = []
                for i in range(T):
                    frame_gray_i = cv2.resize(frames_gray[i], (FRAME_SIZE_SEG, FRAME_SIZE_SEG))
                    frame_rgb_i = np.stack([frame_gray_i] * 3, axis=-1).astype(np.uint8)
                    original_rgb.append(frame_rgb_i)
                    overlay = process_frame_with_unet(frame_gray_i, model_unet, device)
                    processed_frames.append(overlay)

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

        # --- Pre-compute default frame (frame 0) segmentation for instant display ---
        try:
            seg_frame_0 = cv2.resize(frames_gray[0], (FRAME_SIZE_SEG, FRAME_SIZE_SEG))
            frame_rgb_0 = np.stack([seg_frame_0] * 3, axis=-1).astype(np.uint8)
            frame_input_0 = np.stack([seg_frame_0.astype(np.float32)] * 3, axis=-1)
            frame_input_0 = normalize_image(frame_input_0)
            frame_t_0 = torch.from_numpy(frame_input_0).permute(2, 0, 1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                logits = model_unet(frame_t_0)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                mask_0 = (probs > 0.8).astype(np.uint8)
                num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_0)
                if num > 1:
                    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    mask_0 = (labels == largest).astype(np.uint8)
            contour_data_0 = process_mask(mask_0, frame_rgb_0)
            st.session_state["_def_seg_mask"] = mask_0
            st.session_state["_def_seg_contour"] = contour_data_0
            st.session_state["_def_seg_frame_rgb"] = frame_rgb_0
        except Exception as e:
            st.error(f"Default frame pre-computation: {e}")

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

    st.markdown("---")

    # --- Segmentation ---
    st.header("Left Ventricle Segmentation")
    seg_frame = st.slider("Select frame", 0, frames_gray.shape[0] - 1, 0)

    frame = cv2.resize(
        frames_gray[seg_frame],
        (FRAME_SIZE_SEG, FRAME_SIZE_SEG),
    )
    frame_rgb = np.stack([frame] * 3, axis=-1).astype(np.uint8)

    try:
        if seg_frame == 0 and "_def_seg_mask" in st.session_state:
            mask = st.session_state["_def_seg_mask"]
            contour_data = st.session_state["_def_seg_contour"]
            frame_rgb = st.session_state["_def_seg_frame_rgb"]
        else:
            frame_input = np.stack([frame.astype(np.float32)] * 3, axis=-1)
            frame_input = normalize_image(frame_input)
            frame_t = torch.from_numpy(frame_input).permute(2, 0, 1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                logits = model_unet(frame_t)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                mask = (probs > 0.8).astype(np.uint8)
                num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
                if num > 1:
                    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    mask = (labels == largest).astype(np.uint8)
            contour_data = process_mask(mask, frame_rgb)

        st.markdown("<h3 style='text-align:center;'>U-Net Segmentation Results</h3>", unsafe_allow_html=True)

        _, col3, _, col4, _, col5, _ = st.columns([0.5, 2, 0.5, 2, 0.5, 2, 0.5])
        with col3:
            st.image(frame_rgb, caption="Original Frame", channels="GRAY", width='stretch')
        with col4:
            overlay_img = overlay_mask(frame_rgb, mask)
            st.image(overlay_img, caption="Mask Overlay", width='stretch')
        with col5:
            if contour_data["overlay"] is not None:
                st.image(contour_data["overlay"], caption="Contours & Key Points", width='stretch')

        if contour_data["area_px"] is not None:
            st.info(f"LV Area: {contour_data['area_px']:.0f} px²")
        if contour_data["keypoints"]:
            kp = contour_data["keypoints"]
            st.json({
                "centroid": [round(v, 1) for v in kp["centroid"]] if kp["centroid"] else None,
                "apex": [round(v, 1) for v in kp["apex"]] if kp["apex"] else None,
                "basal_septal": [round(v, 1) for v in kp["basal_septal"]] if kp["basal_septal"] else None,
                "basal_lateral": [round(v, 1) for v in kp["basal_lateral"]] if kp["basal_lateral"] else None,
                "mid_septal": [round(v, 1) for v in kp["mid_septal"]] if kp["mid_septal"] else None,
                "mid_lateral": [round(v, 1) for v in kp["mid_lateral"]] if kp["mid_lateral"] else None,
            })

    except Exception as e:
        st.error(f"U-Net: {e}")

    # --- Google Gemini AI Section ---
    st.header("Google Gemini AI Analysis")
    st.markdown(
        "Use Google's generative multimodal APIs to compute semantic structural tracking and evaluate clinical presentations."
    )

    model_tier = st.selectbox(
        "Select Gemini Model Suite Configuration",
        options=["Gemini 3 Flash", "Gemini 3 Pro"],
        index=0,
        help="Toggle between high-speed performance (Flash) and heavy clinical reasoning capabilities (Pro).",
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

    if st.button("Run Comprehensive AI Analysis", type="primary"):
        with st.spinner(f"Processing framework utilizing {model_tier}..."):
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
                    caption=f"Generated Frame Contours ({model_tier}) \n Image may not be precise",
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
    st.info("Upload an echocardiogram video to see predictions.")
    st.markdown("""
    ### Models:
    - **ResNet18**: EF regression (trained on 12 videos)
    - **U-Net**: Left ventricle segmentation (trained on 12 videos)

    ### Metrics:
    - EF prediction: MAE, RMSE
    - Segmentation: Dice coefficient, IoU
    """)
