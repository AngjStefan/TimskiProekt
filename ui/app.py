import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import numpy as np
import torch
import cv2
import tempfile

from src.config import MODELS_DIR, FRAME_SIZE, FRAME_SIZE_SEG, DEVICE
from src.models.regression import EchoResNet
from src.models.segmentation import UNet
from src.preprocessing.extract_frames import extract_single_video
from src.preprocessing.normalize import normalize_frames
from src.postprocessing.contours import process_mask
from src.visualization.overlay import overlay_mask

st.set_page_config(page_title="EchoNet-Dynamic Analysis", layout="wide")
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

uploaded_file = st.file_uploader("Choose an AVI video", type=["avi", "mp4", "mpeg"])

if uploaded_file is not None:
    ext = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Extracting frames ..."):
        frames = extract_single_video(tmp_path, target_size=FRAME_SIZE)
        frames_gray = frames  # (T, H, W)

    # --- Regression predictions ---
    st.header("Ejection Fraction Prediction")
    col1, col2 = st.columns(2)

    # Prepare video tensor once
    T = frames_gray.shape[0]
    idxs = np.linspace(0, T - 1, 32, dtype=int)
    sampled = frames_gray[idxs]
    sampled = normalize_frames(sampled)
    sampled = np.stack([sampled, sampled, sampled], axis=-1)  # (T, H, W, 3)
    video_t = torch.from_numpy(sampled).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)

    with col1:
        st.subheader("ResNet18")
        try:
            model18 = load_regression("resnet18")
            with torch.no_grad():
                ef18 = model18(video_t).item() * 100
            st.metric("EF (%)", f"{ef18:.1f}")
        except Exception as e:
            st.error(f"ResNet18: {e}")

    with col2:
        st.subheader("U-Net (LV Area)")
        try:
            model34 = load_regression("resnet18")
            st.metric("EF via U-Net", "see below")
        except Exception:
            st.info("Segmentation below")

    # --- Segmentation ---
    st.header("Left Ventricle Segmentation")
    seg_frame = st.slider("Select frame", 0, frames_gray.shape[0] - 1, 0)

    frame = frames_gray[seg_frame]
    frame_rgb = np.stack([frame] * 3, axis=-1).astype(np.uint8)
    frame_input = normalize_frames(frame)
    frame_input = np.stack([frame_input] * 3, axis=-1)
    frame_t = torch.from_numpy(frame_input).permute(2, 0, 1).unsqueeze(0).float().to(device)

    try:
        model_unet = load_unet()
        with torch.no_grad():
            logits = model_unet(frame_t)
            mask = torch.sigmoid(logits).squeeze().cpu().numpy()

        st.subheader("U-Net Segmentation Results")

        # Post-process with contours
        mask_resized = cv2.resize(mask, (FRAME_SIZE, FRAME_SIZE),
                                  interpolation=cv2.INTER_NEAREST)
        contour_data = process_mask(mask_resized, frame_rgb)

        col3, col4, col5 = st.columns(3)
        with col3:
            st.image(frame_rgb, caption="Original Frame", channels="GRAY", width=300)
        with col4:
            overlay_img = overlay_mask(frame_rgb, mask_resized)
            st.image(overlay_img, caption="Mask Overlay", width=300)
        with col5:
            if contour_data["overlay"] is not None:
                st.image(contour_data["overlay"], caption="Contours & Key Points", width=300)

        if contour_data["area_px"] is not None:
            st.info(f"LV Area: {contour_data['area_px']:.0f} px²")
        if contour_data["keypoints"]:
            kp = contour_data["keypoints"]
            st.json({
                "centroid": [round(v, 1) for v in kp["centroid"]] if kp["centroid"] else None,
                "apex": [round(v, 1) for v in kp["apex"]] if kp["apex"] else None,
                "basal_left": [round(v, 1) for v in kp["basal_left"]] if kp["basal_left"] else None,
                "basal_right": [round(v, 1) for v in kp["basal_right"]] if kp["basal_right"] else None,
            })

    except Exception as e:
        st.error(f"U-Net: {e}")

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
