import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import binary_fill_holes
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download
import streamlit as st


HF_REPO_ID = "terminator1140/steel-defect-unetpp"
MODEL_FILENAME = "unetpp_effb3_scse_best.pt"

IMG_HEIGHT, IMG_WIDTH = 128, 800
DEFECT_CLASSES = ["defect_a", "defect_b", "defect_c", "defect_d"]
CLASS_COLORS = {
    "defect_a": (255, 110, 30),
    "defect_b": (255, 40, 150),
    "defect_c": (130, 110, 255),
    "defect_d": (40, 210, 140),
}
MIN_AREAS = {"defect_a": 60, "defect_b": 40, "defect_c": 200, "defect_d": 180}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

clahe_iso = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_aniso = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 16))

st.set_page_config(page_title="SurfaceGuard AI - Steel Defect Segmentation", layout="wide")


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILENAME):
        weights_path = MODEL_FILENAME
    else:
        weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)

    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b3",
        encoder_weights=None,
        decoder_attention_type="scse",
        in_channels=3,
        classes=4,
    )
    try:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_path, map_location="cpu")

    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_3ch_input(img_rgb):
    if img_rgb.ndim == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb.copy()

    if gray.shape != (IMG_HEIGHT, IMG_WIDTH):
        gray = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)

    ch_clahe = clahe_iso.apply(gray)
    ch_fine = clahe_aniso.apply(gray)
    grad_x = cv2.convertScaleAbs(cv2.Sobel(ch_clahe, cv2.CV_16S, 1, 0, ksize=3))
    grad_y = cv2.convertScaleAbs(cv2.Sobel(ch_clahe, cv2.CV_16S, 0, 1, ksize=3))
    abs_grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
    ch_texture = cv2.addWeighted(ch_fine, 0.75, abs_grad, 0.25, 0)
    return gray, np.stack([gray, ch_clahe, ch_texture], axis=-1)


def predict_defects(model, img_rgb, threshold=0.40, use_tta=True):
    gray, img_3ch = build_3ch_input(img_rgb)
    border_mask = gray <= 5

    img_norm = (img_3ch.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0)

    with torch.inference_mode():
        p0 = torch.sigmoid(torch.clamp(model(x), -15.0, 15.0))
        if use_tta:
            p1 = torch.flip(torch.sigmoid(torch.clamp(model(torch.flip(x, [-1])), -15.0, 15.0)), [-1])
            p2 = torch.flip(torch.sigmoid(torch.clamp(model(torch.flip(x, [-2])), -15.0, 15.0)), [-2])
            probs = ((p0 + p1 + p2) / 3.0)[0].numpy()
        else:
            probs = p0[0].numpy()

    base_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    overlay = base_rgb.copy()
    summary_rows = []

    for c_idx, d_name in enumerate(DEFECT_CLASSES):
        prob_map = probs[c_idx]
        prob_map[border_mask] = 0.0
        peak_conf = float(prob_map.max())

        raw_mask = (prob_map >= threshold).astype(np.uint8)
        if raw_mask.sum() >= MIN_AREAS[d_name]:
            filled = binary_fill_holes(raw_mask > 0).astype(np.uint8)
            mask = filled if (filled.sum() - raw_mask.sum()) <= 120 else raw_mask
        else:
            mask = np.zeros_like(raw_mask)

        area_px = int(mask.sum())
        if area_px >= MIN_AREAS[d_name]:
            color = CLASS_COLORS[d_name]
            overlay[mask > 0] = color
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(base_rgb, contours, -1, color, thickness=2)
            status = "Detected"
        else:
            area_px = 0
            status = "Clean"

        summary_rows.append({
            "Defect Class": d_name.upper(),
            "Status": status,
            "Mask Area (px)": area_px,
            "Peak Confidence": round(peak_conf, 3),
        })

    blended = cv2.addWeighted(base_rgb, 0.72, overlay, 0.28, 0)
    return base_rgb, blended, pd.DataFrame(summary_rows)


st.title("Industrial Steel Surface Defect Segmentation")
st.write(
    "Detect and segment 4 defect classes on `128 x 800` steel strips using **U-Net++ (EfficientNet-B3 + SCSE)** "
    "with 3-channel CLAHE-Sobel preprocessing."
)
st.caption("Color Legend: Orange = DEFECT_A | Pink = DEFECT_B | Purple = DEFECT_C | Green = DEFECT_D")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Probability Threshold", 0.15, 0.85, 0.40, 0.05)
    use_tta = st.checkbox("Enable Test-Time Augmentation (TTA)", value=True)
    input_mode = st.radio("Choose Input Source", ["Use Sample Strip", "Upload Image"])

img_rgb = None
if input_mode == "Use Sample Strip":
    samples = [f for f in ["sample_1.jpg", "sample_2.jpg", "sample_3.jpg"] if os.path.exists(f)]
    if samples:
        chosen = st.selectbox("Select a Sample Steel Strip", samples)
        img_rgb = np.array(Image.open(chosen).convert("RGB"))
    else:
        st.warning("No sample images found in repository. Switch to 'Upload Image'.")
else:
    uploaded = st.file_uploader("Upload a Steel Strip Image (.jpg / .png)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        img_rgb = np.array(Image.open(uploaded).convert("RGB"))

if img_rgb is not None:
    with st.spinner("Loading U-Net++ model and running segmentation..."):
        model = load_model()
        orig_view, overlay_view, summary_df = predict_defects(model, img_rgb, threshold, use_tta)

    st.subheader("Original Steel Strip")
    st.image(orig_view, width="stretch")

    st.subheader("Segmented Defect Overlay")
    st.image(overlay_view, width="stretch")

    st.subheader("Per-Class Detection Summary")
    st.dataframe(summary_df, width="stretch", hide_index=True)
