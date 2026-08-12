"""
AI-Based Body Measurement Web App (v2)
----------------------------------------
Stack: Streamlit (UI) + OpenCV (image handling) + MediaPipe (pose + segmentation)

What's new in v2:
- Optional SIDE-VIEW photo: uses MediaPipe Selfie Segmentation to measure real
  body depth at chest/waist/hip height, instead of guessing depth as a fixed
  percentage of width. This makes circumference estimates noticeably more accurate.
- Height input can be entered in cm OR feet+inches.
- All measurement results can be displayed in cm OR inches.
- Colorful custom UI instead of plain default Streamlit styling.

Run with:  streamlit run body_measurement_app.py
"""

import math
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from PIL import Image

st.set_page_config(page_title="AI Body Measurement", page_icon="📏", layout="wide")

# ============================================================
# COLORFUL UI STYLING
# ============================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(160deg, #1a1033 0%, #2b1055 40%, #4b1f6f 100%);
}
h1, h2, h3, p, label, .stMarkdown, .stCaption {
    color: #f5f0ff !important;
}
.measure-card {
    background: linear-gradient(135deg, #ff6ec4, #7873f5);
    border-radius: 16px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.35);
}
.measure-card.alt {
    background: linear-gradient(135deg, #36d1c4, #5b86e5);
}
.measure-label {
    font-size: 14px;
    font-weight: 600;
    color: #ffffffcc;
    margin-bottom: 4px;
}
.measure-value {
    font-size: 30px;
    font-weight: 800;
    color: #ffffff;
}
.info-box {
    background: rgba(255,255,255,0.08);
    border-left: 4px solid #ff6ec4;
    border-radius: 8px;
    padding: 12px 16px;
    margin-top: 10px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

mp_pose = mp.solutions.pose
mp_selfie = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils

CM_PER_INCH = 2.54


# ============================================================
# UNIT HELPERS
# ============================================================
def ft_in_to_cm(feet, inches):
    return (feet * 12 + inches) * CM_PER_INCH


def cm_to_display(value_cm, unit):
    """Convert a cm value to the chosen display unit."""
    return value_cm / CM_PER_INCH if unit == "inches" else value_cm


def unit_label(unit):
    return "in" if unit == "inches" else "cm"


# ============================================================
# MEASUREMENT HELPERS
# ============================================================
def pixel_distance(p1, p2, image_width, image_height):
    x1, y1 = p1.x * image_width, p1.y * image_height
    x2, y2 = p2.x * image_width, p2.y * image_height
    return math.hypot(x2 - x1, y2 - y1)


def estimate_circumference(width_cm, depth_cm):
    """Ramanujan's ellipse perimeter approximation, given real width and depth."""
    a = width_cm / 2
    b = depth_cm / 2
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def get_pixel_height(landmarks, w, h):
    """Nose-to-ankle pixel distance, scaled up to approximate full standing height."""
    L = mp_pose.PoseLandmark
    nose = landmarks[L.NOSE]
    left_ankle = landmarks[L.LEFT_ANKLE]
    right_ankle = landmarks[L.RIGHT_ANKLE]
    ankle = left_ankle if left_ankle.y > right_ankle.y else right_ankle
    return pixel_distance(nose, ankle, w, h) * 1.12


def measure_mask_width_at_y(mask, y_pixel):
    """Given a boolean silhouette mask, measure the pixel width of the body at row y."""
    if y_pixel < 0 or y_pixel >= mask.shape[0]:
        return None
    row = mask[y_pixel, :]
    xs = np.where(row)[0]
    if len(xs) < 2:
        return None
    return xs[-1] - xs[0]


# ============================================================
# UI - HEADER
# ============================================================
st.title("📏 AI-Based Body Measurement")
st.caption("OpenCV + MediaPipe Pose + Selfie Segmentation + Streamlit")

st.markdown(
    "**Instructions:** Stand straight facing the camera for the front photo, "
    "arms slightly away from your body, full body visible, plain background, good lighting. "
    "For the side photo, turn 90° and stand the same distance from the camera."
)

# ============================================================
# UNIT SELECTION
# ============================================================
col_a, col_b = st.columns(2)
with col_a:
    height_unit = st.radio("Height input unit", ["cm", "feet & inches"], horizontal=True)
with col_b:
    display_unit = st.radio("Show results in", ["cm", "inches"], horizontal=True)

if height_unit == "cm":
    height_cm = st.number_input("Enter your actual height (cm)", min_value=100.0, max_value=220.0, value=165.0, step=0.5)
else:
    c1, c2 = st.columns(2)
    with c1:
        feet = st.number_input("Feet", min_value=3, max_value=7, value=5, step=1)
    with c2:
        inches = st.number_input("Inches", min_value=0.0, max_value=11.9, value=5.0, step=0.5)
    height_cm = ft_in_to_cm(feet, inches)
    st.caption(f"= {height_cm:.1f} cm")

# ============================================================
# PHOTO CAPTURE
# ============================================================
front_col, side_col = st.columns(2)
with front_col:
    st.subheader("Front-facing photo (required)")
    front_file = st.camera_input("Capture front view", key="front")
with side_col:
    st.subheader("Side-view photo (optional, improves accuracy)")
    side_file = st.camera_input("Capture side view", key="side")

# ============================================================
# PROCESS FRONT PHOTO
# ============================================================
if front_file is not None:
    image = Image.open(front_file)
    front_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    fh, fw, _ = front_bgr.shape

    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        front_results = pose.process(cv2.cvtColor(front_bgr, cv2.COLOR_BGR2RGB))

    if not front_results.pose_landmarks:
        st.error("No pose detected in the front photo. Make sure your full body is visible with good lighting.")
    else:
        landmarks = front_results.pose_landmarks.landmark
        L = mp_pose.PoseLandmark

        annotated = front_bgr.copy()
        mp_drawing.draw_landmarks(annotated, front_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Front view - detected landmarks", use_container_width=True)

        # Calibration
        pixel_height = get_pixel_height(landmarks, fw, fh)
        px_to_cm_front = height_cm / pixel_height

        # Key landmarks
        l_shoulder, r_shoulder = landmarks[L.LEFT_SHOULDER], landmarks[L.RIGHT_SHOULDER]
        l_hip, r_hip = landmarks[L.LEFT_HIP], landmarks[L.RIGHT_HIP]
        l_elbow, l_wrist = landmarks[L.LEFT_ELBOW], landmarks[L.LEFT_WRIST]
        l_knee, l_ankle = landmarks[L.LEFT_KNEE], landmarks[L.LEFT_ANKLE]

        # Linear measurements (cm)
        shoulder_width_cm = pixel_distance(l_shoulder, r_shoulder, fw, fh) * px_to_cm_front
        hip_width_cm = pixel_distance(l_hip, r_hip, fw, fh) * px_to_cm_front
        arm_length_cm = (
            pixel_distance(l_shoulder, l_elbow, fw, fh) + pixel_distance(l_elbow, l_wrist, fw, fh)
        ) * px_to_cm_front
        leg_length_cm = (
            pixel_distance(l_hip, l_knee, fw, fh) + pixel_distance(l_knee, l_ankle, fw, fh)
        ) * px_to_cm_front

        # Pixel y-positions for chest/waist/hip level (used to align with side photo)
        chest_y_norm = (l_shoulder.y + r_shoulder.y) / 2
        hip_y_norm = (l_hip.y + r_hip.y) / 2
        waist_y_norm = chest_y_norm + (hip_y_norm - chest_y_norm) * 0.6  # ~60% down torso

        # ------------------------------------------------------
        # Try to get REAL depth from the side photo, if provided
        # ------------------------------------------------------
        chest_depth_cm = waist_depth_cm = hip_depth_cm = None

        if side_file is not None:
            side_image = Image.open(side_file)
            side_bgr = cv2.cvtColor(np.array(side_image), cv2.COLOR_RGB2BGR)
            sh, sw, _ = side_bgr.shape

            with mp_pose.Pose(static_image_mode=True) as pose_side:
                side_pose_results = pose_side.process(cv2.cvtColor(side_bgr, cv2.COLOR_BGR2RGB))

            if side_pose_results.pose_landmarks:
                side_landmarks = side_pose_results.pose_landmarks.landmark
                pixel_height_side = get_pixel_height(side_landmarks, sw, sh)
                px_to_cm_side = height_cm / pixel_height_side

                with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
                    seg_results = seg.process(cv2.cvtColor(side_bgr, cv2.COLOR_BGR2RGB))
                mask = seg_results.segmentation_mask > 0.5

                chest_y_px = int(chest_y_norm * sh)
                waist_y_px = int(waist_y_norm * sh)
                hip_y_px = int(hip_y_norm * sh)

                chest_px = measure_mask_width_at_y(mask, chest_y_px)
                waist_px = measure_mask_width_at_y(mask, waist_y_px)
                hip_px = measure_mask_width_at_y(mask, hip_y_px)

                if chest_px: chest_depth_cm = chest_px * px_to_cm_side
                if waist_px: waist_depth_cm = waist_px * px_to_cm_side
                if hip_px: hip_depth_cm = hip_px * px_to_cm_side

                st.image(cv2.cvtColor(side_bgr, cv2.COLOR_BGR2RGB), caption="Side view captured", use_container_width=True, channels="RGB")
            else:
                st.warning("No pose detected in the side photo — falling back to estimated depth for circumference.")

        # ------------------------------------------------------
        # Circumference: use real depth if available, else estimate
        # ------------------------------------------------------
        used_real_depth = all([chest_depth_cm, waist_depth_cm, hip_depth_cm])

        if used_real_depth:
            chest_circumference = estimate_circumference(shoulder_width_cm, chest_depth_cm)
            waist_circumference = estimate_circumference(hip_width_cm * 0.9, waist_depth_cm)
            hip_circumference = estimate_circumference(hip_width_cm, hip_depth_cm)
        else:
            chest_circumference = estimate_circumference(shoulder_width_cm, shoulder_width_cm * 0.55)
            waist_circumference = estimate_circumference(hip_width_cm * 0.9, hip_width_cm * 0.9 * 0.65)
            hip_circumference = estimate_circumference(hip_width_cm, hip_width_cm * 0.70)

        # ------------------------------------------------------
        # Display results (colorful cards)
        # ------------------------------------------------------
        st.subheader("📐 Estimated Measurements")
        u = display_unit
        label = unit_label(u)

        results = [
            ("Shoulder Width", shoulder_width_cm, ""),
            ("Arm Length", arm_length_cm, ""),
            ("Leg Length", leg_length_cm, ""),
            ("Chest (est.)", chest_circumference, "alt"),
            ("Waist (est.)", waist_circumference, "alt"),
            ("Hip (est.)", hip_circumference, "alt"),
        ]

        col1, col2 = st.columns(2)
        for i, (name, value_cm, style) in enumerate(results):
            value = cm_to_display(value_cm, u)
            card_class = "measure-card alt" if style == "alt" else "measure-card"
            target_col = col1 if i % 2 == 0 else col2
            with target_col:
                st.markdown(
                    f"""<div class="{card_class}">
                        <div class="measure-label">{name}</div>
                        <div class="measure-value">{value:.1f} {label}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        if used_real_depth:
            st.markdown(
                '<div class="info-box">✅ Chest, waist, and hip figures use <b>real depth measured from your side photo</b> — much more accurate than a guessed ratio.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="info-box">ℹ️ No usable side photo was provided, so chest/waist/hip are '
                'estimated using an assumed depth ratio. Add a side-view photo above for more accurate results.</div>',
                unsafe_allow_html=True,
            )
else:
    st.info("Capture a front-facing photo above to get started.")
