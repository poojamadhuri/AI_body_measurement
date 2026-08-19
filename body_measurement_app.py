"""
AI-Based Body Measurement Web App (v6)
----------------------------------------
Stack: Streamlit (UI) + OpenCV (image handling) + MediaPipe (pose + segmentation)

WHAT'S NEW IN v6 (added on top of v5 - nothing below was removed or rewritten):
1. CROP STEP FOR EVERY PHOTO
   After each photo is captured (front/side/back), you now get a trim-with-preview
   step (top/bottom/left/right sliders) before it's confirmed, so background above
   your head, below your feet, or off to the sides never enters the measurement
   calculation - only your body region is analyzed.
2. BACK-VIEW PHOTO (new, optional)
   A third capture slot for a back-facing photo. It's analyzed the same way as a
   front photo (landmarks + framing checks) and its shoulder/hip width readings
   are folded into the multi-shot average for extra consistency.
3. SIDE (AND BACK) PHOTOS NOW SHOW DETECTED LANDMARKS + FRAMING WARNINGS
   Previously only the front photo(s) drew pose landmarks and ran the framing
   checks (tilt / distance / centering / cut-off head-feet). The side photo was
   measured "blind" - no landmarks were ever drawn on it and validate_framing()
   was never called on it, so a badly-framed side photo could silently produce a
   bad chest/waist/hip depth with no warning. v6 runs the same landmark drawing
   and framing validation on the side and back photos that the front photo already
   had, which directly improves the reliability of the depth-based measurements.
4. STANDING DISTANCE UPDATED TO 10 FEET (~3 m)
   Instructions and captions now say 10 feet, matching the reference posture
   guide, instead of the old 2-2.5 m guidance.
5. INSTRUCTIONS SHOWN AS IMAGES
   The capture-instructions and posture-reference graphics are now displayed as
   images at the top of the Measurements page (loaded from the instructions/
   folder shipped next to this file), in addition to the existing text.
6. REORDERED SETTINGS + FEEDBACK SECTION
   "Height input unit" now sits directly above the height entry field instead
   of in a separate row above it. "Show results in" moved from the top of the
   page down to sit right above the "Estimated Measurements" results, since it
   only affects how results are displayed, not the photo-capture flow. A new
   feedback section (star rating + comments) was added at the bottom of the
   Measurements page, kept in-session only (see the honesty note in the code).

WHAT WAS ALREADY IN v5 (unchanged, still true):
1. CUSTOM CAMERA COMPONENT WITH A COUNTDOWN TIMER (replaces st.camera_input)
   st.camera_input has no self-timer and sizes each box independently, which is
   why the front and side boxes rendered at different heights. v5 replaces it
   with a small custom component (see timer_camera_component/index.html) that:
     - renders every camera box at the exact same fixed size (fixes the
       mismatched-height issue directly, rather than patching it with CSS)
     - shows a live preview, then a 3-second countdown for front photos and a
       5-second countdown for the side photo before auto-capturing, so there's
       time to step back into position after pressing the button
   This file must be deployed together with the timer_camera_component/ folder
   sitting next to it - see the deployment note further down.

2. REAL, WORKING EMAIL OTP (previously demo-only)
   If you add your own email credentials to Streamlit secrets, "Send OTP" now
   sends an actual email with the code via SMTP, instead of just printing the
   code on screen. If no credentials are configured, it automatically falls
   back to the old demo-mode banner - nothing breaks if you skip this setup.
   Phone/SMS OTP is wired up the same way for Twilio, but stays demo-mode by
   default since a free, no-signup SMS gateway doesn't exist - see the setup
   notes near send_otp_sms_real() below for exactly what's needed to turn it on.

WHY v4 EXISTED (still true, unchanged in v5):
v3 worked, but measurements changed noticeably every time a photo was re-uploaded,
even for the same person standing roughly the same way. The cause: v3 calibrated
EVERY measurement off a single number - nose-to-ankle pixel distance x a guessed
1.12 fudge factor. That number is unstable (head tilt, chin angle, slightly
turning towards the camera all shift where the "nose" landmark falls), and since
every other measurement is scaled off it, any wobble in that one number wobbles
ALL the results together. That's exactly the "size changes every time" symptom.

WHAT'S NEW / FIXED IN v4 (nothing removed - only added/replaced):
1. SILHOUETTE-BASED HEIGHT CALIBRATION (the main fix)
   Instead of nose-to-ankle x 1.12, we now use MediaPipe Selfie Segmentation to
   find the actual top-of-head to bottom-of-feet pixel span from your real body
   outline. This does not depend on head tilt, facial landmark visibility, or a
   guessed multiplier - it measures your actual silhouette, so it stays stable
   across photos. If segmentation fails for some reason, we fall back to the old
   nose-ankle method and clearly label the result as lower-confidence.

2. PHOTO FRAMING VALIDATION (catches the #2 cause of inconsistency)
   Before calculating anything, the app checks:
     - are your shoulders roughly level (camera tilted / you leaning = skewed widths)
     - do you fill a sensible fraction of the frame (too far = more pixel error,
       too close = cropped head/feet)
     - are you roughly centered horizontally
   If a check fails, you get a specific, actionable warning instead of a silently
   wrong number - this is what actually stops "different distance each time" from
   turning into "different measurements each time".

3. OPTIONAL MULTI-PHOTO AVERAGING (recommended - biggest consistency boost)
   You can now take up to 3 front photos in one session. The app measures each
   one independently, averages the results, and shows you a consistency score
   (how much the shots agreed with each other). If they disagree by more than a
   few percent, it tells you plainly instead of quietly picking one number.

Everything else - login flow, Home page, gender-specific size charts, body-shape
detection, side-photo depth measurement, pastel UI - is unchanged from v3.

Run with:  streamlit run body_measurement_app_v4.py
"""

import io
import os
import math
import random
import base64
import smtplib
import datetime
from email.mime.text import MIMEText

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import mediapipe as mp
from PIL import Image

st.set_page_config(page_title="AI Body Measurement", page_icon="📏", layout="wide")

# ============================================================
# PASTEL UI STYLING (unchanged from v3)
# ============================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(160deg, #fdf2f8 0%, #f3e8ff 45%, #e6fbf5 100%);
}
h1, h2, h3, h4, p, label, .stMarkdown, .stCaption, span {
    color: #4a3f55 !important;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fbe8f7 0%, #ece5ff 100%);
}
.measure-card {
    background: linear-gradient(135deg, #ffd6ec, #d9d6ff);
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 4px 14px rgba(150,120,180,0.18);
}
.measure-card.alt {
    background: linear-gradient(135deg, #c8f7ec, #cfe4ff);
}
.measure-label {
    font-size: 14px;
    font-weight: 700;
    color: #6b5b7a !important;
    margin-bottom: 4px;
}
.measure-value {
    font-size: 28px;
    font-weight: 800;
    color: #3f3550 !important;
}
.info-box {
    background: rgba(255,255,255,0.55);
    border-left: 5px solid #f5a9d0;
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 10px;
    font-size: 14px;
}
.warn-box {
    background: rgba(255,244,214,0.75);
    border-left: 5px solid #e8a93e;
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 10px;
    font-size: 14px;
}
.good-box {
    background: rgba(220,247,236,0.75);
    border-left: 5px solid #3ec98e;
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 10px;
    font-size: 14px;
}
.feature-card {
    background: rgba(255,255,255,0.55);
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 12px;
    box-shadow: 0 3px 10px rgba(150,120,180,0.12);
}
.pill {
    display: inline-block;
    background: linear-gradient(135deg, #f5a9d0, #b9a7f5);
    color: white !important;
    padding: 6px 16px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 15px;
}
.stButton>button {
    background: linear-gradient(135deg, #f5a9d0, #b9a7f5);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 8px 20px;
}
</style>
""", unsafe_allow_html=True)

mp_pose = mp.solutions.pose
mp_selfie = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils

# ============================================================
# NEW IN v5: CUSTOM TIMER-CAMERA COMPONENT
# ============================================================
# DEPLOYMENT NOTE: this line loads timer_camera_component/index.html from disk,
# so that folder must sit in the same directory as this .py file - both locally
# and in whatever you push to GitHub for Streamlit Community Cloud. If the
# folder is missing, this line will raise an error on startup (not silently
# fall back), since the app has no working camera input without it.
_COMPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "timer_camera_component")
_timer_camera = components.declare_component("timer_camera", path=_COMPONENT_DIR)


def timer_camera_input(label, seconds, key):
    """Renders the custom countdown camera and returns a PIL Image once the
    user has captured a shot, or None before that. This is the drop-in
    replacement for st.camera_input used everywhere below."""
    data_url = _timer_camera(label=label, seconds=seconds, key=key, default=None)
    if not data_url:
        return None
    try:
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None


# ============================================================
# NEW IN v6: INSTRUCTION IMAGES
# ============================================================
# DEPLOYMENT NOTE: same pattern as timer_camera_component/ above - this
# "instructions" folder must sit next to this .py file, both locally and in
# whatever you push to GitHub for Streamlit Community Cloud. If a file is
# missing, we just skip showing that image instead of raising an error, since
# the app is still fully usable without the instruction graphics.
_INSTRUCTIONS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "instructions"
)

POSTURE_REFERENCE_IMG = os.path.join(
    _INSTRUCTIONS_DIR,
    "posture_reference.jpeg"
)


def show_instruction_images():
    """Show only the main front/side posture instruction image."""
    if os.path.exists(POSTURE_REFERENCE_IMG):
        st.image(
            POSTURE_REFERENCE_IMG,
            use_container_width=True
        )
    else:
        st.warning(f"Image not found: {POSTURE_REFERENCE_IMG}")

# ============================================================
# NEW IN v6: CROP STEP FOR EVERY PHOTO
# (superseded below - kept only so nothing that already referenced this
# function breaks; capture_crop_and_confirm() no longer calls it, see the
# automatic-crop replacement further down)
# ============================================================
def crop_controls(image, key_prefix):
    """Shows trim sliders (top/bottom/left/right, as a % of the photo) with a
    live preview, and returns the cropped PIL image. This is what lets extra
    background above the head / below the feet / off the sides be excluded
    before the photo is ever handed to the pose + segmentation pipeline, so
    only the body region is analyzed."""
    w, h = image.size
    st.caption(
        "🔲 Adjust the crop so only your body (head to feet) is inside the box - "
        "trim away extra background, then confirm below."
    )
    c1, c2 = st.columns(2)
    with c1:
        top_pct = st.slider("Trim top (%)", 0, 70, 0, step=1, key=f"{key_prefix}_crop_top")
        bottom_pct = st.slider("Trim bottom (%)", 0, 70, 0, step=1, key=f"{key_prefix}_crop_bottom")
    with c2:
        left_pct = st.slider("Trim left (%)", 0, 70, 0, step=1, key=f"{key_prefix}_crop_left")
        right_pct = st.slider("Trim right (%)", 0, 70, 0, step=1, key=f"{key_prefix}_crop_right")

    left = int(w * left_pct / 100)
    right = int(w * (1 - right_pct / 100))
    top = int(h * top_pct / 100)
    bottom = int(h * (1 - bottom_pct / 100))
    if right <= left:
        right = left + 1
    if bottom <= top:
        bottom = top + 1

    cropped = image.crop((left, top, right, bottom))
    st.image(cropped, caption="Crop preview - this is exactly what gets measured", use_container_width=True)
    return cropped


# ============================================================
# NEW: AUTOMATIC CROP (replaces the manual trim-slider step above)
# ============================================================
def auto_crop_to_silhouette(image, margin_frac=0.06):
    """Automatically crops a just-captured photo to the detected body
    silhouette, instead of asking the user to drag trim sliders. Runs selfie
    segmentation, finds the bounding box of the person's mask, and crops to
    that box with a small margin so head/feet aren't clipped. Falls back to
    the original, uncropped image if a usable silhouette can't be found, so a
    failed detection never blocks the capture flow."""
    try:
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
            seg_results = seg.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        mask = seg_results.segmentation_mask > 0.5
        ys, xs = np.where(mask)
        if len(ys) < 2 or len(xs) < 2:
            return image
        top, bottom = int(ys.min()), int(ys.max())
        left, right = int(xs.min()), int(xs.max())
        my = int((bottom - top) * margin_frac)
        mx = int((right - left) * margin_frac)
        top = max(0, top - my)
        bottom = min(h, bottom + my)
        left = max(0, left - mx)
        right = min(w, right + mx)
        if right <= left or bottom <= top:
            return image
        return image.crop((left, top, right, bottom))
    except Exception:
        return image


# ============================================================
# NEW: PRE-CAPTURE READINESS CHECK
# ============================================================
def check_capture_readiness(image):
    """Runs pose + segmentation on a just-captured (pre-crop) photo and
    returns a plain readiness verdict covering full-body detection, head/feet
    visibility, lighting, distance, and body position. Shown to the user as
    either "✅ Ready to Capture" or a specific list of what to fix - the same
    plain-language issue strings validate_framing() already produces, so the
    Retake-instruction mapping used later in the app works for both."""
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    issues = []

    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        pose_results = pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    if not pose_results.pose_landmarks:
        issues.append(
            "No body detected in the photo. Make sure your full body is visible with good lighting."
        )
        return {"ready": False, "issues": issues}

    landmarks = pose_results.pose_landmarks.landmark
    L = mp_pose.PoseLandmark
    key_indices = [
        L.NOSE, L.LEFT_ANKLE, L.RIGHT_ANKLE, L.LEFT_SHOULDER, L.RIGHT_SHOULDER,
        L.LEFT_HIP, L.RIGHT_HIP, L.LEFT_KNEE, L.RIGHT_KNEE,
    ]
    if not check_visibility(landmarks, key_indices):
        issues.append(
            "Some key body points (shoulders/hips/knees/feet) aren't clearly visible - "
            "make sure your full body is in frame with good lighting."
        )

    with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
        seg_results = seg.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    mask = seg_results.segmentation_mask > 0.5

    # Reuses the same head/feet cut-off, distance, tilt, and centering checks
    # already used after capture, just run a step earlier (right after the
    # shot is taken, before the auto-crop below).
    issues.extend(validate_framing(landmarks, mask, w, h))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    if brightness < 70:
        issues.append("The photo looks quite dark. Move to a brighter, more evenly lit spot.")
    elif brightness > 200:
        issues.append("The photo looks overexposed/too bright. Avoid harsh direct light or backlighting.")

    return {"ready": len(issues) == 0, "issues": issues}


def capture_crop_and_confirm(label, seconds, slot_key, final_state_key):
    """Combines the existing timer_camera_input capture with a pre-capture
    readiness check and an automatic crop, then a confirm button. Returns
    True once a final (cropped) image is stored in
    st.session_state[final_state_key]; returns False while still
    capturing/checking, in which case the caller should stop rendering
    further (same early-return pattern the rest of this page already uses).

    NEW: also offers "Upload from gallery" as an alternate source for this same
    slot, using a plain st.file_uploader. This sits entirely outside the custom
    timer_camera component (no changes to that component needed), so it can't
    break the existing camera capture path - it just hands the app a PIL image
    the exact same way a camera capture would, before the checks below.

    UPDATED: the old manual trim-slider crop step has been replaced with an
    automatic, silhouette-based crop (auto_crop_to_silhouette), and a
    "✅ Ready to Capture" / "what to fix" readiness check (check_capture_readiness)
    now runs on the raw photo before it's cropped and confirmed."""
    raw_key = f"{slot_key}_raw"
    if st.session_state.get(final_state_key) is not None:
        return True

    if st.session_state.get(raw_key) is None:
        source = st.radio(
            "Photo source",
            ["📷 Use camera", "🖼️ Upload from gallery"],
            horizontal=True,
            key=f"{slot_key}_source",
        )

        if source == "🖼️ Upload from gallery":
            uploaded = st.file_uploader(
                f"{label} — choose a full-body photo",
                type=["jpg", "jpeg", "png"],
                key=f"{slot_key}_uploader",
            )
            if uploaded is not None:
                try:
                    img = Image.open(uploaded).convert("RGB")
                except Exception:
                    st.error("Couldn't read that image file — try a JPG or PNG.")
                    return False
                st.session_state[raw_key] = img
                st.rerun()
            return False

        captured = timer_camera_input(label, seconds=seconds, key=slot_key)
        if captured is not None:
            st.session_state[raw_key] = captured
            st.rerun()
        return False

    raw_image = st.session_state[raw_key]

    # NEW: readiness check runs first, before any cropping happens.
    readiness = check_capture_readiness(raw_image)
    if readiness["ready"]:
        st.markdown(
            '<div class="good-box">✅ Ready to Capture — full body, head, feet, '
            'lighting, and distance all look good.</div>',
            unsafe_allow_html=True,
        )
    else:
        tags = []
        for issue in readiness["issues"]:
            tag = map_issue_to_retake_instruction(issue)
            if tag not in tags:
                tags.append(tag)
        fix_text = "<br>".join(f"• {t}" for t in tags)
        st.markdown(
            f'<div class="warn-box">⚠️ <b>Fix before continuing:</b><br>{fix_text}</div>',
            unsafe_allow_html=True,
        )

    # NEW: automatic crop replaces the old manual trim sliders.
    cropped = auto_crop_to_silhouette(raw_image)
    st.image(cropped, caption="Auto-cropped preview - this is exactly what gets measured", use_container_width=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        confirm_label = "✅ Confirm photo" if readiness["ready"] else "⚠️ Use anyway"
        if st.button(confirm_label, key=f"{slot_key}_confirm_crop"):
            st.session_state[final_state_key] = cropped
            st.session_state[raw_key] = None
            st.rerun()
    with cc2:
        if st.button("🔁 Retake this photo", key=f"{slot_key}_retake_raw"):
            st.session_state[raw_key] = None
            st.rerun()
    return False


# ============================================================

# NEW IN v6: FEEDBACK SECTION
# ============================================================
def show_feedback_section():
    """Renders a simple star-rating + comments feedback form at the bottom of
    the Measurements page. Feedback is kept in st.session_state for this
    session only - there's no backend/database wired up yet, so it isn't sent
    anywhere or saved between sessions. This mirrors the same honesty pattern
    already used for the notifications toggle on the Home page."""
    st.subheader("💬 Feedback")
    st.caption("Tell us how the measurements and the overall app experience felt.")

    rating = st.select_slider(
        "How accurate did the results feel?",
        options=["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
        value="⭐⭐⭐",
        key="feedback_rating",
    )
    comments = st.text_area(
        "Anything that felt off, or any suggestions?",
        placeholder="e.g. the waist number seemed a little large, or the crop step was handy...",
        key="feedback_comments",
    )

    if st.button("Submit feedback", key="submit_feedback"):
        st.session_state.feedback_log.append({"rating": rating, "comments": comments.strip()})
        st.success("✅ Thanks for the feedback!")

    st.markdown(
        """<div class="info-box">ℹ️ Feedback is only kept for this session right now — no
        backend/database is connected yet, so it isn't saved once you close the app. To make
        this persist, a small database (e.g. SQLite, Google Sheets, or Firebase) would need
        to be wired up.</div>""",
        unsafe_allow_html=True,
    )

    if st.session_state.feedback_log:
        with st.expander(f"This session's feedback ({len(st.session_state.feedback_log)})", expanded=False):
            for i, entry in enumerate(st.session_state.feedback_log, 1):
                st.markdown(f"**{i}. {entry['rating']}** — {entry['comments'] or '_(no comment)_'}")


# ============================================================
# NEW IN v5: REAL OTP SENDING (email fully working, SMS optional)
# ============================================================
def has_email_credentials():
    """True if the app owner has configured real email-sending credentials in
    Streamlit secrets (.streamlit/secrets.toml locally, or the 'Secrets' panel
    on Streamlit Community Cloud). Needed keys:
        EMAIL_ADDRESS       - the Gmail (or other SMTP) address to send FROM
        EMAIL_APP_PASSWORD  - a 16-character Gmail "App Password", NOT your
                               normal Gmail password. Create one at
                               https://myaccount.google.com/apppasswords
                               (requires 2-Step Verification to be on).
    Optional keys: SMTP_SERVER (default smtp.gmail.com), SMTP_PORT (default 587).
    """
    try:
        return "EMAIL_ADDRESS" in st.secrets and "EMAIL_APP_PASSWORD" in st.secrets
    except Exception:
        return False


def send_otp_email_real(to_email, otp):
    """Sends a real OTP email via SMTP. Returns True on success, False on any
    failure (caller falls back to demo-mode display of the code)."""
    try:
        sender = st.secrets["EMAIL_ADDRESS"]
        app_password = st.secrets["EMAIL_APP_PASSWORD"]
        smtp_server = st.secrets.get("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(st.secrets.get("SMTP_PORT", 587))
    except Exception:
        return False

    msg = MIMEText(
        f"Your AI Body Measurement verification code is: {otp}\n\n"
        f"If you didn't request this, you can ignore this email."
    )
    msg["Subject"] = "Your AI Body Measurement verification code"
    msg["From"] = sender
    msg["To"] = to_email

    try:
        with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(sender, app_password)
            server.sendmail(sender, [to_email], msg.as_string())
        return True
    except Exception as e:
        st.session_state["otp_send_error"] = str(e)
        return False


def has_sms_credentials():
    """True if Twilio credentials are configured. See send_otp_sms_real() for
    the exact setup this requires - it's optional and off by default."""
    try:
        return all(
            k in st.secrets
            for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER")
        )
    except Exception:
        return False


def send_otp_sms_real(to_phone, otp):
    """Sends a real OTP SMS via Twilio. OPTIONAL - not required for the app to
    work, and off unless you set it up:
        1. pip install twilio  (and add "twilio" to requirements.txt)
        2. Create a free Twilio trial account at https://www.twilio.com/try-twilio
        3. Add TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER to secrets
        4. Trial accounts can only SEND to phone numbers you've verified in the
           Twilio console - that's a Twilio limitation, not something this code
           can work around. For a real public launch you'd need a paid Twilio
           account. Until any of this is set up, phone sign-in silently stays
           in demo mode - nothing breaks.
    Returns True on success, False on any failure or if not configured."""
    try:
        from twilio.rest import Client  # optional dependency, only needed here
    except ImportError:
        return False
    if not has_sms_credentials():
        return False
    try:
        client = Client(st.secrets["TWILIO_ACCOUNT_SID"], st.secrets["TWILIO_AUTH_TOKEN"])
        client.messages.create(
            body=f"Your AI Body Measurement verification code is: {otp}",
            from_=st.secrets["TWILIO_FROM_NUMBER"],
            to=to_phone,
        )
        return True
    except Exception as e:
        st.session_state["otp_send_error"] = str(e)
        return False


CM_PER_INCH = 2.54
VISIBILITY_THRESHOLD = 0.5

SIZE_LABELS = ["XS", "S", "M", "L", "XL", "XXL", "XXXL"]

WOMEN_BUST_BOUNDS = [84, 90, 96, 104, 112, 120]
WOMEN_WAIST_BOUNDS = [64, 70, 76, 84, 92, 100]
WOMEN_HIP_BOUNDS = [90, 96, 102, 110, 118, 126]

MEN_CHEST_BOUNDS = [91, 97, 103, 111, 119, 127]
MEN_WAIST_BOUNDS = [76, 82, 89, 99, 109, 119]

# ------------------------------------------------------------
# KIDS (Boys / Girls) size chart + ratios
# Age-bracket labels used for kidswear (matches common children's
# clothing charts). Bound tables are the upper cm limit of each
# bracket for the given measurement, same pattern as the adult
# bound tables above (_bound_to_index picks the first bracket the
# measured value fits under).
# ------------------------------------------------------------
KIDS_SIZE_LABELS = ["2-3Y", "4-5Y", "6-7Y", "8-9Y", "10-11Y", "12-13Y", "14-15Y", "16Y+"]

BOYS_CHEST_BOUNDS = [53, 57, 61, 66, 71, 76, 81]
BOYS_WAIST_BOUNDS = [51, 53, 55, 58, 61, 64, 67]

GIRLS_CHEST_BOUNDS = [52, 55, 59, 63, 68, 74, 80]
GIRLS_WAIST_BOUNDS = [50, 52, 54, 57, 60, 63, 66]
GIRLS_HIP_BOUNDS = [53, 57, 61, 66, 72, 79, 86]

# Width->depth ratios used to estimate chest/waist/hip circumference from a
# front-photo width when no side-photo depth is available. Children's torsos
# are proportionally rounder (closer to circular cross-section) than an
# adult's, so these sit higher than the adult ratios below.
RATIO_TABLE = {
    "Women": (0.58, 0.68, 0.72),
    "Men": (0.50, 0.62, 0.68),
    "Boys": (0.60, 0.66, 0.66),
    "Girls": (0.60, 0.66, 0.70),
}

# ------------------------------------------------------------
# CLOTHING ITEMS - what the measurements are being taken for.
# Shown as a purpose selector above the photo-capture section so the
# results/instructions can be framed around the actual garment.
# ------------------------------------------------------------
CLOTHING_ITEMS = {
    "Women": [
        "Blouse", "Chudidhar / Salwar Suit", "Saree Petticoat", "Kurti / Top",
        "Frock / Western Dress", "Skirt", "Lehenga Choli", "Trousers / Pants",
        "Jeans", "Jacket / Coat", "Nightwear / Kaftan",
    ],
    "Men": [
        "Shirt", "T-Shirt", "Kurta Pajama", "Trousers / Pants", "Jeans",
        "Formal Suit / Blazer", "Shorts", "Nightwear",
    ],
    "Boys": [
        "Shirt", "T-Shirt", "Shorts", "Trousers / Pants", "Kurta Pajama",
        "Jacket / Sweater", "Nightwear",
    ],
    "Girls": [
        "Frock", "Chudidhar / Salwar Suit", "Skirt & Top", "Kurti / Top",
        "Lehenga Choli", "Jacket / Sweater", "Nightwear / Gown",
    ],
}

# NEW: not every garment uses a full-length sleeve. Full sleeve (shoulder->
# wrist) and half sleeve (shoulder->elbow) already existed; this adds the two
# missing everyday options - 3/4 sleeve (shoulder-> mid-forearm, common on
# kurtis/tops) and Sleeveless (no sleeve length at all - shoulder width /
# armhole is what matters there instead). The user picks one so the app can
# point out exactly the number relevant to what they're stitching/buying,
# instead of only ever showing the full-arm figure.
SLEEVE_TYPES = ["Full Sleeve", "Half Sleeve", "3/4 Sleeve", "Sleeveless"]


# ============================================================
# SESSION STATE INIT
# ============================================================
def init_state():
    defaults = {
        "logged_in": False,
        "page": "login",
        "language": "en",
        "contact_method": "Phone Number",
        "contact_value": "",
        "generated_otp": None,
        "otp_sent": False,
        "notifications_enabled": True,
        "gender": "Women",
        "kid_type": "Boys",
        "kid_age": 8,
        "clothing_item": None,
        # NEW: which sleeve style the measurements are for - Full / Half /
        # 3/4 / Sleeveless - defaults to Full Sleeve, same as the figure the
        # app already showed before this option existed.
        "sleeve_type": "Full Sleeve",
        "use_multi_shot": False,
        "front_image_1": None,
        "front_image_2": None,
        "front_image_3": None,
        "side_image": None,
        "back_image": None,
        # NEW in v6: staging slots that hold a just-captured photo while the
        # user adjusts the crop, before it's confirmed into the *_image state
        # above. Kept separate so "retake" vs "confirm crop" can't collide.
        "front1_raw": None,
        "front2_raw": None,
        "front3_raw": None,
        "side_raw": None,
        "back_raw": None,
        "use_back_view": False,
        "otp_is_real": False,
        "otp_send_error": None,
        "feedback_log": [],
        # SmartMeasure opening-slide flag (added; existing measurement state is unchanged)
        "smartmeasure_intro_seen": False,
        # NEW: Measurement History - stores only numeric results (height, chest,
        # waist, hip, size, body shape) plus a date/time. Body photos are never
        # written into this list, only the numbers derived from them.
        "measurement_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ============================================================
# LANGUAGE / TRANSLATIONS (English / Telugu)
# ============================================================
# Lightweight translation layer - covers navigation, page titles, section
# headers, buttons, and the core measurement/history labels a user reads
# most often. Anything not in this dict simply falls back to the English
# key itself, so nothing ever renders blank.
TRANSLATIONS = {
    "en": {
        "app_pill": "📏 Body Measurement",
        "language_label": "🌐 Language",
        "welcome": "Welcome 👋",
        "signin_sub": "Sign in to get your accurate, AI-assisted body measurements.",
        "sign_in_with": "Sign in with",
        "phone_number": "Phone Number",
        "email": "Email",
        "send_otp": "Send OTP",
        "enter_otp": "Enter the 4-digit OTP",
        "verify_continue": "Verify & Continue",
        "nav_home": "🏠 Home",
        "nav_measurements": "📐 Measurements",
        "navigate": "Navigate",
        "log_out": "Log out",
        "home_title": "🏠 Home",
        "what_app_can_do": "What this app can do",
        "preferences": "Preferences",
        "profile_setting": "Profile setting",
        "measurement_profile": "Measurement profile",
        "kids_category": "Kids category",
        "measurements_title": "📐 Measurements",
        "who_for": "Who are you taking measurements for?",
        "women": "Women",
        "men": "Men",
        "kids": "Kids",
        "boys": "Boys",
        "girls": "Girls",
        "child_age": "Child's age (years)",
        "height_unit": "Height input unit",
        "enter_height": "Enter your actual height (cm)",
        "enter_child_height": "Enter the child's actual height (cm)",
        "what_for": "what are u taking measurements for?",
        "front_photos": "Front-facing photo(s) (required)",
        "side_photo": "Side-view photo (optional, improves accuracy)",
        "back_photo": "Back-view photo (optional)",
        "estimated_measurements": "📐 Estimated Measurements",
        "show_results_in": "Show results in",
        "shoulder_width": "Shoulder Width",
        "full_sleeve_length": "Full Sleeve Length (Shoulder→Wrist)",
        "half_sleeve_length": "Half Sleeve Length (Shoulder→Elbow)",
        "three_quarter_sleeve_length": "3/4 Sleeve Length (Shoulder→Mid-Forearm)",
        "sleeve_type": "Sleeve type",
        "sleeveless": "Sleeveless",
        "sleeveless_note": "No sleeve length needed for this cut — use the Shoulder Width above as your armhole/shoulder reference instead.",
        "waist_length": "Waist Length (Shoulder→Waist)",
        "leg_length": "Leg Length",
        "bust": "Bust",
        "chest": "Chest",
        "waist_est": "Waist (est.)",
        "hip_est": "Hip (est.)",
        "detected_size_shape": "👗 Detected Size & Shape",
        "estimated_size": "Estimated Size",
        "body_shape": "Body Shape",
        "measurement_history": "🕘 Measurement History",
        "save_to_history": "💾 Save this result to history",
        "compare_entries": "Compare two entries",
        "older_entry": "Older entry",
        "newer_entry": "Newer entry",
        "clear_history": "🗑️ Clear history",
        "show_history_in": "Show history in",
    },
    "te": {
        "app_pill": "📏 శరీర కొలతలు",
        "language_label": "🌐 భాష",
        "welcome": "స్వాగతం 👋",
        "signin_sub": "మీ ఖచ్చితమైన, AI-సహాయక శరీర కొలతలు పొందడానికి సైన్ ఇన్ చేయండి.",
        "sign_in_with": "దీనితో సైన్ ఇన్ చేయండి",
        "phone_number": "ఫోన్ నంబర్",
        "email": "ఇమెయిల్",
        "send_otp": "OTP పంపండి",
        "enter_otp": "4-అంకెల OTP నమోదు చేయండి",
        "verify_continue": "ధృవీకరించి కొనసాగించండి",
        "nav_home": "🏠 హోమ్",
        "nav_measurements": "📐 కొలతలు",
        "navigate": "నావిగేట్ చేయండి",
        "log_out": "లాగ్ అవుట్",
        "home_title": "🏠 హోమ్",
        "what_app_can_do": "ఈ యాప్ ఏమి చేయగలదు",
        "preferences": "ప్రాధాన్యతలు",
        "profile_setting": "ప్రొఫైల్ సెట్టింగ్",
        "measurement_profile": "కొలత ప్రొఫైల్",
        "kids_category": "పిల్లల వర్గం",
        "measurements_title": "📐 కొలతలు",
        "who_for": "మీరు ఎవరి కోసం కొలతలు తీసుకుంటున్నారు?",
        "women": "మహిళలు",
        "men": "పురుషులు",
        "kids": "పిల్లలు",
        "boys": "అబ్బాయిలు",
        "girls": "అమ్మాయిలు",
        "child_age": "పిల్లల వయస్సు (సంవత్సరాలు)",
        "height_unit": "ఎత్తు ఇన్‌పుట్ యూనిట్",
        "enter_height": "మీ నిజమైన ఎత్తును నమోదు చేయండి (సెం.మీ.)",
        "enter_child_height": "పిల్లల నిజమైన ఎత్తును నమోదు చేయండి (సెం.మీ.)",
        "what_for": "మీరు దేని కోసం కొలతలు తీసుకుంటున్నారు?",
        "front_photos": "ముందు వైపు ఫోటో(లు) (అవసరం)",
        "side_photo": "పక్క వైపు ఫోటో (ఐచ్ఛికం, ఖచ్చితత్వాన్ని మెరుగుపరుస్తుంది)",
        "back_photo": "వెనుక వైపు ఫోటో (ఐచ్ఛికం)",
        "estimated_measurements": "📐 అంచనా వేసిన కొలతలు",
        "show_results_in": "ఫలితాలను చూపించండి",
        "shoulder_width": "భుజాల వెడల్పు",
        "full_sleeve_length": "పూర్తి చేతుల పొడవు (భుజం→మణికట్టు)",
        "half_sleeve_length": "అర్ధ చేతుల పొడవు (భుజం→మోచేయి)",
        "three_quarter_sleeve_length": "3/4 చేతుల పొడవు (భుజం→మధ్య ముంజేయి)",
        "sleeve_type": "చేతుల రకం",
        "sleeveless": "చేతులు లేనిది",
        "sleeveless_note": "ఈ కటింగ్‌కు చేతుల పొడవు అవసరం లేదు — పైన ఉన్న భుజాల వెడల్పును ఆర్మ్‌హోల్/భుజం సూచనగా వాడండి.",
        "waist_length": "నడుము పొడవు (భుజం→నడుము)",
        "leg_length": "కాలి పొడవు",
        "bust": "బస్ట్",
        "chest": "ఛాతీ",
        "waist_est": "నడుము (అంచనా)",
        "hip_est": "హిప్ (అంచనా)",
        "detected_size_shape": "👗 గుర్తించిన సైజు & ఆకారం",
        "estimated_size": "అంచనా సైజు",
        "body_shape": "శరీర ఆకారం",
        "measurement_history": "🕘 కొలతల చరిత్ర",
        "save_to_history": "💾 ఈ ఫలితాన్ని చరిత్రలో సేవ్ చేయండి",
        "compare_entries": "రెండు ఎంట్రీలను పోల్చండి",
        "older_entry": "పాత ఎంట్రీ",
        "newer_entry": "కొత్త ఎంట్రీ",
        "clear_history": "🗑️ చరిత్రను క్లియర్ చేయండి",
        "show_history_in": "చరిత్రను చూపించండి",
    },
}


def tr(key):
    """Translate a UI string key into the current session language, falling
    back to English (or the raw key) if a translation is missing."""
    lang = st.session_state.get("language", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, TRANSLATIONS["en"].get(key, key))


def render_language_selector():
    """Shown at the very entrance of the app - a compact language toggle
    (English / Telugu) that persists in session_state and drives tr()
    everywhere else, on every page including the opening slide."""
    _, lang_col = st.columns([5, 1.3])
    with lang_col:
        choice = st.selectbox(
            tr("language_label"), ["English", "తెలుగు"],
            index=0 if st.session_state.language == "en" else 1,
            key="language_selector_top",
            label_visibility="collapsed",
        )
        st.session_state.language = "en" if choice == "English" else "te"


# ============================================================
# UNIT HELPERS
# ============================================================
def ft_in_to_cm(feet, inches):
    return (feet * 12 + inches) * CM_PER_INCH


def cm_to_display(value_cm, unit):
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


def get_pixel_height_nose_ankle(landmarks, w, h):
    """v3's original fallback method: nose-to-ankle pixel distance, scaled by a
    fixed 1.12 fudge factor. Kept ONLY as a fallback for when segmentation-based
    height (below) is unavailable, since it's less stable on its own."""
    L = mp_pose.PoseLandmark
    nose = landmarks[L.NOSE]
    left_ankle = landmarks[L.LEFT_ANKLE]
    right_ankle = landmarks[L.RIGHT_ANKLE]
    ankle = left_ankle if left_ankle.y > right_ankle.y else right_ankle
    return pixel_distance(nose, ankle, w, h) * 1.12


def get_pixel_height_from_silhouette(mask, x_center_frac, image_height, band_frac=0.18):
    """NEW in v4 - the main fix for inconsistent results.

    Finds the top-of-head to bottom-of-feet pixel span directly from the actual
    body silhouette (segmentation mask), scanning a vertical band centered on the
    torso's x-position rather than the whole frame width (so background clutter
    or a second person in frame doesn't get picked up).

    This is far more stable photo-to-photo than a landmark-based estimate because
    it doesn't care about head tilt, chin angle, or facial landmark confidence -
    it just measures where your actual outline starts and ends.

    Returns pixel height, or None if the silhouette couldn't be read reliably.
    """
    h, w = mask.shape
    x_center = int(np.clip(x_center_frac, 0, 1) * w)
    band = max(1, int(band_frac * w / 2))
    x0, x1 = max(0, x_center - band), min(w, x_center + band)
    if x1 <= x0:
        return None
    col_any = mask[:, x0:x1].any(axis=1)
    ys = np.where(col_any)[0]
    if len(ys) < 2:
        return None
    top_y, bottom_y = int(ys[0]), int(ys[-1])
    px_height = bottom_y - top_y
    # Sanity check: silhouette should span a believable fraction of the frame.
    # If it's absurdly small the mask likely picked up noise, not a person.
    if px_height < 0.15 * image_height:
        return None
    return px_height, top_y, bottom_y


def check_visibility(landmarks, indices):
    for idx in indices:
        lm = landmarks[idx]
        vis = getattr(lm, "visibility", 1.0)
        if vis < VISIBILITY_THRESHOLD:
            return False
    return True


def measure_torso_width_at_y(mask, y_pixel, anchor_x_pixel, max_gap=2):
    """Unchanged from v3 - measures torso-only width at a row by walking outward
    from a known anchor x-position, stopping at a real gap in the silhouette so a
    hanging arm doesn't get counted as part of the torso depth/width."""
    if y_pixel < 0 or y_pixel >= mask.shape[0]:
        return None
    row = mask[y_pixel, :]
    w = row.shape[0]
    anchor_x_pixel = int(np.clip(anchor_x_pixel, 0, w - 1))
    if not row[anchor_x_pixel]:
        xs = np.where(row)[0]
        if len(xs) < 2:
            return None
        anchor_x_pixel = xs[np.argmin(np.abs(xs - anchor_x_pixel))]

    left = anchor_x_pixel
    gap = 0
    x = anchor_x_pixel - 1
    while x >= 0:
        if row[x]:
            left = x
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break
        x -= 1

    right = anchor_x_pixel
    gap = 0
    x = anchor_x_pixel + 1
    while x < w:
        if row[x]:
            right = x
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break
        x += 1

    if right <= left:
        return None
    return right - left


# ============================================================
# NEW IN v4: FRAMING VALIDATION
# ============================================================
def validate_framing(landmarks, mask, fw, fh):
    """Checks the photo for the conditions most likely to cause inconsistent
    results, and returns a list of plain-language issues (empty list = good).
    This is what catches "stood a bit closer/farther/tilted this time" BEFORE
    it turns into a different number, instead of after."""
    issues = []
    L = mp_pose.PoseLandmark

    l_sh, r_sh = landmarks[L.LEFT_SHOULDER], landmarks[L.RIGHT_SHOULDER]
    shoulder_tilt_frac = abs(l_sh.y - r_sh.y)
    if shoulder_tilt_frac > 0.035:
        issues.append(
            "Your shoulders look tilted in this photo (or the camera isn't level). "
            "Stand straight-on with the camera held level for consistent width readings."
        )

    l_hip, r_hip = landmarks[L.LEFT_HIP], landmarks[L.RIGHT_HIP]
    hip_center_x = (l_hip.x + r_hip.x) / 2
    if hip_center_x < 0.30 or hip_center_x > 0.70:
        issues.append(
            "You're off to one side of the frame. Center yourself so there's roughly "
            "equal space on both sides."
        )

    if mask is not None:
        result = get_pixel_height_from_silhouette(mask, hip_center_x, fh)
        if result is not None:
            px_height, top_y, bottom_y = result
            frame_fraction = px_height / fh
            if frame_fraction < 0.55:
                issues.append(
                    f"You're a bit far from the camera (you fill only about "
                    f"{frame_fraction*100:.0f}% of the frame height). Step closer so your "
                    f"body takes up most of the frame - this reduces pixel-measurement error."
                )
            elif frame_fraction > 0.97:
                issues.append(
                    "You're too close / cropped - your head or feet may be cut off. "
                    "Step back until your full body, with a little headroom, is visible."
                )
            if top_y <= 2:
                issues.append("The top of your head looks cut off - step back a little.")
            if bottom_y >= fh - 2:
                issues.append("Your feet look cut off at the bottom of the photo - step back a little.")
        else:
            issues.append(
                "Couldn't clearly separate you from the background - use a plain, "
                "well-lit background for a more reliable reading."
            )

    return issues


# ============================================================
# NEW IN v4: PER-PHOTO ANALYSIS (used for single or multi-shot)
# ============================================================
def analyze_front_photo(front_bgr, height_cm):
    """Runs pose + segmentation on one front photo and returns a results dict,
    or a dict with an 'error' key if the photo can't be used. Keeping this as a
    single function is what lets us run it once (single-photo mode) or 2-3 times
    and average (multi-shot mode) without duplicating logic."""
    fh, fw, _ = front_bgr.shape
    L = mp_pose.PoseLandmark

    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        pose_results = pose.process(cv2.cvtColor(front_bgr, cv2.COLOR_BGR2RGB))

    if not pose_results.pose_landmarks:
        return {"error": "No pose detected. Make sure your full body is visible with good lighting."}

    landmarks = pose_results.pose_landmarks.landmark

    key_indices = [
        L.NOSE, L.LEFT_ANKLE, L.RIGHT_ANKLE, L.LEFT_SHOULDER, L.RIGHT_SHOULDER,
        L.LEFT_HIP, L.RIGHT_HIP, L.LEFT_ELBOW, L.RIGHT_ELBOW, L.LEFT_WRIST, L.RIGHT_WRIST,
        L.LEFT_KNEE, L.RIGHT_KNEE,
    ]
    confident = check_visibility(landmarks, key_indices)

    with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
        seg_results = seg.process(cv2.cvtColor(front_bgr, cv2.COLOR_BGR2RGB))
    mask = seg_results.segmentation_mask > 0.5

    issues = validate_framing(landmarks, mask, fw, fh)

    # --- Calibration: silhouette-based (v4 main fix), fallback to nose-ankle ---
    l_hip, r_hip = landmarks[L.LEFT_HIP], landmarks[L.RIGHT_HIP]
    hip_center_x = (l_hip.x + r_hip.x) / 2
    silhouette_result = get_pixel_height_from_silhouette(mask, hip_center_x, fh)

    if silhouette_result is not None:
        pixel_height = silhouette_result[0]
        calibration_method = "silhouette"
    else:
        pixel_height = get_pixel_height_nose_ankle(landmarks, fw, fh)
        calibration_method = "landmark-fallback"
        issues.append(
            "Used a less precise backup calibration method for this photo (couldn't "
            "read a clean silhouette) - results may be a little less accurate."
        )

    px_to_cm = height_cm / pixel_height

    l_shoulder, r_shoulder = landmarks[L.LEFT_SHOULDER], landmarks[L.RIGHT_SHOULDER]
    l_elbow, r_elbow = landmarks[L.LEFT_ELBOW], landmarks[L.RIGHT_ELBOW]
    l_wrist, r_wrist = landmarks[L.LEFT_WRIST], landmarks[L.RIGHT_WRIST]
    l_knee, r_knee = landmarks[L.LEFT_KNEE], landmarks[L.RIGHT_KNEE]
    l_ankle, r_ankle = landmarks[L.LEFT_ANKLE], landmarks[L.RIGHT_ANKLE]

    shoulder_width_cm = pixel_distance(l_shoulder, r_shoulder, fw, fh) * px_to_cm
    hip_width_cm = pixel_distance(l_hip, r_hip, fw, fh) * px_to_cm

    left_arm_cm = (
        pixel_distance(l_shoulder, l_elbow, fw, fh) + pixel_distance(l_elbow, l_wrist, fw, fh)
    ) * px_to_cm
    right_arm_cm = (
        pixel_distance(r_shoulder, r_elbow, fw, fh) + pixel_distance(r_elbow, r_wrist, fw, fh)
    ) * px_to_cm
    arm_length_cm = (left_arm_cm + right_arm_cm) / 2

    # NEW: half-sleeve length (shoulder -> elbow only). Not every garment has
    # full-length sleeves - shirts/tops/frocks etc often cut at the elbow -
    # so this is tracked as its own measurement point rather than folded
    # into the full arm_length_cm above.
    left_half_sleeve_cm = pixel_distance(l_shoulder, l_elbow, fw, fh) * px_to_cm
    right_half_sleeve_cm = pixel_distance(r_shoulder, r_elbow, fw, fh) * px_to_cm
    half_sleeve_length_cm = (left_half_sleeve_cm + right_half_sleeve_cm) / 2

    # NEW: 3/4-sleeve length (shoulder -> mid-forearm). A lot of everyday
    # wear - kurtis, tops, casual shirts - is cut here rather than at the
    # full wrist or the elbow, so it needs its own point too: shoulder-to-
    # elbow (the first half of the arm) plus half of elbow-to-wrist (the
    # midpoint of the forearm) lands right at the usual 3/4-sleeve hem.
    left_three_quarter_sleeve_cm = left_half_sleeve_cm + (
        pixel_distance(l_elbow, l_wrist, fw, fh) * px_to_cm
    ) * 0.5
    right_three_quarter_sleeve_cm = right_half_sleeve_cm + (
        pixel_distance(r_elbow, r_wrist, fw, fh) * px_to_cm
    ) * 0.5
    three_quarter_sleeve_length_cm = (
        left_three_quarter_sleeve_cm + right_three_quarter_sleeve_cm
    ) / 2

    left_leg_cm = (
        pixel_distance(l_hip, l_knee, fw, fh) + pixel_distance(l_knee, l_ankle, fw, fh)
    ) * px_to_cm
    right_leg_cm = (
        pixel_distance(r_hip, r_knee, fw, fh) + pixel_distance(r_knee, r_ankle, fw, fh)
    ) * px_to_cm
    leg_length_cm = (left_leg_cm + right_leg_cm) / 2

    chest_y_norm = (l_shoulder.y + r_shoulder.y) / 2
    hip_y_norm = (l_hip.y + r_hip.y) / 2
    waist_y_norm = chest_y_norm + (hip_y_norm - chest_y_norm) * 0.6

    # NEW: waist point - vertical shoulder-to-waist length (garment/blouse
    # length reference for anything cut at the waist, e.g. crop tops, cropped
    # blouses, waist-length jackets), distinct from the full torso/leg spans
    # already computed elsewhere.
    waist_length_cm = abs(waist_y_norm - chest_y_norm) * fh * px_to_cm

    # --------------------------------------------------------
    # FIX: true silhouette-based front widths for chest/waist/hip
    # --------------------------------------------------------
    # BUG THIS FIXES: chest/waist/hip circumference used to be calculated from
    # shoulder_width_cm / hip_width_cm above, which are landmark-to-landmark
    # distances between MediaPipe's shoulder/hip JOINT points. Those joints sit
    # noticeably INSIDE the body's actual outline (a few cm in from the real
    # skin/clothing edge on each side), so every circumference computed from
    # them was systematically too small - which is exactly what pushes a
    # true M/L result down into the S bracket. The fix: walk out from the
    # torso center along the segmentation mask (the same edge-walking already
    # used for side-view depth) to find the REAL visible body width at chest,
    # waist, and hip height, and use that for circumference instead. Falls
    # back to the old landmark-based estimate only if the mask can't be read
    # cleanly at that row (e.g. bad lighting/background).
    chest_anchor_x_px = ((l_shoulder.x + r_shoulder.x) / 2) * fw
    hip_anchor_x_px = ((l_hip.x + r_hip.x) / 2) * fw
    waist_anchor_x_px = chest_anchor_x_px + (hip_anchor_x_px - chest_anchor_x_px) * 0.6

    chest_y_px = int(chest_y_norm * fh)
    waist_y_px = int(waist_y_norm * fh)
    hip_y_px = int(hip_y_norm * fh)

    chest_width_px = measure_torso_width_at_y(mask, chest_y_px, chest_anchor_x_px)
    waist_width_px = measure_torso_width_at_y(mask, waist_y_px, waist_anchor_x_px)
    hip_width_px = measure_torso_width_at_y(mask, hip_y_px, hip_anchor_x_px)

    front_width_fallback_used = False
    if chest_width_px:
        chest_width_front_cm = chest_width_px * px_to_cm
    else:
        chest_width_front_cm = shoulder_width_cm
        front_width_fallback_used = True

    if waist_width_px:
        waist_width_front_cm = waist_width_px * px_to_cm
    else:
        waist_width_front_cm = hip_width_cm * 0.9
        front_width_fallback_used = True

    if hip_width_px:
        hip_width_front_cm = hip_width_px * px_to_cm
    else:
        hip_width_front_cm = hip_width_cm
        front_width_fallback_used = True

    if front_width_fallback_used:
        issues.append(
            "Couldn't read a clean body outline at chest/waist/hip height on this "
            "photo, so a less precise backup width estimate was used there - a "
            "plain, well-lit background improves this."
        )

    # Front-view arms hanging flush against the torso can get merged into the
    # waist/hip width reading (same issue already flagged for the side photo).
    front_arm_overlap_warning = False
    arm_y_positions = [l_elbow.y, r_elbow.y, l_wrist.y, r_wrist.y]
    for target_y in (waist_y_norm, hip_y_norm):
        if any(abs(ay - target_y) < 0.03 for ay in arm_y_positions):
            front_arm_overlap_warning = True
            break
    if front_arm_overlap_warning:
        issues.append(
            "Your arms look close to your torso at waist/hip height. Stand with a "
            "small gap between your arms and your sides for the most accurate "
            "waist/hip width."
        )

    return {
        "error": None,
        "landmarks": landmarks,
        "pose_landmarks_proto": pose_results.pose_landmarks,
        "mask": mask,
        "fw": fw,
        "fh": fh,
        "confident": confident,
        "issues": issues,
        "calibration_method": calibration_method,
        "shoulder_width_cm": shoulder_width_cm,
        "hip_width_cm": hip_width_cm,
        "chest_width_front_cm": chest_width_front_cm,
        "waist_width_front_cm": waist_width_front_cm,
        "hip_width_front_cm": hip_width_front_cm,
        "arm_length_cm": arm_length_cm,
        "half_sleeve_length_cm": half_sleeve_length_cm,
        "three_quarter_sleeve_length_cm": three_quarter_sleeve_length_cm,
        "leg_length_cm": leg_length_cm,
        "waist_length_cm": waist_length_cm,
        "chest_y_norm": chest_y_norm,
        "hip_y_norm": hip_y_norm,
        "waist_y_norm": waist_y_norm,
    }


def average_with_consistency(values):
    """Returns (mean, consistency_pct) where consistency_pct is 100 minus the
    coefficient-of-variation as a percent - a simple, readable way to show the
    user how much their shots agreed with each other."""
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    if mean == 0:
        return mean, 100.0
    std = float(np.std(arr))
    cv_pct = (std / mean) * 100
    consistency_pct = max(0.0, 100.0 - cv_pct * 4)  # scaled so small % noise reads clearly
    return mean, consistency_pct


# ============================================================
# NEW: RETAKE INSTRUCTION MAPPING
# ============================================================
def map_issue_to_retake_instruction(issue_text):
    """Turns one of the existing plain-language warning strings (from
    validate_framing / analyze_front_photo / check_capture_readiness) into a
    short, actionable retake instruction, e.g. "Feet not visible -> Step
    back" or "Side view unclear -> Turn 90 degrees". Pattern-matches on the
    warning text itself, so it keeps working even as new warnings are added -
    anything unmatched just gets a generic fallback instead of breaking."""
    text = issue_text.lower()
    if "feet" in text and "cut off" in text:
        return "Feet not visible → Step back"
    if "head" in text and "cut off" in text:
        return "Head cut off → Step back"
    if "far from the camera" in text:
        return "Too far away → Step closer"
    if "too close" in text or "cropped" in text:
        return "Too close / cropped → Step back"
    if "tilted" in text:
        return "Camera or shoulders tilted → Stand straight, hold camera level"
    if "off to one side" in text or "center" in text:
        return "Off-center → Move to the middle of the frame"
    if "separate you from the background" in text or "plain, well-lit background" in text:
        return "Background unclear → Use a plain, well-lit background"
    if "arm" in text and ("close" in text or "overlap" in text or "hanging" in text):
        return "Arms too close to body → Add a small gap between arms and torso"
    if "side photo" in text or "side view" in text:
        return "Side view unclear → Turn 90°"
    if "quite dark" in text:
        return "Photo too dark → Move to better lighting"
    if "overexposed" in text or "too bright" in text:
        return "Photo too bright → Reduce direct light/backlighting"
    if "low confidence" in text or "key body points" in text:
        return "Body points unclear → Make sure your full body is visible with good lighting"
    if "no body detected" in text or "no pose detected" in text:
        return "Body not detected → Stand fully in frame and retake"
    if "backup calibration" in text or "silhouette" in text:
        return "Outline unclear → Use a plain background, retake"
    return "Retake following the tips above"


# ============================================================
# SIZE + BODY SHAPE DETECTION (unchanged from v3)
# ============================================================
def _bound_to_index(value, bounds):
    for i, b in enumerate(bounds):
        if value <= b:
            return i
    return len(bounds)


def detect_size(gender, chest_or_bust_cm, waist_cm, hip_cm=None):
    if gender == "Women":
        indices = [
            _bound_to_index(chest_or_bust_cm, WOMEN_BUST_BOUNDS),
            _bound_to_index(waist_cm, WOMEN_WAIST_BOUNDS),
        ]
        if hip_cm is not None:
            indices.append(_bound_to_index(hip_cm, WOMEN_HIP_BOUNDS))
        avg_idx = round(sum(indices) / len(indices))
        avg_idx = max(0, min(avg_idx, len(SIZE_LABELS) - 1))
        return SIZE_LABELS[avg_idx]
    elif gender == "Men":
        indices = [
            _bound_to_index(chest_or_bust_cm, MEN_CHEST_BOUNDS),
            _bound_to_index(waist_cm, MEN_WAIST_BOUNDS),
        ]
        avg_idx = round(sum(indices) / len(indices))
        avg_idx = max(0, min(avg_idx, len(SIZE_LABELS) - 1))
        return SIZE_LABELS[avg_idx]
    elif gender == "Boys":
        # Boys' kidswear is conventionally sized off chest + waist (no hip).
        indices = [
            _bound_to_index(chest_or_bust_cm, BOYS_CHEST_BOUNDS),
            _bound_to_index(waist_cm, BOYS_WAIST_BOUNDS),
        ]
        avg_idx = round(sum(indices) / len(indices))
        avg_idx = max(0, min(avg_idx, len(KIDS_SIZE_LABELS) - 1))
        return KIDS_SIZE_LABELS[avg_idx]
    else:  # "Girls"
        indices = [
            _bound_to_index(chest_or_bust_cm, GIRLS_CHEST_BOUNDS),
            _bound_to_index(waist_cm, GIRLS_WAIST_BOUNDS),
        ]
        if hip_cm is not None:
            indices.append(_bound_to_index(hip_cm, GIRLS_HIP_BOUNDS))
        avg_idx = round(sum(indices) / len(indices))
        avg_idx = max(0, min(avg_idx, len(KIDS_SIZE_LABELS) - 1))
        return KIDS_SIZE_LABELS[avg_idx]


def detect_body_shape(gender, chest_or_bust_cm, waist_cm, hip_cm):
    if gender == "Women":
        bust, waist, hip = chest_or_bust_cm, waist_cm, hip_cm
        bust_hip_diff_ratio = abs(bust - hip) / max(bust, hip)
        if waist <= bust * 0.75 and waist <= hip * 0.75 and bust_hip_diff_ratio <= 0.05:
            return "Hourglass", "Bust and hip are close in size with a clearly defined, smaller waist."
        if hip > bust * 1.05:
            return "Pear / Triangle", "Hips are noticeably wider than the bust."
        if bust > hip * 1.05:
            return "Inverted Triangle", "Bust/shoulders are noticeably wider than the hips."
        if waist >= bust * 0.95 and waist >= hip * 0.95:
            return "Apple / Oval", "Waist is close to or larger than bust and hip."
        return "Rectangle", "Bust, waist, and hip are all fairly similar in width."
    elif gender == "Men":
        chest, waist, hip = chest_or_bust_cm, waist_cm, hip_cm
        if chest > waist * 1.1 and chest > hip * 1.05:
            return "V-Shape / Athletic", "Chest and shoulders are noticeably broader than the waist."
        if waist >= chest * 0.97 and waist >= hip * 0.97:
            return "Oval", "Waist is close to or larger than chest and hip."
        return "Rectangle", "Chest, waist, and hip are fairly similar in width."
    elif gender == "Boys":
        # Kidswear commonly classifies boys' builds as Slim/Regular/Husky
        # (based on waist relative to chest) instead of adult shape categories.
        chest, waist = chest_or_bust_cm, waist_cm
        if waist <= chest * 0.88:
            return "Slim Build", "Waist is notably smaller than chest — slim-fit bottoms will likely sit better."
        if waist >= chest * 0.98:
            return "Husky Build", "Waist is close to or larger than chest — husky/relaxed-fit bottoms are worth trying."
        return "Regular Build", "Waist and chest are proportionate — standard/regular-fit sizing should work well."
    else:  # "Girls"
        # For girls, waist relative to hip is the more standard fit indicator.
        waist, hip = waist_cm, hip_cm
        if waist <= hip * 0.85:
            return "Slim Build", "Waist is notably smaller than hip — slim-fit bottoms will likely sit better."
        if waist >= hip * 0.95:
            return "Fuller Build", "Waist is close to hip in size — relaxed-fit or adjustable-waist bottoms may fit better."
        return "Regular Build", "Waist and hip are proportionate — standard/regular-fit sizing should work well."


# ============================================================
# OTP HELPERS (demo only - unchanged from v3, see honesty note)
# ============================================================
def generate_otp():
    return str(random.randint(1000, 9999))


# ============================================================
# LOGIN PAGE (unchanged from v3)
# ============================================================
def page_login():
    st.markdown(f"<div class='pill'>{tr('app_pill')}</div>", unsafe_allow_html=True)
    st.title(tr("welcome"))
    st.write(tr("signin_sub"))

    contact_options = [tr("phone_number"), tr("email")]
    contact_choice = st.radio(tr("sign_in_with"), contact_options, horizontal=True)
    st.session_state.contact_method = "Phone Number" if contact_choice == contact_options[0] else "Email"

    if st.session_state.contact_method == "Phone Number":
        st.session_state.contact_value = st.text_input(
            tr("phone_number"), value=st.session_state.contact_value, placeholder="+91 98765 43210"
        )
    else:
        st.session_state.contact_value = st.text_input(
            tr("email"), value=st.session_state.contact_value, placeholder="you@example.com"
        )

    if st.button(tr("send_otp")):
        if not st.session_state.contact_value.strip():
            st.error("Please enter your phone number or email first.")
        else:
            otp = generate_otp()
            st.session_state.generated_otp = otp
            st.session_state.otp_send_error = None

            sent_for_real = False
            if st.session_state.contact_method == "Email" and has_email_credentials():
                sent_for_real = send_otp_email_real(st.session_state.contact_value, otp)
            elif st.session_state.contact_method == "Phone Number" and has_sms_credentials():
                sent_for_real = send_otp_sms_real(st.session_state.contact_value, otp)

            st.session_state.otp_is_real = sent_for_real
            st.session_state.otp_sent = True

    if st.session_state.otp_sent:
        if st.session_state.otp_is_real:
            st.markdown(
                f"""<div class="good-box">✅ A real verification code was just sent to
                <b>{st.session_state.contact_value}</b>. Check your inbox (and spam folder)
                and enter the code below.</div>""",
                unsafe_allow_html=True,
            )
        else:
            demo_reason = ""
            if st.session_state.otp_send_error:
                demo_reason = f"<br><span style='font-size:12px;'>Send attempt failed: {st.session_state.otp_send_error}</span>"
            st.markdown(
                f"""<div class="info-box">🔐 <b>Demo mode:</b> your verification code is
                <b>{st.session_state.generated_otp}</b>. Enter this code below to continue.
                {demo_reason}</div>""",
                unsafe_allow_html=True,
            )
            with st.expander("Developer setup", expanded=False):
                st.caption(
                    "To enable real email OTP delivery, configure EMAIL_ADDRESS and "
                    "EMAIL_APP_PASSWORD in Streamlit secrets. SMS delivery requires the "
                    "optional Twilio credentials described in the developer configuration."
                )
        entered_otp = st.text_input(tr("enter_otp"), max_chars=4)
        if st.button(tr("verify_continue")):
            if entered_otp == st.session_state.generated_otp:
                st.session_state.logged_in = True
                st.session_state.page = "home"
                st.rerun()
            else:
                st.error("Incorrect OTP. Please try again.")


# ============================================================
# HOME PAGE (unchanged from v3, feature list updated)
# ============================================================
def page_home():
    st.title(tr("home_title"))
    st.write(f"Signed in as **{st.session_state.contact_value}**")

    st.subheader(tr("what_app_can_do"))
    features = [
        ("📸 Guided Photo Capture", "Take a front photo (required) and a side photo (optional but recommended) for measuring."),
        ("🎯 Silhouette-Based Calibration", "v4: height calibration now uses your actual body outline instead of a guessed multiplier, for much more consistent results photo to photo."),
        ("✅ Photo Framing Checks", "v4: the app checks your distance, tilt, and centering before measuring, and tells you exactly what to fix if something's off."),
        ("🔁 Multi-Photo Averaging", "v4: optionally take up to 3 front photos - the app averages them and shows a consistency score."),
        ("📏 Manual Height Entry", "Camera-based height guessing is unreliable, so you enter your real height in cm or feet & inches for accurate calibration."),
        ("🧍 Pose & Body Detection", "MediaPipe Pose finds your body landmarks; Selfie Segmentation measures real body depth from the side photo."),
        ("👗 Automatic Size Detection", "Estimates clothing size — adult XS–XXXL, or kids' age brackets (2-3Y – 16Y+) for Boys/Girls — from chest/bust, waist, and hip measurements."),
        ("🧒 Kids Sizing (Boys & Girls)", "A dedicated Kids profile with age input and boys'/girls' size charts and build (Slim/Regular/Husky) detection, kept separate from the adult charts for accuracy."),
        ("👕 Clothing Purpose Selector", "Pick what the measurements are for — blouse, chudidhar, shirt, frock, kurta and more — from lists tailored to Women, Men, Boys, or Girls."),
        ("🔺 Body Shape Detection", "Classifies your body shape (e.g. Hourglass, Pear, Inverted Triangle, Rectangle, V-Shape, Oval)."),
        ("⚧ Gender-Specific Settings", "Women's, Men's, and Kids' (Boys/Girls) profiles each use different measurement ratios and size charts for better accuracy."),
        ("🔁 Unit Conversion", "View every result in centimeters or inches, instantly."),
        ("✂️ Crop Before Measuring", "v6: trim every photo (front, side, back) down to just your body before it's analyzed."),
        ("🔄 Optional Back-View Photo", "v6: add a back-facing photo for an extra, cross-checked data point."),
        ("🧍 Landmarks + Framing Checks on Every Photo", "v6: the side and back photos now get the same landmark overlay and framing warnings the front photo already had."),
        ("✂️ Elbow & Waist Measurement Points", "Full sleeve (shoulder→wrist), 3/4 sleeve (shoulder→mid-forearm), half sleeve (shoulder→elbow), and waist length (shoulder→waist) are all measured separately, since not everyone wants a full-length sleeve — pick Full, Half, 3/4, or Sleeveless on the measurements form and the result that matches gets called out for you."),
        ("🌐 English / Telugu", "Switch the app language from the selector at the very top of the screen — it applies everywhere, right from the first page."),
    ]
    for name, desc in features:
        st.markdown(
            f"""<div class="feature-card"><b>{name}</b><br><span style="font-size:14px;">{desc}</span></div>""",
            unsafe_allow_html=True,
        )

    st.subheader(tr("preferences"))
    st.session_state.notifications_enabled = st.toggle(
        "Enable reminder notifications (e.g. re-measure every few weeks)",
        value=st.session_state.notifications_enabled,
    )
    st.markdown(
        """<div class="info-box">ℹ️ This toggle only saves a local preference for this
        session right now — no push/SMS notification service is connected yet. To make
        this actually notify you, a backend plus a service like Firebase Cloud
        Messaging, OneSignal, or Twilio would need to be integrated.</div>""",
        unsafe_allow_html=True,
    )

    st.subheader(tr("profile_setting"))
    _profile_values = ["Women", "Men", "Kids"]
    _profile_labels = [tr("women"), tr("men"), tr("kids")]
    _current_profile = st.session_state.gender if st.session_state.gender in ("Women", "Men") else "Kids"
    _profile_choice_label = st.radio(
        tr("measurement_profile"), _profile_labels,
        index=_profile_values.index(_current_profile),
        horizontal=True,
    )
    profile_choice = _profile_values[_profile_labels.index(_profile_choice_label)]
    if profile_choice == "Kids":
        _kid_values = ["Boys", "Girls"]
        _kid_labels = [tr("boys"), tr("girls")]
        _kid_choice_label = st.radio(
            tr("kids_category"), _kid_labels,
            index=0 if st.session_state.kid_type == "Boys" else 1,
            horizontal=True,
            key="home_kid_type",
        )
        st.session_state.kid_type = _kid_values[_kid_labels.index(_kid_choice_label)]
        st.session_state.gender = st.session_state.kid_type
    else:
        st.session_state.gender = profile_choice
    st.caption("This affects which size chart and body-shape rules are used on the Measurements page.")


# ============================================================
# MEASUREMENTS PAGE (v4: framing checks + optional multi-shot)
# ============================================================
def page_measurements():
    st.title(tr("measurements_title"))

    st.markdown(
            "**Instructions:** Stand straight facing the camera for the front photo, "
            "arms slightly away from your body, full body visible (head to feet, with a little "
            "margin), plain background, good lighting, camera held level at chest height "
            "(about 4.5 ft), about **10 feet (~3 meters) back**. For the side photo, turn 90° "
            "and stand the same 10 feet from the camera - keep a small gap between your arm and "
            "your torso (rest your hand slightly forward or bend the elbow a touch) so your arm "
            "doesn't overlap your torso outline. For the optional back photo, turn all the way "
            "around (back to the camera), same distance, arms slightly away from your body."
    )

    # NEW in v6: instruction graphics shown at the top of the page
    show_instruction_images()

    category_values = ["Women", "Men", "Kids"]
    category_labels = [tr("women"), tr("men"), tr("kids")]
    current_category = st.session_state.gender if st.session_state.gender in ("Women", "Men") else "Kids"
    category_choice_label = st.radio(
        tr("who_for"), category_labels,
        index=category_values.index(current_category),
        horizontal=True,
    )
    category = category_values[category_labels.index(category_choice_label)]

    kid_age = st.session_state.kid_age
    if category == "Kids":
        kid_col1, kid_col2 = st.columns(2)
        with kid_col1:
            kid_values = ["Boys", "Girls"]
            kid_labels = [tr("boys"), tr("girls")]
            kid_choice_label = st.radio(
                tr("kids_category"), kid_labels,
                index=0 if st.session_state.kid_type == "Boys" else 1,
                horizontal=True,
            )
            st.session_state.kid_type = kid_values[kid_labels.index(kid_choice_label)]
        with kid_col2:
            kid_age = st.number_input(
                tr("child_age"), min_value=2, max_value=16,
                value=int(st.session_state.kid_age), step=1,
            )
        st.session_state.kid_age = kid_age
        st.caption(
            "Age is used to sanity-check the detected kids' size (2-3Y … 16Y+) "
            "against the actual measured chest/waist — the real photo measurements "
            "still drive the numbers, age just helps flag anything off."
        )
        gender = st.session_state.kid_type  # "Boys" or "Girls" — used as the effective profile below
    else:
        gender = category
    st.session_state.gender = gender

    st.markdown(
        """<div class="info-box">📏 <b>For accurate results, always type your real height
        manually</b> (measured with a tape/wall, not guessed) — every other measurement is
        calibrated off this number, so an inaccurate height throws everything else off.</div>""",
        unsafe_allow_html=True,
    )

    height_unit = st.radio(tr("height_unit"), ["cm", "feet & inches"], horizontal=True)
    default_height_cm = 165.0 if category != "Kids" else 130.0
    min_height_cm = 70.0 if category == "Kids" else 100.0
    if height_unit == "cm":
        height_cm = st.number_input(
            tr("enter_height") if category != "Kids" else tr("enter_child_height"),
            min_value=min_height_cm, max_value=220.0, value=default_height_cm, step=0.5,
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            feet = st.number_input("Feet", min_value=2, max_value=7, value=5, step=1)
        with c2:
            inches = st.number_input("Inches", min_value=0.0, max_value=11.9, value=5.0, step=0.5)
        height_cm = ft_in_to_cm(feet, inches)
        st.caption(f"= {height_cm:.1f} cm")

    st.session_state.use_multi_shot = st.checkbox(
        "📸 Take up to 3 front photos and average them (recommended - reduces photo-to-photo variation)",
        value=st.session_state.use_multi_shot,
    )
    st.session_state.use_back_view = st.checkbox(
        "🔄 Also take a back-view photo (optional - adds another data point for consistency)",
        value=st.session_state.use_back_view,
    )

    # ------------------------------------------------------------
    # WHAT ARE THE MEASUREMENTS FOR? (clothing purpose)
    # Shown above the photo-capture section so the garment context is set
    # before any photos are taken. Options change based on the profile
    # (Women / Men / Boys / Girls) chosen above.
    # ------------------------------------------------------------
    st.write("---")
    clothing_options = CLOTHING_ITEMS[gender]
    current_item = st.session_state.clothing_item
    default_index = clothing_options.index(current_item) if current_item in clothing_options else 0
    st.session_state.clothing_item = st.selectbox(
        tr("what_for"), clothing_options,
        index=default_index,
    )

    # NEW: sleeve type. Not everyone needs the full shoulder->wrist figure -
    # kurtis/tops are often 3/4, plenty of blouses/frocks are sleeveless - so
    # ask up front and use it below to point at the one number that's
    # actually relevant, instead of leaving the person to guess which of the
    # four sleeve figures applies to their garment.
    current_sleeve = st.session_state.sleeve_type
    sleeve_default_index = SLEEVE_TYPES.index(current_sleeve) if current_sleeve in SLEEVE_TYPES else 0
    st.session_state.sleeve_type = st.radio(
        tr("sleeve_type"), SLEEVE_TYPES,
        index=sleeve_default_index,
        horizontal=True,
    )
    st.caption(f"Measuring for: **{gender}** → **{st.session_state.clothing_item}** → **{st.session_state.sleeve_type}**")
    st.write("---")

    # Camera capture is deliberately sequential. Only ONE timer-camera component
    # is mounted at a time, so the browser never has multiple components fighting
    # over the same physical webcam. NEW in v6: every capture now also goes
    # through a crop-and-confirm step (capture_crop_and_confirm) before the
    # photo is considered "stored", so background is trimmed out first.
    front_col, side_col, back_col = st.columns(3)

    with front_col:
        st.subheader(tr("front_photos"))
        st.caption("A 5-second countdown starts after you click the button below, so you have time to get in position.")

        # If multi-shot is switched off, only shot 1 is used.
        if not st.session_state.use_multi_shot:
            st.session_state.front_image_2 = None
            st.session_state.front_image_3 = None
            st.session_state.front2_raw = None
            st.session_state.front3_raw = None

        # -------------------------
        # FRONT SHOT 1
        # -------------------------
        if not capture_crop_and_confirm("Front view - shot 1", 5, "front1", "front_image_1"):
            if st.session_state.get("front1_raw") is None:
                st.info("Capture your first front-facing photo to continue.")
            return
        else:
            st.success("✅ Front shot 1 captured, cropped, and stored.")
            if st.button("Retake shot 1", key="retake_front1"):
                st.session_state.front_image_1 = None
                st.session_state.front_image_2 = None
                st.session_state.front_image_3 = None
                st.session_state.front1_raw = None
                st.session_state.front2_raw = None
                st.session_state.front3_raw = None
                st.rerun()

        # -------------------------
        # FRONT SHOT 2
        # -------------------------
        if st.session_state.use_multi_shot:
            if not capture_crop_and_confirm("Front view - shot 2", 5, "front2", "front_image_2"):
                st.caption("Shot 1 is stored. Reposition slightly (or stay in the same position), then take shot 2.")
                return
            else:
                st.success("✅ Front shot 2 captured, cropped, and stored.")
                if st.button("Retake shot 2", key="retake_front2"):
                    st.session_state.front_image_2 = None
                    st.session_state.front_image_3 = None
                    st.session_state.front2_raw = None
                    st.session_state.front3_raw = None
                    st.rerun()

            # -------------------------
            # FRONT SHOT 3
            # -------------------------
            if not capture_crop_and_confirm("Front view - shot 3", 5, "front3", "front_image_3"):
                st.caption("Shot 2 is stored. Reposition slightly, then take the final shot 3.")
                return
            else:
                st.success("✅ Front shot 3 captured, cropped, and stored.")

    # Only after every required front shot is stored do we mount the side camera.
    with side_col:
        st.subheader(tr("side_photo"))
        st.caption("A 5-second countdown gives you time to turn 90° after clicking.")

        if not capture_crop_and_confirm("Side view", 5, "side", "side_image"):
            pass
        else:
            st.success("✅ Side photo captured, cropped, and stored.")
            if st.button("Retake side photo", key="retake_side"):
                st.session_state.side_image = None
                st.session_state.side_raw = None
                st.rerun()

    # Back-view camera only mounts if the user opted in above.
    with back_col:
        if st.session_state.use_back_view:
            st.subheader(tr("back_photo"))
            st.caption("A 5-second countdown gives you time to turn all the way around after clicking.")

            if not capture_crop_and_confirm("Back view", 5, "back", "back_image"):
                pass
            else:
                st.success("✅ Back photo captured, cropped, and stored.")
                if st.button("Retake back photo", key="retake_back"):
                    st.session_state.back_image = None
                    st.session_state.back_raw = None
                    st.rerun()
        else:
            st.session_state.back_image = None
            st.session_state.back_raw = None

    front_image_1 = st.session_state.front_image_1
    front_image_2 = st.session_state.front_image_2
    front_image_3 = st.session_state.front_image_3
    side_image = st.session_state.side_image
    back_image = st.session_state.back_image

    front_images = [img for img in [front_image_1, front_image_2, front_image_3] if img is not None]

    # ------------------------------------------------------
    # Analyze each front photo independently
    # ------------------------------------------------------
    shot_results = []
    for i, image in enumerate(front_images):
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        result = analyze_front_photo(bgr, height_cm)
        if result["error"]:
            st.error(f"Shot {i+1}: {result['error']}")
            continue
        shot_results.append(result)

    if not shot_results:
        st.error("No usable front photo — please retake with your full body visible and good lighting.")
        return

    # Show the first successful shot, annotated with detected landmarks, for visual confirmation
    first = shot_results[0]
    annotated_preview = cv2.cvtColor(np.array(front_images[0]), cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(annotated_preview, first["pose_landmarks_proto"], mp_pose.POSE_CONNECTIONS)
    st.image(
        cv2.cvtColor(annotated_preview, cv2.COLOR_BGR2RGB),
        caption=f"Front view — shot 1 of {len(front_images)} (landmarks detected)",
        use_container_width=True,
    )

    # ------------------------------------------------------
    # Surface framing issues from every shot, clearly
    # ------------------------------------------------------
    # NEW: all_warnings collects every plain-language warning across front,
    # back, and side photos, so the Retake Guidance section further down can
    # turn them into short instructions in one place.
    all_warnings = []
    any_issues = False
    for i, r in enumerate(shot_results):
        if r["issues"]:
            any_issues = True
            all_warnings.extend(r["issues"])
            issue_text = "<br>".join(f"• {msg}" for msg in r["issues"])
            st.markdown(
                f'<div class="warn-box">⚠️ <b>Shot {i+1} framing notes:</b><br>{issue_text}</div>',
                unsafe_allow_html=True,
            )
        if not r["confident"]:
            all_warnings.append(
                f"Shot {i+1}: some key body points were detected with low confidence "
                f"(lighting, loose clothing, or part of the body out of frame)."
            )
            st.markdown(
                f'<div class="warn-box">⚠️ Shot {i+1}: some key body points were detected with '
                f'low confidence (lighting, loose clothing, or part of the body out of frame).</div>',
                unsafe_allow_html=True,
            )
    if not any_issues:
        st.markdown('<div class="good-box">✅ Framing looks good on all photos used.</div>', unsafe_allow_html=True)

    # ------------------------------------------------------
    # NEW in v6: back-view photo analysis
    # Reuses analyze_front_photo() since a back-facing photo has the same
    # shoulder/hip landmark geometry as a front-facing one - it draws
    # landmarks and runs the same framing checks that were previously only
    # applied to front shots, and its width readings extend the average
    # instead of being wasted.
    # ------------------------------------------------------
    back_result = None
    if back_image is not None:
        back_bgr_full = cv2.cvtColor(np.array(back_image), cv2.COLOR_RGB2BGR)
        back_result = analyze_front_photo(back_bgr_full, height_cm)
        if back_result["error"]:
            st.warning(f"Back photo: {back_result['error']}")
            back_result = None
        else:
            annotated_back = back_bgr_full.copy()
            mp_drawing.draw_landmarks(annotated_back, back_result["pose_landmarks_proto"], mp_pose.POSE_CONNECTIONS)
            st.image(
                cv2.cvtColor(annotated_back, cv2.COLOR_BGR2RGB),
                caption="Back view captured (landmarks detected)",
                use_container_width=True,
            )
            if back_result["issues"]:
                all_warnings.extend(back_result["issues"])
                issue_text = "<br>".join(f"• {msg}" for msg in back_result["issues"])
                st.markdown(
                    f'<div class="warn-box">⚠️ <b>Back photo framing notes:</b><br>{issue_text}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="good-box">✅ Back photo framing looks good.</div>', unsafe_allow_html=True)

    # ------------------------------------------------------
    # Average across shots + consistency score
    # ------------------------------------------------------
    shoulder_vals = [r["shoulder_width_cm"] for r in shot_results]
    hip_vals = [r["hip_width_cm"] for r in shot_results]
    chest_width_front_vals = [r["chest_width_front_cm"] for r in shot_results]
    waist_width_front_vals = [r["waist_width_front_cm"] for r in shot_results]
    hip_width_front_vals = [r["hip_width_front_cm"] for r in shot_results]
    if back_result is not None:
        # Extra data points from the back photo, folded into the same
        # consistency-averaged figures used for the front-only case.
        shoulder_vals.append(back_result["shoulder_width_cm"])
        hip_vals.append(back_result["hip_width_cm"])
        chest_width_front_vals.append(back_result["chest_width_front_cm"])
        waist_width_front_vals.append(back_result["waist_width_front_cm"])
        hip_width_front_vals.append(back_result["hip_width_front_cm"])
    arm_vals = [r["arm_length_cm"] for r in shot_results]
    half_sleeve_vals = [r["half_sleeve_length_cm"] for r in shot_results]
    three_quarter_sleeve_vals = [r["three_quarter_sleeve_length_cm"] for r in shot_results]
    leg_vals = [r["leg_length_cm"] for r in shot_results]
    waist_length_vals = [r["waist_length_cm"] for r in shot_results]

    shoulder_width_cm, shoulder_consistency = average_with_consistency(shoulder_vals)
    hip_width_cm, hip_consistency = average_with_consistency(hip_vals)
    chest_width_front_cm, _ = average_with_consistency(chest_width_front_vals)
    waist_width_front_cm, _ = average_with_consistency(waist_width_front_vals)
    hip_width_front_cm, _ = average_with_consistency(hip_width_front_vals)
    arm_length_cm, _ = average_with_consistency(arm_vals)
    half_sleeve_length_cm, _ = average_with_consistency(half_sleeve_vals)
    three_quarter_sleeve_length_cm, _ = average_with_consistency(three_quarter_sleeve_vals)
    leg_length_cm, _ = average_with_consistency(leg_vals)
    waist_length_cm, _ = average_with_consistency(waist_length_vals)

    if len(shot_results) > 1:
        overall_consistency = min(shoulder_consistency, hip_consistency)
        if overall_consistency >= 90:
            st.markdown(
                f'<div class="good-box">✅ <b>Consistency across your {len(shot_results)} shots: '
                f'{overall_consistency:.0f}%</b> — your photos agree closely, results below should be reliable.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="warn-box">⚠️ <b>Consistency across your {len(shot_results)} shots: '
                f'{overall_consistency:.0f}%</b> — your shots vary more than expected. Try to stand in the '
                f'same spot, distance, and pose for each shot. Using the average anyway below.</div>',
                unsafe_allow_html=True,
            )

    # Use first successful shot's landmarks/mask for row-fraction geometry
    chest_y_norm = first["chest_y_norm"]
    hip_y_norm = first["hip_y_norm"]
    waist_y_norm = first["waist_y_norm"]

    # ------------------------------------------------------
    # Side photo - real depth (unchanged logic from v3, but now uses
    # silhouette-based height calibration too, for consistency with the front)
    # ------------------------------------------------------
    chest_depth_cm = waist_depth_cm = hip_depth_cm = None
    arm_overlap_warning = False

    if side_image is not None:
        side_bgr = cv2.cvtColor(np.array(side_image), cv2.COLOR_RGB2BGR)
        sh, sw, _ = side_bgr.shape

        with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose_side:
            side_pose_results = pose_side.process(cv2.cvtColor(side_bgr, cv2.COLOR_BGR2RGB))

        if side_pose_results.pose_landmarks:
            side_landmarks = side_pose_results.pose_landmarks.landmark
            SL = mp_pose.PoseLandmark

            with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
                seg_results = seg.process(cv2.cvtColor(side_bgr, cv2.COLOR_BGR2RGB))
            mask = seg_results.segmentation_mask > 0.5

            # NEW in v6: the side photo previously skipped the framing checks
            # that the front photo already ran (distance / centering / cut-off
            # head-feet). A badly-framed side photo silently produced a bad
            # chest/waist/hip depth with no warning - this closes that gap.
            side_hip_center_x_for_check = (side_landmarks[SL.LEFT_HIP].x + side_landmarks[SL.RIGHT_HIP].x) / 2
            side_issues = validate_framing(side_landmarks, mask, sw, sh)

            s_l_hip, s_r_hip = side_landmarks[SL.LEFT_HIP], side_landmarks[SL.RIGHT_HIP]
            side_hip_center_x = (s_l_hip.x + s_r_hip.x) / 2
            silhouette_result_side = get_pixel_height_from_silhouette(mask, side_hip_center_x, sh)
            if silhouette_result_side is not None:
                pixel_height_side = silhouette_result_side[0]
            else:
                pixel_height_side = get_pixel_height_nose_ankle(side_landmarks, sw, sh)
                st.caption("ℹ️ Side photo used backup calibration (couldn't read a clean silhouette).")
            px_to_cm_side = height_cm / pixel_height_side

            s_l_shoulder, s_r_shoulder = side_landmarks[SL.LEFT_SHOULDER], side_landmarks[SL.RIGHT_SHOULDER]
            s_l_elbow, s_r_elbow = side_landmarks[SL.LEFT_ELBOW], side_landmarks[SL.RIGHT_ELBOW]
            s_l_wrist, s_r_wrist = side_landmarks[SL.LEFT_WRIST], side_landmarks[SL.RIGHT_WRIST]

            side_chest_y_norm = (s_l_shoulder.y + s_r_shoulder.y) / 2
            side_hip_y_norm = (s_l_hip.y + s_r_hip.y) / 2
            side_waist_y_norm = side_chest_y_norm + (side_hip_y_norm - side_chest_y_norm) * 0.6

            def _visible_x(lm_left, lm_right):
                vis_l = getattr(lm_left, "visibility", 1.0)
                vis_r = getattr(lm_right, "visibility", 1.0)
                return lm_left.x if vis_l >= vis_r else lm_right.x

            shoulder_x_norm = _visible_x(s_l_shoulder, s_r_shoulder)
            hip_x_norm = _visible_x(s_l_hip, s_r_hip)
            waist_x_norm = shoulder_x_norm + (hip_x_norm - shoulder_x_norm) * 0.6

            chest_y_px = int(side_chest_y_norm * sh)
            waist_y_px = int(side_waist_y_norm * sh)
            hip_y_px = int(side_hip_y_norm * sh)

            chest_px = measure_torso_width_at_y(mask, chest_y_px, shoulder_x_norm * sw)
            waist_px = measure_torso_width_at_y(mask, waist_y_px, waist_x_norm * sw)
            hip_px = measure_torso_width_at_y(mask, hip_y_px, hip_x_norm * sw)

            if chest_px: chest_depth_cm = chest_px * px_to_cm_side
            if waist_px: waist_depth_cm = waist_px * px_to_cm_side
            if hip_px: hip_depth_cm = hip_px * px_to_cm_side

            arm_y_positions = [s_l_elbow.y, s_r_elbow.y, s_l_wrist.y, s_r_wrist.y]
            row_tolerance = 0.03
            for target_y in (side_waist_y_norm, side_hip_y_norm):
                if any(abs(ay - target_y) < row_tolerance for ay in arm_y_positions):
                    arm_overlap_warning = True
                    break

            # NEW in v6: draw detected landmarks on the side photo too (this was
            # previously only done for the front photo).
            annotated_side = side_bgr.copy()
            mp_drawing.draw_landmarks(annotated_side, side_pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            st.image(
                cv2.cvtColor(annotated_side, cv2.COLOR_BGR2RGB),
                caption="Side view captured (landmarks detected)",
                use_container_width=True,
                channels="RGB",
            )

            if side_issues:
                all_warnings.extend(f"Side photo: {msg}" for msg in side_issues)
                issue_text = "<br>".join(f"• {msg}" for msg in side_issues)
                st.markdown(
                    f'<div class="warn-box">⚠️ <b>Side photo framing notes:</b><br>{issue_text}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="good-box">✅ Side photo framing looks good.</div>', unsafe_allow_html=True)

            if arm_overlap_warning:
                all_warnings.append(
                    "Side photo: your arm is hanging close to your torso at waist/hip height."
                )
                st.warning(
                    "🫲 Your arm looks like it's hanging close to your torso at waist/hip "
                    "height in the side photo. We've done our best to measure the torso "
                    "only, but for the most accurate depth, retake the side photo with "
                    "your hand resting slightly forward or your elbow bent a little so "
                    "there's a visible gap between your arm and your side."
                )
        else:
            all_warnings.append("Side photo: no pose detected.")
            st.warning("No pose detected in the side photo — falling back to estimated depth for circumference.")

    # ------------------------------------------------------
    # NEW: RETAKE GUIDANCE
    # Turns every warning collected above (front/back/side framing, low
    # confidence, arm overlap) into short, actionable retake instructions,
    # e.g. "Feet not visible → Step back" or "Side view unclear → Turn 90°",
    # plus a single button to clear all captured photos and start over.
    # ------------------------------------------------------
    st.subheader("🔁 Retake Guidance")
    if all_warnings:
        retake_instructions = []
        for w in all_warnings:
            tag = map_issue_to_retake_instruction(w)
            if tag not in retake_instructions:
                retake_instructions.append(tag)
        instr_text = "<br>".join(f"• {t}" for t in retake_instructions)
        st.markdown(
            f'<div class="warn-box">📋 <b>Quick fixes for your next shot:</b><br>{instr_text}</div>',
            unsafe_allow_html=True,
        )
        if st.button("🔁 Retake Photo(s)", key="retake_guidance_button"):
            st.session_state.front_image_1 = None
            st.session_state.front_image_2 = None
            st.session_state.front_image_3 = None
            st.session_state.front1_raw = None
            st.session_state.front2_raw = None
            st.session_state.front3_raw = None
            st.session_state.side_image = None
            st.session_state.side_raw = None
            st.session_state.back_image = None
            st.session_state.back_raw = None
            st.rerun()
    else:
        st.caption("No retakes needed — every photo used looks good.")

    # ------------------------------------------------------
    # Circumference: real depth if available, else estimate
    # ------------------------------------------------------
    used_real_depth = all([chest_depth_cm, waist_depth_cm, hip_depth_cm])
    chest_ratio, waist_ratio, hip_ratio = RATIO_TABLE[gender]

    if used_real_depth:
        chest_circumference = estimate_circumference(chest_width_front_cm, chest_depth_cm)
        waist_circumference = estimate_circumference(waist_width_front_cm, waist_depth_cm)
        hip_circumference = estimate_circumference(hip_width_front_cm, hip_depth_cm)
    else:
        chest_circumference = estimate_circumference(chest_width_front_cm, chest_width_front_cm * chest_ratio)
        waist_circumference = estimate_circumference(waist_width_front_cm, waist_width_front_cm * waist_ratio)
        hip_circumference = estimate_circumference(hip_width_front_cm, hip_width_front_cm * hip_ratio)

    detected_size = detect_size(gender, chest_circumference, waist_circumference, hip_circumference)
    shape_name, shape_desc = detect_body_shape(gender, chest_circumference, waist_circumference, hip_circumference)

    # ------------------------------------------------------
    # Display results
    # ------------------------------------------------------
    st.subheader(tr("estimated_measurements"))
    display_unit = st.radio(tr("show_results_in"), ["cm", "inches"], horizontal=True, key="display_unit_choice")
    u = display_unit
    label = unit_label(u)

    results = [
        (tr("shoulder_width"), shoulder_width_cm, ""),
        (tr("full_sleeve_length"), arm_length_cm, ""),
        (tr("three_quarter_sleeve_length"), three_quarter_sleeve_length_cm, ""),
        (tr("half_sleeve_length"), half_sleeve_length_cm, ""),
        (tr("leg_length"), leg_length_cm, ""),
        (tr("waist_length"), waist_length_cm, ""),
        ((tr("bust") if gender == "Women" else tr("chest")) + " (est.)", chest_circumference, "alt"),
        # Girls also show "Chest" (not "Bust"), same as Men/Boys — handled by the check above.
        (tr("waist_est"), waist_circumference, "alt"),
        (tr("hip_est"), hip_circumference, "alt"),
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

    # ------------------------------------------------------
    # NEW: SLEEVE TYPE CALLOUT
    # Four sleeve figures are shown above, but the person only picked one
    # garment style on the measurements form. This calls out just that one
    # so they don't have to guess which of the four numbers is "theirs" -
    # and for Sleeveless there's no length at all, just a note that the
    # armhole/shoulder width is what actually matters for that cut.
    # ------------------------------------------------------
    sleeve_map = {
        "Full Sleeve": (tr("full_sleeve_length"), arm_length_cm),
        "3/4 Sleeve": (tr("three_quarter_sleeve_length"), three_quarter_sleeve_length_cm),
        "Half Sleeve": (tr("half_sleeve_length"), half_sleeve_length_cm),
    }
    selected_sleeve = st.session_state.sleeve_type
    if selected_sleeve == "Sleeveless":
        st.markdown(
            f'<div class="info-box">✂️ <b>{tr("sleeve_type")}: {tr("sleeveless")}</b> — '
            f'{tr("sleeveless_note")}</div>',
            unsafe_allow_html=True,
        )
    else:
        sel_name, sel_value_cm = sleeve_map[selected_sleeve]
        sel_value = cm_to_display(sel_value_cm, u)
        st.markdown(
            f"""<div class="measure-card alt">
                <div class="measure-label">✂️ {tr("sleeve_type")}: {selected_sleeve} — {sel_name}</div>
                <div class="measure-value">{sel_value:.1f} {label}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.subheader(tr("detected_size_shape"))
    size_col, shape_col = st.columns(2)
    with size_col:
        st.markdown(
            f"""<div class="measure-card">
                <div class="measure-label">{tr("estimated_size")} ({gender})</div>
                <div class="measure-value">{detected_size}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with shape_col:
        st.markdown(
            f"""<div class="measure-card alt">
                <div class="measure-label">{tr("body_shape")}</div>
                <div class="measure-value">{shape_name}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    st.caption(shape_desc)
    st.caption("Size and shape are estimates from a generic international chart and standard heuristics — actual brand sizing varies.")

    if used_real_depth:
        st.markdown(
            '<div class="info-box">✅ Chest/bust, waist, and hip figures use <b>real depth measured from your side photo</b> — much more accurate than a guessed ratio.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="info-box">ℹ️ No usable side photo was provided, so chest/waist/hip are '
            'estimated using an assumed depth ratio. Add a side-view photo above for more accurate results.</div>',
            unsafe_allow_html=True,
        )

    calib_methods = {r["calibration_method"] for r in shot_results}
    if calib_methods == {"silhouette"}:
        st.caption("📐 Height calibration: silhouette-based (v4, high consistency).")
    else:
        st.caption("📐 Height calibration: mixed/fallback method used on at least one shot — see notes above.")

    # ------------------------------------------------------
    # NEW: MEASUREMENT HISTORY
    # Saves only the numeric results (height, chest/bust, waist, hip, size,
    # body shape) plus a date/time - body photos are never written into this
    # list, only the numbers already computed above.
    # ------------------------------------------------------
    st.write("---")
    st.subheader(tr("measurement_history"))
    st.caption(
        "Save this result to compare against future ones. Only your height, "
        "chest/bust, waist, hip, size, body shape, clothing item, and sleeve type are stored — your photos are never saved."
    )

    if st.button(tr("save_to_history"), key="save_measurement_history"):
        st.session_state.measurement_history.append({
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "gender": gender,
            "clothing_item": st.session_state.clothing_item,
            "sleeve_type": st.session_state.sleeve_type,
            "kid_age": kid_age if category == "Kids" else None,
            "height_cm": round(height_cm, 1),
            "chest_cm": round(chest_circumference, 1),
            "waist_cm": round(waist_circumference, 1),
            "hip_cm": round(hip_circumference, 1),
            "half_sleeve_cm": round(half_sleeve_length_cm, 1),
            "three_quarter_sleeve_cm": round(three_quarter_sleeve_length_cm, 1),
            "waist_length_cm": round(waist_length_cm, 1),
            "size": detected_size,
            "body_shape": shape_name,
        })
        st.success("✅ Saved to measurement history (this session).")

    if st.session_state.measurement_history:
        n = len(st.session_state.measurement_history)
        st.caption(f"{n} saved entr{'y' if n == 1 else 'ies'} this session.")

        history_unit = st.radio(tr("show_history_in"), ["cm", "inches"], horizontal=True, key="history_unit_choice")
        hu = history_unit
        hu_label = unit_label(hu)

        table_rows = [
            {
                "Date": entry["date"],
                "Gender": entry["gender"],
                "For": entry.get("clothing_item") or "-",
                "Sleeve": entry.get("sleeve_type") or "-",
                f"Height ({hu_label})": round(cm_to_display(entry["height_cm"], hu), 1),
                f"Chest/Bust ({hu_label})": round(cm_to_display(entry["chest_cm"], hu), 1),
                f"Waist ({hu_label})": round(cm_to_display(entry["waist_cm"], hu), 1),
                f"Hip ({hu_label})": round(cm_to_display(entry["hip_cm"], hu), 1),
                f"Half Sleeve ({hu_label})": round(cm_to_display(entry.get("half_sleeve_cm", 0), hu), 1) if entry.get("half_sleeve_cm") is not None else "-",
                f"3/4 Sleeve ({hu_label})": round(cm_to_display(entry.get("three_quarter_sleeve_cm", 0), hu), 1) if entry.get("three_quarter_sleeve_cm") is not None else "-",
                f"Waist Length ({hu_label})": round(cm_to_display(entry.get("waist_length_cm", 0), hu), 1) if entry.get("waist_length_cm") is not None else "-",
                "Size": entry["size"],
                "Body Shape": entry["body_shape"],
            }
            for entry in st.session_state.measurement_history
        ]
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        if len(st.session_state.measurement_history) >= 2:
            st.markdown(f"**{tr('compare_entries')}**")
            options = [f"{i + 1}. {e['date']}" for i, e in enumerate(st.session_state.measurement_history)]
            cmp_c1, cmp_c2 = st.columns(2)
            with cmp_c1:
                choice_a = st.selectbox(tr("older_entry"), options, index=0, key="compare_entry_a")
            with cmp_c2:
                choice_b = st.selectbox(tr("newer_entry"), options, index=len(options) - 1, key="compare_entry_b")

            entry_a = st.session_state.measurement_history[options.index(choice_a)]
            entry_b = st.session_state.measurement_history[options.index(choice_b)]

            def _compare_field(field):
                a = cm_to_display(entry_a[field], hu)
                b = cm_to_display(entry_b[field], hu)
                diff = b - a
                arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "＝")
                return a, b, diff, arrow

            for field, name in [
                ("height_cm", "Height"),
                ("chest_cm", "Chest/Bust"),
                ("waist_cm", "Waist"),
                ("hip_cm", "Hip"),
                ("half_sleeve_cm", "Half Sleeve"),
                ("waist_length_cm", "Waist Length"),
            ]:
                if field not in entry_a or field not in entry_b or entry_a.get(field) is None or entry_b.get(field) is None:
                    continue
                a, b, diff, arrow = _compare_field(field)
                st.markdown(
                    f"""<div class="measure-card">
                        <div class="measure-label">{name}</div>
                        <div class="measure-value">{a:.1f} → {b:.1f} {hu_label} &nbsp; {arrow} {abs(diff):.1f} {hu_label}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            if entry_a["size"] != entry_b["size"]:
                st.info(f"Size changed: {entry_a['size']} → {entry_b['size']}")
            if entry_a["body_shape"] != entry_b["body_shape"]:
                st.info(f"Body shape changed: {entry_a['body_shape']} → {entry_b['body_shape']}")

        if st.button(tr("clear_history"), key="clear_measurement_history"):
            st.session_state.measurement_history = []
            st.rerun()
    else:
        st.caption("No saved measurements yet this session.")

    st.markdown(
        """<div class="info-box">ℹ️ Measurement history is only kept for this session right now — no
        backend/database is connected yet, so it resets when you close the app. Photos are never part
        of this history, only the numeric results shown above.</div>""",
        unsafe_allow_html=True,
    )

    st.write("---")
    show_feedback_section()



# ============================================================
# SMARTMEASURE OPENING SLIDE (added without changing the app logic)
# ============================================================
_SMARTMEASURE_DIR = os.path.dirname(os.path.abspath(__file__))
_SMARTMEASURE_ASSETS = os.path.join(_SMARTMEASURE_DIR, "smartmeasure_assets")
_SMARTMEASURE_LOGO_CANDIDATES = [
    os.path.join(_SMARTMEASURE_ASSETS, "logo.png"),
    os.path.join(_SMARTMEASURE_ASSETS, "logo.jpg"),
    os.path.join(_SMARTMEASURE_ASSETS, "logo.jpeg"),
    os.path.join(_SMARTMEASURE_ASSETS, "smartmeasure_logo.png"),
    os.path.join(_SMARTMEASURE_ASSETS, "smartmeasure_logo.jpg"),
]

def _smartmeasure_logo_path():
    for _p in _SMARTMEASURE_LOGO_CANDIDATES:
        if os.path.exists(_p):
            return _p
    return None

_SMARTMEASURE_CSR_LOGO = os.path.join(
    _SMARTMEASURE_ASSETS, "csr_partner_logos.png"
)

def page_smartmeasure_intro():
    """SmartMeasure animated opening page."""
    bg_path = os.path.join(_SMARTMEASURE_ASSETS, "smartmeasure_logo.jpeg")
    csr_path = os.path.join(_SMARTMEASURE_ASSETS, "csr_banner.png")

    def image_uri(path):
        if not os.path.exists(path):
            return ""
        ext = os.path.splitext(path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        with open(path, "rb") as f:
            return f"data:{mime};base64," + base64.b64encode(f.read()).decode("utf-8")

    bg_uri = image_uri(bg_path)
    csr_uri = image_uri(csr_path)

    st.markdown("""
    <style>
    #MainMenu, header, footer {visibility:hidden;}
    .block-container {padding-top:0!important; padding-bottom:0!important; max-width:100%!important;}

    .sm-page {
        min-height:100vh; display:flex; align-items:center; justify-content:center;
        padding:20px; box-sizing:border-box; overflow:hidden;
        background:
          radial-gradient(circle at 10% 15%,rgba(45,190,220,.14),transparent 28%),
          radial-gradient(circle at 90% 80%,rgba(130,70,210,.14),transparent 30%),
          linear-gradient(135deg,#f9fcff,#f7f4ff 52%,#f6fffb);
    }

    .sm-slide {
        width:min(1180px,96vw); min-height:88vh; position:relative; overflow:hidden;
        border-radius:30px; border:1px solid rgba(90,76,120,.13);
        background:rgba(255,255,255,.84);
        box-shadow:0 28px 80px rgba(57,44,83,.16);
        backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
        display:flex; flex-direction:column; justify-content:space-between;
        animation:sm-card-in .75s ease-out both;
    }

    .sm-watermark {
        position:absolute; width:min(520px,70vw); height:min(520px,70vw);
        left:50%; top:43%; transform:translate(-50%,-50%);
        object-fit:contain; opacity:.12; pointer-events:none;
        animation:sm-watermark-in 1.1s ease-out both;
    }

    .sm-content {position:relative;z-index:2;text-align:center;padding:clamp(48px,8vh,82px) 28px 18px;}
    .sm-kicker {font-size:12px;font-weight:800;letter-spacing:.22em;color:#695d78!important;margin-bottom:12px;animation:sm-rise .7s .12s both;}
    .sm-title {
        margin:0!important;font-size:clamp(52px,8vw,92px)!important;line-height:.96!important;
        font-weight:900!important;letter-spacing:-.045em!important;
        background:linear-gradient(90deg,#173b80,#5b28a7,#0d8094);
        -webkit-background-clip:text;background-clip:text;color:transparent!important;
        animation:sm-rise .8s .22s both;
    }
    .sm-subtitle {margin-top:16px;font-size:clamp(18px,2.2vw,28px);font-weight:700;color:#343246!important;animation:sm-rise .8s .34s both;}
    .sm-description {width:min(760px,90vw);margin:20px auto 0;font-size:clamp(14px,1.4vw,18px);line-height:1.75;color:#5d5868!important;animation:sm-rise .8s .46s both;}
    .sm-hint {margin-top:20px;font-size:12px;color:#777080!important;animation:sm-rise .8s .58s both;}
    .sm-footer {position:relative;z-index:3;width:100%;text-align:center;padding:0 22px 24px;animation:sm-bottom-in .9s .62s both;}
    .sm-branding-label {color:#71677a!important;font-size:10px;font-weight:900;letter-spacing:.20em;margin-bottom:7px;}
    .sm-csr-image {display:block;width:min(980px,94vw);max-height:108px;height:auto;object-fit:contain;margin:0 auto;border-radius:7px;}
    .sm-csr {margin-top:8px;color:#5d5565!important;font-size:12px;line-height:1.5;}
    .sm-csr b {color:#393241!important;}

    @keyframes sm-card-in {from{opacity:0;transform:translateY(28px) scale(.975)}to{opacity:1;transform:translateY(0) scale(1)}}
    @keyframes sm-watermark-in {from{opacity:0;transform:translate(-50%,-50%) scale(.90)}to{opacity:.12;transform:translate(-50%,-50%) scale(1)}}
    @keyframes sm-rise {from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
    @keyframes sm-bottom-in {from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}

    div[data-testid="stButton"] {display:flex;justify-content:center;position:relative;z-index:20;margin-top:-72px;margin-bottom:35px;}
    div[data-testid="stButton"]>button {
        border:0!important;border-radius:999px!important;padding:12px 34px!important;
        font-size:16px!important;font-weight:800!important;color:white!important;
        background:linear-gradient(90deg,#f7c1d9,#f9d4e5)!important;
        box-shadow:0 12px 28px rgba(69,46,133,.24)!important;
        transition:transform .2s ease,box-shadow .2s ease!important;
    }
    div[data-testid="stButton"]>button:hover {transform:translateY(-2px) scale(1.02);box-shadow:0 16px 34px rgba(69,46,133,.30)!important;}

    @media(max-width:700px){
        .sm-slide{min-height:92vh;border-radius:22px}.sm-content{padding-top:40px}
        .sm-watermark{width:360px;height:360px;top:40%}.sm-csr-image{max-height:75px}
    }
    </style>
    """, unsafe_allow_html=True)

    description = (
        "SmartMeasure is an AI-based body measurement system that uses "
        "computer vision to estimate body measurements automatically. "
        "The system is designed to make body measurement faster, simple, "
        "and convenient using a camera-based interface."
    )

    st.markdown(
        f"""
        <div class="sm-page">
          <div class="sm-slide">
            <img class="sm-watermark" src="{bg_uri}" alt="SmartMeasure logo">
            <div class="sm-content">
              <div class="sm-kicker">AI • COMPUTER VISION • BODY MEASUREMENT</div>
              <h1 class="sm-title">SmartMeasure</h1>
              <div class="sm-subtitle">An AI-Based Body Measurement System</div>
              <div class="sm-description">{description}</div>
              <div class="sm-hint">Click below to continue to the SmartMeasure application</div>
            </div>
            <div class="sm-footer">
              <div class="sm-branding-label">CSR PARTNERS</div>
              <img class="sm-csr-image" src="{csr_uri}" alt="CSR partners">
              <div class="sm-csr">
                CSR by <b>Microsoft</b>, <b>SAP</b>, and <b>APSSDC</b>
                &nbsp;•&nbsp; Implemented by <b>Edunet Foundation</b>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("✨  Enter SmartMeasure", key="smartmeasure_intro_enter"):
        st.session_state.smartmeasure_intro_seen = True
        st.rerun()


# APP ROUTER (unchanged from v3)
# ============================================================
# Language selector rendered first, at the very entrance of the app, before
# the intro slide / login / everything else - so it's the first control the
# user sees and it applies to every page from that point on.
render_language_selector()

if not st.session_state.smartmeasure_intro_seen:
    page_smartmeasure_intro()
elif not st.session_state.logged_in:
    page_login()
else:
    st.sidebar.markdown(f"<div class='pill'>{tr('app_pill')}</div>", unsafe_allow_html=True)
    st.sidebar.write("")
    nav_labels = [tr("nav_home"), tr("nav_measurements")]
    nav_choice = st.sidebar.radio(tr("navigate"), nav_labels)
    st.sidebar.write("---")
    if st.sidebar.button(tr("log_out")):
        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.session_state.otp_sent = False
        st.session_state.generated_otp = None
        st.rerun()

    if nav_choice == nav_labels[0]:
        page_home()
    else:
        page_measurements()
