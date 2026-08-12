# 📏 AI-Based Body Measurement Web App

An AI-powered web application that estimates body measurements from photos using pose detection — built with **OpenCV**, **MediaPipe**, and **Streamlit**.

Developed as part of the **AICW (AI Careers for Women)** program by Edunet Foundation, in collaboration with Microsoft and SAP.

---

## 🎯 What It Does

Stand in front of a camera, enter your height, and the app automatically detects your body landmarks and estimates:

- Shoulder width
- Arm length
- Leg length
- Chest, waist, and hip circumference (estimated)

An optional **side-view photo** improves circumference accuracy by measuring real body depth instead of relying on an assumed ratio.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Streamlit** | Web app interface |
| **OpenCV** | Image capture and processing |
| **MediaPipe Pose** | Detects 33 body landmarks (shoulders, hips, elbows, knees, etc.) |
| **MediaPipe Selfie Segmentation** | Extracts body silhouette from the side-view photo to measure real depth |
| **NumPy** | Numerical calculations |
| **Pillow** | Image handling |

---

## ⚙️ How It Works

1. **Calibration** — The user enters their actual height. The app measures the same height in pixels (nose to ankle) from the photo, giving a pixel-to-cm conversion ratio.
2. **Pose Detection** — MediaPipe Pose locates key body landmarks in the front-facing photo.
3. **Linear Measurements** — Pixel distances between landmarks (e.g., left shoulder to right shoulder) are converted to real-world centimeters using the calibration ratio.
4. **Circumference Estimation** — Since a single 2D photo only captures width, chest/waist/hip circumference is estimated using an elliptical approximation (Ramanujan's ellipse perimeter formula), combining width with either:
   - **Real depth** (if a side-view photo is provided, measured via body silhouette segmentation), or
   - **An assumed depth ratio** (fallback, less accurate)
5. **Unit Flexibility** — Height can be entered in cm or feet & inches; results can be displayed in cm or inches.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11 (MediaPipe does not yet support the newest Python releases)

### Installation

```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```

### Run the app

```bash
python -m streamlit run body_measurement_app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📸 Screenshots

*(Add 2-3 screenshots here of the app running with sample results)*

```
![App screenshot](screenshots/demo1.png)
```

---

## ⚠️ Known Limitations

- Circumference values (chest/waist/hip) are **estimates**, not tape-measure accurate, since they're derived from 2D photos rather than true 3D body scans.
- Accuracy depends on camera angle, lighting, and how straight the person stands.
- The side-view depth measurement assumes the person stood at roughly the same distance from the camera in both photos.
- Best results come from plain, high-contrast backgrounds and full-body visibility in frame.

---

## 🔮 Future Improvements

- Migrate from MediaPipe's legacy `solutions` API to the newer Tasks API (`PoseLandmarker`)
- Validate measurement accuracy against real tape-measure data and report error margins
- Add size-chart mapping (S/M/L/XL) for garment recommendations
- Integrate into a tailor/artisan marketplace platform for automated customer measurement intake

---

## 👩‍💻 Author

Built by Poja as part of the AICW program (Edunet Foundation × Microsoft × SAP).
Mentor: Abdul.
