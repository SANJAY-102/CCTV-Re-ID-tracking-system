# CCTV Person Tracking & Physical Identity Re-ID System

A robust, production-grade surveillance person tracking and emergency identity recovery system. Designed to maintain continuous physical identity across occlusions, crossing paths, posture shifts, and temporary disappearance without false ID-swapping.

> **Foundational Core Principle:**  
> ### *"TRACK FIRST. IDENTIFY ONLY WHEN TRACKING COMPLETELY FAILS."*

---

## 🌟 Key Highlights

- **Anti-ID Swapping Engine**: Strict 1-to-1 physical identity ownership ensures two crossing individuals never swap badges or employee profiles.
- **Motion-Aware Tracking (99%+ of execution)**: High-frequency ByteTrack + 8-state Kalman Filter with velocity vectors, directional compass, and trajectory tracing.
- **Emergency Deep Re-ID (<1% of execution)**: OSNet feature extractor is invoked only upon complete tracking failure, matching against multi-sample galleries with similarity gating.
- **Dual Runtime Ecosystem**:
  1. **Browser In-Place Web App (`index.html`)**: Complete zero-server tracking running directly in the browser via WebGPU / ONNX Runtime Web.
  2. **Python Surveillance Engine (`cctv_tracking/`)**: High-performance CLI backend for CCTV video files, RTSP streams, and live webcams.

---

## 📂 Repository Structure

```
├── index.html                   # Interactive browser-based CCTV Re-ID tracker (WebGPU / ONNX Runtime)
├── webm-muxer.min.js            # Client-side video encoding helper
├── STORY_OF_REID_ENGINE.md      # The journey and engineering story behind the system
├── .gitignore                   # Excludes private video footage, local models, and caches
│
└── cctv_tracking/               # Python CCTV Tracking Backend
    ├── main.py                  # CLI pipeline entrypoint (video / webcam / synthetic benchmark)
    ├── config.py                # Strongly typed dataclasses for detector, tracker, Re-ID & visualization
    ├── requirements.txt         # Python dependencies
    ├── README.md                # Python engine documentation
    ├── detector/                # YOLO Person Detector (GPU CUDA / CPU fallback)
    ├── tracker/                 # Motion-Aware ByteTrack, Kalman Filter & trajectory management
    ├── reid/                    # OSNet deep visual feature extraction & gallery management
    ├── identity/                # 1-to-1 identity manager & dormant memory bank
    ├── database/                # Persistent SQLite/JSON employee registry
    ├── visualization/           # Real-time HUD, bounding boxes, bottom-center tracking points
    ├── scratch/                 # Identity switch audit & path analysis scripts
    └── tests/                   # Automated pytest suite & synthetic test scenarios
```

---

## 🚀 Quick Start

### 1. Browser Application (WebGPU / ONNX)
Simply open `index.html` in any modern web browser (Chrome, Edge, Brave):
- Run directly or host via **GitHub Pages**.
- Load any video file locally or switch to webcam mode.
- All AI processing executes 100% locally on your machine—no video is uploaded to the cloud.

### 2. Python Surveillance Engine
Ensure Python 3.10+ is installed:

```bash
cd cctv_tracking
pip install -r requirements.txt
```

#### Run on a Video File:
```bash
python -m cctv_tracking.main --video /path/to/cctv_video.mp4 --save-output output_tracked.mp4
```

#### Run with Webcam:
```bash
python -m cctv_tracking.main --webcam 0
```

#### Run Synthetic Verification Benchmark:
```bash
python -m cctv_tracking.main --synthetic crossing
```

#### Run Automated Test Suite:
```bash
pytest cctv_tracking/tests/ -v
```

---

## 📖 The Story Behind the Engine
Read [STORY_OF_REID_ENGINE.md](STORY_OF_REID_ENGINE.md) for the story of how this system was engineered, challenges encountered with chair shrinking and path brushing, and our physical common-sense solutions.

---

## 🛡️ License & Privacy
- **Privacy First**: Designed for surveillance compliance; video frames are processed locally and never transmitted across the network.
