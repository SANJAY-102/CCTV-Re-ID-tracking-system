# CCTV Person Tracking + Emergency Identity Recovery System

> **Core Foundational Principle:**
> ### *"TRACK FIRST. IDENTIFY ONLY WHEN TRACKING COMPLETELY FAILS."*

---

## 📌 Architecture & Design Principles

This project is a **ground-up Python implementation** engineered for high-performance CCTV surveillance, moving person analysis, and persistent employee tracking.

### 1. High-Frequency Tracking vs Rare Re-ID
- **Primary Tracker (High Load - 99%+ Workload)**:
  - Observation-Centric ByteTrack (OC-ByteTrack) with 8-state Kalman Filter (`[cx, cy, a, h, vx, vy, va, vh]`).
  - 2-stage bipartite Hungarian matching (High-confidence + Low-confidence detections).
  - Heavy motion analysis: Velocity vector $(v_x, v_y)$, smoothed speed, 8-compass direction (N, NE, E, SE, S, SW, W, NW), and motion states (`MOVING`, `STATIONARY`, `LOST`, `RECOVERED`).
  - Visible bottom-center tracking point `●` (`(x1+x2)/2, y2`) and continuous trajectory trail ($P_1 \to P_2 \to P_3 \dots$).
- **Emergency Re-ID Recovery (Rare Load - <1% Workload)**:
  - Re-ID **never** runs on standard frames, speed changes, direction shifts, or temporary occlusions.
  - Re-ID is invoked **strictly once** when a confirmed track experiences complete failure (lost buffer expires) and an unassociated person appears.
  - Generates 512-dim L2-normalized deep visual embeddings.
  - Matches against the Multi-Embedding Employee Gallery with strict similarity gating ($\ge 0.70$).
  - Once identity `EMP-XXXX` is recovered and linked to the new `T-ID`, **control is immediately returned to the primary tracker**.

### 2. Strict ID Separation
- `T-ID` (`T-001`, `T-002`, `T-017`): Ephemeral tracking session ID belonging to the primary tracker.
- `EMP-ID` (`EMP-0001`, `EMP-0002`): Persistent physical employee profile.
- **Anti-Swapping Lock**: Active 1-to-1 identity ownership prevents active tracks from stealing each other's `EMP-ID` during crossings.

---

## 📂 Project Structure

```
cctv_tracking/
│
├── main.py                     # CLI entrypoint for video, webcam, or synthetic benchmark
├── config.py                   # Centralized configuration dataclasses & presets
├── requirements.txt            # Python dependencies
├── README.md                   # Full documentation & usage guide
│
├── detector/
│   ├── __init__.py
│   └── yolo_detector.py        # YOLOv8 Person detector (GPU/CPU fallback)
│
├── tracker/
│   ├── __init__.py
│   ├── kalman.py               # 8-state Kalman Filter
│   ├── matching.py             # Hungarian linear sum assignment & IoU distance
│   ├── motion.py               # Velocity, direction, speed, stationary/moving classifier
│   ├── trajectory.py           # Bottom-center point tracker & trajectory queue
│   ├── track.py                # Track lifecycle state machine & T-ID generator
│   └── primary_tracker.py      # OC-ByteTrack 2-stage association engine
│
├── reid/
│   ├── __init__.py
│   ├── model.py                # 512-dim deep Re-ID feature extractor
│   ├── gallery.py              # Multi-embedding employee gallery & quality filter
│   └── recovery.py             # Emergency Re-ID recovery engine
│
├── identity/
│   ├── __init__.py
│   ├── employee.py             # Persistent Employee dataclass
│   └── identity_manager.py     # 1-to-1 ownership arbitrator & lost track memory
│
├── database/
│   ├── __init__.py
│   └── database.py             # JSON/SQLite persistent employee storage
│
├── visualization/
│   ├── __init__.py
│   └── annotations.py          # CCTV bounding boxes, ● tracking point, trail, HUD
│
└── tests/
    ├── __init__.py
    ├── synthetic_generator.py  # Realistic multi-person synthetic scenario generator
    └── test_tracking.py        # 12 automated test scenarios verifying rules
```

---

## 🚀 Installation

Ensure Python 3.10+ is installed:

```bash
cd cctv_tracking
pip install -r requirements.txt
```

*(GPU CUDA acceleration is automatically used if available; otherwise, CPU fallback is enabled seamlessly).*

---

## 🏃 Running the System

### 1. Run on CCTV / Surveillance Video File
```bash
python -m cctv_tracking.main --video path/to/cctv_footage.mp4 --save-output output_tracked.mp4
```

### 2. Run on Live Webcam
```bash
python -m cctv_tracking.main --webcam 0
```

### 3. Run Built-In Benchmark Synthetic Scenarios
You can run any of the built-in synthetic scenarios with full visual tracking, trajectory trails, and live HUD overlay:

```bash
# Crossing scenario (verifies zero ID swapping)
python -m cctv_tracking.main --synthetic crossing --save-output crossing_demo.mp4

# Person exit and return (verifies emergency Re-ID identity recovery)
python -m cctv_tracking.main --synthetic return --save-output return_demo.mp4

# Multi-person tracking
python -m cctv_tracking.main --synthetic multi --save-output multi_demo.mp4

# Fast running person
python -m cctv_tracking.main --synthetic fast --save-output fast_demo.mp4
```

---

## 🧪 Automated Test Suite

Run the comprehensive pytest suite verifying all 12 operational conditions:

```bash
python -m pytest cctv_tracking/tests/test_tracking.py -v
```

### Verified Test Cases:
1. `test_scenario_1_single_moving_person`: Continuous smooth tracking, velocity calculation, trajectory queue, and zero redundant Re-ID.
2. `test_scenario_2_multiple_moving_people`: 3 distinct tracks, 3 distinct T-IDs, 3 distinct EMP-IDs.
3. `test_scenario_3_crossing_people_no_identity_swap`: Two people crossing directly; verifies zero identity swap and zero Re-ID calls during crossing.
4. `test_scenario_4_fast_movement`: High velocity (18 px/f) movement continuity via Kalman prediction.
5. `test_scenario_5_temporary_detection_miss_zero_reid`: Temporary miss (occlusion); verifies recovery with **zero Re-ID calls**.
6. `test_scenario_6_and_8_complete_failure_and_reid_recovery`: Complete track loss $\to$ person returns $\to$ Re-ID recovers `EMP-0001` with new `T-ID`.
7. `test_scenario_9_new_person_entering`: New entrant auto-enrolled as `EMP-0002`.
8. `test_scenario_11_stationary_and_moving_coexistence`: Simultaneous stationary and moving person classification.
9. `test_gallery_anti_contamination`: Strict rejection of low-quality or tiny crops to protect gallery integrity.
10. `test_performance_tracker_dominance`: Confirms Tracker Updates $\gg$ Re-ID Calls ($>97\%$ tracker load).

---

## 📊 Live HUD & Telemetry Overlay

The on-screen HUD displays real-time statistics:
- **FPS** & Total Frames
- **Active Tracks** & **Moving Tracks**
- **Stationary Tracks** & **Lost Buffer Tracks**
- **Primary Tracker Updates: Y**
- **Emergency Re-ID Calls: X**
- **Emergency Re-ID Recoveries**
- **Total EMP Profiles Registered**
