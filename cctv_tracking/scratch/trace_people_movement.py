"""
Trace tracks, positions, and identities across the entire video.
Logs bounding box trajectory, velocity, and detection continuity.
"""

import sys
import os
import cv2
import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARENT_DIR = os.path.abspath(os.path.join(PROJECT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from cctv_tracking.config import AppConfig
from cctv_tracking.detector.yolo_detector import YoloDetector
from cctv_tracking.tracker.primary_tracker import MotionAwareByteTrack
from cctv_tracking.reid.model import ReIDExtractor
from cctv_tracking.reid.gallery import EmployeeGallery
from cctv_tracking.reid.recovery import EmergencyReIDRecovery
from cctv_tracking.identity.identity_manager import IdentityManager


def trace_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    config = AppConfig.default()
    config.identity.fixed_mode = True
    config.identity.max_known_employees = 6
    config.identity.disable_dynamic_gallery = True

    detector = YoloDetector(config.detector)
    tracker = MotionAwareByteTrack(config.tracker)
    reid_extractor = ReIDExtractor(config.reid)
    gallery = EmployeeGallery(config.reid)
    recovery = EmergencyReIDRecovery(reid_extractor, gallery, config.reid)
    identity_manager = IdentityManager(recovery, config.identity)

    frame_id = 0
    print(f"Tracking trace starting on {video_path}...")

    # We will log significant events: track creations, lost events, recoveries, crossovers
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        detections = detector.detect(frame, frame_id)
        (
            active_tracks,
            lost_tracks,
            newly_confirmed_tracks,
            expired_tracks,
        ) = tracker.update(detections, frame_id)

        identity_manager.process_tracks(
            frame=frame,
            frame_id=frame_id,
            active_tracks=active_tracks,
            lost_tracks=lost_tracks,
            newly_confirmed_tracks=newly_confirmed_tracks,
            expired_tracks=expired_tracks,
        )

        # Check proximity between all active tracks to detect close crossings
        for i in range(len(active_tracks)):
            for j in range(i + 1, len(active_tracks)):
                t1, t2 = active_tracks[i], active_tracks[j]
                b1, b2 = t1.to_tlbr(), t2.to_tlbr()
                # Compute IoU or distance between t1 and t2
                inter_x1 = max(b1[0], b2[0])
                inter_y1 = max(b1[1], b2[1])
                inter_x2 = min(b1[2], b2[2])
                inter_y2 = min(b1[3], b2[3])
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    iou = inter_area / (area1 + area2 - inter_area)
                    if iou > 0.3:
                        if frame_id % 30 == 0:
                            print(f"[Crossing Frame {frame_id}] HIGH OVERLAP (IoU={iou:.2f}) between {t1.tid_str} ({t1.emp_id}) and {t2.tid_str} ({t2.emp_id})")

        if frame_id % 500 == 0:
            active_str = ", ".join(f"{t.tid_str}:{t.emp_id}" for t in active_tracks)
            print(f"[Frame {frame_id:04d}] Active: [{active_str}] | Lost: {len(lost_tracks)}")

    cap.release()
    print("Trace complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Trace people movement and crossings in video")
    parser.add_argument("--video", type=str, default="sample_video.mp4", help="Path to input surveillance video")
    args = parser.parse_args()

    trace_video(args.video)
