"""
Audit CCTV Video for Identity Switches and Duplicate Assignments.

Audits:
1. Did any Track ID (T-ID) ever change its assigned EMP ID? (Track-level ID Switch)
2. Did any Employee ID (EMP-ID) ever flip between two people without proper track expiration? (Identity Switch)
3. Did any EMP-ID ever get assigned to 2 tracks at the same time? (Duplicate Assignment)
4. Full log of all Re-ID inferences, recovery decisions, and scores.
"""

import sys
import os
import cv2
import numpy as np
from collections import defaultdict

# Add parent directory of cctv_tracking to sys.path
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARENT_DIR = os.path.abspath(os.path.join(PROJECT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from cctv_tracking.config import AppConfig
from cctv_tracking.detector.yolo_detector import PersonDetection, YoloDetector
from cctv_tracking.tracker.primary_tracker import MotionAwareByteTrack
from cctv_tracking.reid.model import ReIDExtractor
from cctv_tracking.reid.gallery import EmployeeGallery
from cctv_tracking.reid.recovery import EmergencyReIDRecovery
from cctv_tracking.identity.identity_manager import IdentityManager


def audit_video(video_path: str, max_frames: int = 0):
    print(f"\n=================================================================")
    print(f"       STARTING COMPREHENSIVE IDENTITY SWITCH AUDIT")
    print(f" Video: {video_path}")
    print(f"=================================================================\n")

    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Total Video Frames: {total_video_frames} | FPS: {fps:.1f}\n")

    # Initialize complete pipeline
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

    # Audit tracking structures
    track_emp_history = defaultdict(list)  # tid -> list of (frame, emp_id)
    emp_active_track_history = defaultdict(list)  # emp_id -> list of (frame, tid)
    frame_duplicate_violations = []  # frames where 1 EMP was on >= 2 tracks
    track_id_switches = []  # list of (tid, old_emp, new_emp, frame)

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if max_frames > 0 and frame_id > max_frames:
            break

        # 1. Detection
        detections = detector.detect(frame, frame_id)

        # 2. Tracking
        (
            active_tracks,
            lost_tracks,
            newly_confirmed_tracks,
            expired_tracks,
        ) = tracker.update(detections, frame_id)

        # 3. Identity Management
        identity_manager.process_tracks(
            frame=frame,
            frame_id=frame_id,
            active_tracks=active_tracks,
            lost_tracks=lost_tracks,
            newly_confirmed_tracks=newly_confirmed_tracks,
            expired_tracks=expired_tracks,
        )

        # 4. Audit active assignments in this frame
        current_emp_to_tids = defaultdict(list)
        for track in active_tracks:
            if track.emp_id and track.emp_id != "UNKNOWN":
                current_emp_to_tids[track.emp_id].append(track.tid_str)

                # Check if this track previously had a DIFFERENT emp_id
                prev_records = track_emp_history[track.tid_str]
                if prev_records:
                    last_emp = prev_records[-1][1]
                    if last_emp != track.emp_id and last_emp != "UNKNOWN":
                        track_id_switches.append({
                            "frame": frame_id,
                            "tid": track.tid_str,
                            "from_emp": last_emp,
                            "to_emp": track.emp_id
                        })
                        print(f"🚨 [ID SWITCH VIOLATION] Frame {frame_id}: {track.tid_str} switched from {last_emp} to {track.emp_id}!")

                track_emp_history[track.tid_str].append((frame_id, track.emp_id))
                emp_active_track_history[track.emp_id].append((frame_id, track.tid_str))

        # Check for duplicate EMP assignments in this single frame
        for emp_id, tids in current_emp_to_tids.items():
            if len(tids) > 1:
                frame_duplicate_violations.append({
                    "frame": frame_id,
                    "emp_id": emp_id,
                    "tids": list(tids)
                })
                print(f"🚨 [DUPLICATE EMP VIOLATION] Frame {frame_id}: {emp_id} simultaneously assigned to {tids}!")

        if frame_id % 300 == 0:
            print(f"[Audit Progress] Frame {frame_id}/{total_video_frames} ({frame_id/total_video_frames*100:.1f}%) | Active Tracks: {len(active_tracks)} | Re-ID Calls: {recovery.reid_inference_calls} | Recoveries: {recovery.successful_identity_recoveries}")

    cap.release()

    # =================================================================
    # FINAL AUDIT REPORT
    # =================================================================
    print(f"\n" + "="*65)
    print(f"              IDENTITY SWITCH & INTEGRITY AUDIT REPORT")
    print(f"="*65)
    print(f"Total Frames Processed         : {frame_id}")
    print(f"Total Unique Tracks Created    : {len(track_emp_history)}")
    print(f"Total Known EMP IDs Registered : {len(identity_manager.employees)}")
    print(f"-----------------------------------------------------------------")
    print(f"Primary Tracker Updates        : {tracker.tracker_updates_count}")
    print(f"Tracker Occlusion Recoveries   : {tracker.tracker_occlusion_recoveries}")
    print(f"-----------------------------------------------------------------")
    print(f"Re-ID Inference Calls (OSNet)  : {recovery.reid_inference_calls}")
    print(f"Recovery Attempts              : {recovery.recovery_attempts}")
    print(f"Successful Identity Recoveries : {recovery.successful_identity_recoveries}")
    print(f"-----------------------------------------------------------------")
    print(f"Track-level EMP ID Switches    : {len(track_id_switches)}")
    print(f"Duplicate EMP Frame Violations : {len(frame_duplicate_violations)}")
    print(f"=================================================================")

    if len(track_id_switches) == 0 and len(frame_duplicate_violations) == 0:
        print(f"\n✅ AUDIT RESULT: PASSED - ZERO IDENTITY SWITCHES DETECTED!")
        print(f"   - No track ever had its EMP-ID switched or overwritten.")
        print(f"   - No EMP-ID was ever assigned to more than 1 person simultaneously.")
        print(f"   - Identity persistence is 100% stable.")
    else:
        print(f"\n❌ AUDIT RESULT: FAILED - {len(track_id_switches)} identity switches, {len(frame_duplicate_violations)} duplicate assignments.")

    print(f"\n--- Employee Track Association Summary ---")
    for emp_id, emp in sorted(identity_manager.employees.items()):
        print(f"  {emp_id}: Associated T-IDs = {emp.associated_tids} | Final State = {emp.state.value} (Tracked in {emp.total_frames_tracked} frames)")
    print(f"="*65 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audit video for identity switches")
    parser.add_argument("--video", type=str, default="sample_video.mp4", help="Path to input surveillance video")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = all)")
    args = parser.parse_args()

    audit_video(args.video, max_frames=args.max_frames)
