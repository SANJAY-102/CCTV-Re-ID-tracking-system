"""
Test Suite for Strict Recovery Session Lifecycle & Identity Event Decoupling.

Validates exact user specifications:
1. One-Shot Recovery Session:
   Person A leaves -> expires -> returns as T008.
   Exactly 1 Re-ID inference call, 1 recovery attempt, 1 successful recovery.
   Simulating 500 subsequent frames of healthy tracking produces:
   reid_inference_calls = 1, recovery_attempts = 1, successful_identity_recoveries = 1.
   (NOT recovery_attempts = 501).
2. Unknown Seventh Person Stability:
   A seventh person enters (T007) when all 6 EMPs are occupied or no match.
   T007 -> UNKNOWN.
   Re-ID session completes; no repeated recovery attempts over 500 frames.
3. Gallery Immunity from UNKNOWN:
   UNKNOWN tracks NEVER contaminate EMP001-EMP006 galleries.
4. One-Time Identity Event Visual Lifecycle:
   Label shows RECOVERED during the event display window, then transitions to TRACKED.
"""

import pytest
import numpy as np

from cctv_tracking.config import AppConfig
from cctv_tracking.tracker.primary_tracker import MotionAwareByteTrack
from cctv_tracking.tracker.track import Track, TrackState
from cctv_tracking.tracker.kalman import KalmanFilter
from cctv_tracking.reid.model import ReIDExtractor
from cctv_tracking.reid.gallery import EmployeeGallery
from cctv_tracking.reid.recovery import EmergencyReIDRecovery
from cctv_tracking.identity.identity_manager import IdentityManager
from cctv_tracking.detector.yolo_detector import PersonDetection
from cctv_tracking.visualization.annotations import Annotator


def create_detection(cx, cy, w=50, h=100, conf=0.95, frame_id=1):
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    return PersonDetection(
        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
        confidence=conf,
        center=(cx, cy),
        bottom_center=(cx, y2),
        frame_id=frame_id
    )


@pytest.fixture
def recovery_system():
    Track.reset_counter()
    config = AppConfig.default()
    config.tracker.max_lost_frames = 20
    config.tracker.min_hits_to_confirm = 3
    config.reid.similarity_threshold = 0.55
    config.reid.min_margin = 0.05
    config.identity.fixed_mode = True
    config.identity.max_known_employees = 6
    config.identity.disable_dynamic_gallery = True

    tracker = MotionAwareByteTrack(config.tracker)
    extractor = ReIDExtractor(config.reid)
    gallery = EmployeeGallery(config.reid)
    recovery = EmergencyReIDRecovery(extractor, gallery, config.reid)
    identity_manager = IdentityManager(recovery, config.identity)
    annotator = Annotator(config.visualization)

    return {
        "config": config,
        "tracker": tracker,
        "recovery": recovery,
        "identity_manager": identity_manager,
        "annotator": annotator
    }


def test_recovery_session_single_shot_and_500_frame_stability(recovery_system):
    """
    Section 13: Person A (T001 -> EMP001) leaves, expires to DORMANT.
    Returns as T008. Exactly 1 inference, 1 attempt, 1 recovery.
    Simulate 500 subsequent frames: counters MUST NOT increment!
    """
    im: IdentityManager = recovery_system["identity_manager"]
    recovery: EmergencyReIDRecovery = recovery_system["recovery"]
    kf = KalmanFilter()

    # 1. Enroll EMP001 with T001
    d1 = create_detection(100, 100)
    t1 = Track(d1, 1, kf)
    t1.state = TrackState.TRACKED
    im.process_tracks(None, frame_id=1, active_tracks=[t1], lost_tracks=[], newly_confirmed_tracks=[t1], expired_tracks=[])
    assert t1.emp_id == "EMP001"
    assert im.is_emp_active("EMP001")

    # Enroll remaining 5 employees so that EMP001-EMP006 are all known
    for i in range(2, 7):
        det = create_detection(100 + i * 50, 100)
        t = Track(det, 1, kf)
        t.state = TrackState.TRACKED
        im.process_tracks(None, frame_id=1, active_tracks=[t], lost_tracks=[], newly_confirmed_tracks=[t], expired_tracks=[])

    assert len(im.employees) == 6

    # 2. T001 leaves and expires
    im.process_tracks(None, frame_id=50, active_tracks=[], lost_tracks=[], newly_confirmed_tracks=[], expired_tracks=[t1])
    assert not im.is_emp_active("EMP001")
    assert "EMP001" in im.dormant_emp_ids

    # Store a dummy reference embedding for EMP001 in dormant gallery
    dummy_crop = np.full((128, 64, 3), 120, dtype=np.uint8)
    dummy_emb = recovery.extractor.extract(dummy_crop)
    im._dormant_gallery.add_embedding("EMP001", dummy_emb, quality_score=0.95, frame_id=1)

    # 3. Person A returns as a newly confirmed track T008 at frame 100
    t8 = Track(create_detection(100, 100), 100, kf)
    t8.state = TrackState.TRACKED
    t8.last_crop = dummy_crop

    # Initial counters
    init_inf = recovery.reid_inference_calls
    init_att = recovery.recovery_attempts
    init_rec = recovery.successful_identity_recoveries

    # Frame 100: Recovery occurs
    im.process_tracks(None, frame_id=100, active_tracks=[t8], lost_tracks=[], newly_confirmed_tracks=[t8], expired_tracks=[])

    assert t8.emp_id == "EMP001"
    assert t8.recovery_attempted is True
    assert t8.identity_event == "RECOVERED"
    assert t8.identity_event_frame == 100
    assert t8.recovered_locked is True
    assert im.is_emp_active("EMP001")
    assert "EMP001" not in im.dormant_emp_ids

    assert recovery.reid_inference_calls == init_inf + 1
    assert recovery.recovery_attempts == init_att + 1
    assert recovery.successful_identity_recoveries == init_rec + 1

    # 4. Simulate 500 subsequent frames of T008 continuously tracked
    for f in range(101, 601):
        t8.frame_id = f
        im.process_tracks(None, frame_id=f, active_tracks=[t8], lost_tracks=[], newly_confirmed_tracks=[], expired_tracks=[])

    # Counters MUST REMAIN EXACTLY THE SAME — ZERO additional recoveries or attempts!
    assert recovery.reid_inference_calls == init_inf + 1
    assert recovery.recovery_attempts == init_att + 1
    assert recovery.successful_identity_recoveries == init_rec + 1
    assert t8.emp_id == "EMP001"
    assert t8.identity_status == "TRACKED"


def test_seventh_person_unknown_no_repeated_reid(recovery_system):
    """
    Section 14: 6 known people exist. Seventh person enters as T007.
    No dormant match -> T007 assigned UNKNOWN.
    Recovery session completed. Over 500 subsequent frames, ZERO re-ID calls!
    """
    im: IdentityManager = recovery_system["identity_manager"]
    recovery: EmergencyReIDRecovery = recovery_system["recovery"]
    kf = KalmanFilter()

    # Enroll 6 known employees
    active_tracks = []
    for i in range(1, 7):
        det = create_detection(100 + i * 50, 100)
        t = Track(det, 1, kf)
        t.state = TrackState.TRACKED
        im.process_tracks(None, frame_id=1, active_tracks=[t], lost_tracks=[], newly_confirmed_tracks=[t], expired_tracks=[])
        active_tracks.append(t)

    assert len(im.employees) == 6

    # Seventh person T007 enters at frame 50
    t7 = Track(create_detection(500, 100), 50, kf)
    t7.state = TrackState.TRACKED
    active_tracks.append(t7)

    calls_before = recovery.reid_inference_calls
    attempts_before = recovery.recovery_attempts

    # Frame 50: T007 confirmed
    im.process_tracks(None, frame_id=50, active_tracks=active_tracks, lost_tracks=[], newly_confirmed_tracks=[t7], expired_tracks=[])

    assert t7.emp_id == "UNKNOWN"
    assert t7.identity_status == "UNKNOWN"
    assert t7.recovery_attempted is True

    # Over 500 subsequent frames, no repeated Re-ID attempts
    for f in range(51, 551):
        t7.frame_id = f
        im.process_tracks(None, frame_id=f, active_tracks=active_tracks, lost_tracks=[], newly_confirmed_tracks=[], expired_tracks=[])

    assert recovery.reid_inference_calls == calls_before
    assert recovery.recovery_attempts == attempts_before
    assert t7.emp_id == "UNKNOWN"
    assert t7.identity_status == "UNKNOWN"


def test_no_gallery_update_from_unknown(recovery_system):
    """
    Section 15: An UNKNOWN track must NEVER update or contaminate the dormant gallery.
    """
    im: IdentityManager = recovery_system["identity_manager"]
    recovery: EmergencyReIDRecovery = recovery_system["recovery"]
    kf = KalmanFilter()

    # Enroll EMP001
    d1 = create_detection(100, 100)
    t1 = Track(d1, 1, kf)
    t1.state = TrackState.TRACKED
    im.process_tracks(None, frame_id=1, active_tracks=[t1], lost_tracks=[], newly_confirmed_tracks=[t1], expired_tracks=[])

    # Seventh person enters as UNKNOWN
    t_unk = Track(create_detection(200, 100), 2, kf)
    t_unk.state = TrackState.TRACKED
    t_unk.emp_id = "UNKNOWN"
    t_unk.identity_status = "UNKNOWN"

    # Frame with dummy image
    dummy_frame = np.full((480, 640, 3), 200, dtype=np.uint8)

    initial_gallery_records = dict(im._dormant_gallery._records)

    # Process 50 frames with t_unk active
    for f in range(3, 53):
        im.process_tracks(dummy_frame, frame_id=f, active_tracks=[t1, t_unk], lost_tracks=[], newly_confirmed_tracks=[], expired_tracks=[])

    # Verify no record created for "UNKNOWN"
    assert "UNKNOWN" not in im._dormant_gallery._records
    # Verify EMP001 gallery is clean
    emp1_samples = im._dormant_gallery.get_embeddings("EMP001")
    assert len(emp1_samples) <= 2


def test_visual_label_event_transition(recovery_system):
    """
    Section 16: Badges show RECOVERED only during event_display_frames,
    then automatically transition to TRACKED!
    """
    annotator: Annotator = recovery_system["annotator"]
    kf = KalmanFilter()

    t = Track(create_detection(100, 100), 100, kf)
    t.state = TrackState.TRACKED
    t.emp_id = "EMP001"
    t.identity_status = "TRACKED"
    t.identity_event = "RECOVERED"
    t.identity_event_frame = 100
    t.identity_confidence = 0.72

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # At frame 100: event active -> RECOVERED
    annotator.draw_track(frame, t, current_frame=100)
    col_100 = annotator.get_state_color(t, current_frame=100)
    assert col_100 == annotator.config.color_recovered

    # At frame 120 (20 frames later, within 45 frames) -> still RECOVERED
    col_120 = annotator.get_state_color(t, current_frame=120)
    assert col_120 == annotator.config.color_recovered

    # At frame 200 (100 frames later, past 45 frames window) -> TRACKED / stationary color!
    col_200 = annotator.get_state_color(t, current_frame=200)
    assert col_200 == annotator.config.color_stationary
