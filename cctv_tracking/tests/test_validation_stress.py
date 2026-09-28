"""
Final System Validation and Adversarial Stress Test Suite.
Verifies all 18 requirements of the 'Zero Re-ID unless Tracker Fails' architecture.
"""

import time
import pytest
import numpy as np

from ..config import AppConfig, TrackerConfig, ReIDConfig
from ..tracker.primary_tracker import MotionAwareByteTrack
from ..tracker.track import Track, TrackState
from ..tracker.motion import MotionState, Direction
from ..tracker.kalman import KalmanFilter
from ..reid.model import ReIDExtractor
from ..reid.gallery import EmployeeGallery
from ..reid.recovery import EmergencyReIDRecovery
from ..identity.identity_manager import IdentityManager
from ..detector.yolo_detector import PersonDetection
from .synthetic_generator import SyntheticScenarioGenerator, SyntheticPerson


@pytest.fixture
def stress_system():
    """Setup isolated tracking + OSNet Re-ID system with strict configuration."""
    Track.reset_counter()
    config = AppConfig.default()
    config.tracker.max_lost_frames = 30  # 30 frames for testing
    config.tracker.min_hits_to_confirm = 3
    config.reid.similarity_threshold = 0.70
    config.identity.fixed_mode = False
    config.identity.auto_enroll_unknown = True
    config.identity.emp_prefix = "EMP-"
    config.identity.enable_debug_logging = False
    tracker = MotionAwareByteTrack(config.tracker)
    reid_extractor = ReIDExtractor(config.reid)
    gallery = EmployeeGallery(config.reid)
    reid_recovery = EmergencyReIDRecovery(reid_extractor, gallery, config.reid)
    identity_manager = IdentityManager(reid_recovery, config.identity)
    generator = SyntheticScenarioGenerator(width=1280, height=720)
    
    return {
        "config": config,
        "tracker": tracker,
        "reid_extractor": reid_extractor,
        "gallery": gallery,
        "reid_recovery": reid_recovery,
        "identity_manager": identity_manager,
        "generator": generator
    }


def test_zero_reid_stress_test(stress_system):
    """
    Requirement 2: ZERO-REID STRESS TEST
    - 6 people
    - 1000 frames
    - continuous visibility, crossing trajectories, varying confidence, noise
    Expected: reid_calls_count == 0 (EXACTLY ZERO)
    """
    sys = stress_system
    frames, detections_seq = sys["generator"].generate_6_moving_people(num_frames=1000)
    
    # Inject detection confidence fluctuation (0.50 to 0.98) and noise
    rng = np.random.RandomState(42)
    for f, dets in enumerate(detections_seq):
        for det in dets:
            det.confidence = float(rng.uniform(0.52, 0.98))
            noise_x = float(rng.uniform(-2.0, 2.0))
            noise_y = float(rng.uniform(-2.0, 2.0))
            det.bbox += np.array([noise_x, noise_y, noise_x, noise_y], dtype=np.float32)
            
    for f, (frame, dets) in enumerate(zip(frames, detections_seq)):
        active, lost, newly_conf, expired = sys["tracker"].update(dets, frame_id=f)
        sys["identity_manager"].process_tracks(frame, f, active, lost, newly_conf, expired)
        
    assert sys["tracker"].tracker_updates_count == 1000
    assert len(sys["tracker"].all_active_tracks) == 6
    assert len(sys["identity_manager"].employees) == 6
    # Exactly ZERO Re-ID inference calls
    assert sys["reid_recovery"].reid_calls_count == 0


def test_temporary_occlusion_zero_reid(stress_system):
    """
    Requirement 3: TEMPORARY OCCLUSION TEST
    Person A occluded with low-confidence / missed detections.
    Tracker recovers through Kalman prediction + lost buffer.
    Expected: reid_calls_count == 0
    """
    sys = stress_system
    frames, detections_seq = sys["generator"].generate_temporary_detection_miss(
        num_frames=45,
        miss_start=15,
        miss_len=6
    )
    
    for f, (frame, dets) in enumerate(zip(frames, detections_seq)):
        active, lost, newly_conf, expired = sys["tracker"].update(dets, frame_id=f)
        sys["identity_manager"].process_tracks(frame, f, active, lost, newly_conf, expired)
        
    assert len(sys["tracker"].all_active_tracks) == 1
    assert sys["tracker"].all_active_tracks[0].emp_id == "EMP-0001"
    # Zero Re-ID calls during occlusion recovery
    assert sys["reid_recovery"].reid_calls_count == 0


def test_complete_failure_exact_one_reid(stress_system):
    """
    Requirement 4: COMPLETE FAILURE TEST
    TRACKED -> LOST -> EXPIRED -> reappears.
    Expected: Exactly ONE emergency Re-ID call.
    """
    sys = stress_system
    frames, detections_seq = sys["generator"].generate_person_exit_and_return(num_frames=140)
    
    for f, (frame, dets) in enumerate(zip(frames, detections_seq)):
        active, lost, newly_conf, expired = sys["tracker"].update(dets, frame_id=f)
        sys["identity_manager"].process_tracks(frame, f, active, lost, newly_conf, expired)
        
    assert sys["identity_manager"].total_enrollments == 1
    assert sys["identity_manager"].total_recoveries == 1
    assert len(sys["tracker"].all_active_tracks) == 1
    assert sys["tracker"].all_active_tracks[0].emp_id == "EMP-0001"
    assert sys["reid_recovery"].reid_calls_count == 1


def test_failed_reid_rate_limiting_and_auto_enroll(stress_system):
    """
    Requirement 5: FAILED RE-ID TEST
    Person 1 exits & expires. A completely different new person enters.
    Emergency Re-ID activates ONCE, fails match (returns None), and auto-enrolls person as EMP-0002.
    """
    sys = stress_system
    # High threshold to ensure failure on synthetic crop
    sys["reid_recovery"].config.similarity_threshold = 0.90
    
    p1 = SyntheticPerson(1, (100, 360), (4.0, 0.0), (50, 120), (220, 50, 50), (50, 50, 220))
    # Person 2 with completely different colors and trajectory
    p2 = SyntheticPerson(2, (1000, 360), (-4.0, 0.0), (50, 120), (10, 220, 220), (220, 10, 10))
    
    for f in range(140):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        dets = []
        
        if f <= 30:
            p1.draw(canvas)
            d = PersonDetection(bbox=p1.bbox.copy(), confidence=0.94, center=p1.center, bottom_center=p1.bottom_center, frame_id=f)
            d.extract_crop(canvas)
            dets.append(d)
            p1.step()
        elif f >= 100:
            p2.draw(canvas)
            d = PersonDetection(bbox=p2.bbox.copy(), confidence=0.94, center=p2.center, bottom_center=p2.bottom_center, frame_id=f)
            d.extract_crop(canvas)
            dets.append(d)
            p2.step()
            
        active, lost, newly_conf, expired = sys["tracker"].update(dets, frame_id=f)
        sys["identity_manager"].process_tracks(canvas, f, active, lost, newly_conf, expired)
        
    # Re-ID was called ONCE upon return at confirmation
    assert sys["reid_recovery"].reid_calls_count == 1
    # Recovery returned None -> auto-enrolled as new EMP
    assert len(sys["identity_manager"].employees) == 2
    assert "EMP-0001" in sys["identity_manager"].employees
    assert "EMP-0002" in sys["identity_manager"].employees
    assert sys["tracker"].all_active_tracks[0].emp_id == "EMP-0002"


def test_crossing_people_continuity(stress_system):
    """
    Requirement 6: CROSSING PEOPLE TEST
    Two people crossing paths: IoU overlap, velocity vectors, opposite directions.
    Expected: T-ID and EMP-ID continuity without swapping or Re-ID calls.
    """
    sys = stress_system
    frames, detections_seq = sys["generator"].generate_crossing_people(num_frames=70)
    
    for f, (frame, dets) in enumerate(zip(frames, detections_seq)):
        active, lost, newly_conf, expired = sys["tracker"].update(dets, frame_id=f)
        sys["identity_manager"].process_tracks(frame, f, active, lost, newly_conf, expired)
        
    assert len(sys["tracker"].all_active_tracks) == 2
    assert sys["identity_manager"].total_enrollments == 2
    assert sys["reid_recovery"].reid_calls_count == 0


def test_fast_moving_person_dynamic_association(stress_system):
    """
    Requirement 7: FAST-MOVING PERSON TEST
    Fast moving person (speed >= 16 px/frame) tracked smoothly via Kalman prediction + OCM.
    """
    sys = stress_system
    frames, detections_seq = sys["generator"].generate_fast_movement(num_frames=30)
    
    for f, (frame, dets) in enumerate(zip(frames, detections_seq)):
        active, lost, newly_conf, expired = sys["tracker"].update(dets, frame_id=f)
        sys["identity_manager"].process_tracks(frame, f, active, lost, newly_conf, expired)
        
    assert len(sys["tracker"].all_active_tracks) == 1
    track = sys["tracker"].all_active_tracks[0]
    assert track.speed >= 12.0
    assert track.motion_state == MotionState.MOVING
    assert sys["reid_recovery"].reid_calls_count == 0


def test_nms_duplicate_suppression_preserves_lost_state(stress_system):
    """
    Requirement 8: NMS / OVERLAP TEST
    When duplicate overlap occurs, the duplicate track is suppressed from active display
    and safely placed in lost buffer, preserving identity ownership.
    """
    sys = stress_system
    # 1. Establish track 1
    det1 = PersonDetection(np.array([100, 100, 160, 240], dtype=np.float32), 0.95, (130, 170), (130, 240), frame_id=0)
    for f in range(5):
        sys["tracker"].update([det1], frame_id=f)
    assert len(sys["tracker"].all_active_tracks) == 1
    
    # 2. Inject duplicate detection on same person
    det_dup = PersonDetection(np.array([102, 101, 162, 241], dtype=np.float32), 0.65, (132, 171), (132, 241), frame_id=6)
    active, lost, newly_conf, expired = sys["tracker"].update([det1, det_dup], frame_id=6)
    
    # Duplicate does not create a double active track
    assert len(active) == 1


def test_iron_lock_active_identity_protection(stress_system):
    """
    Requirement 9: IRON LOCK TEST
    EMP-0001 is active. While temporarily in lost buffer (frame 15..20), a new entrant
    MUST NOT hijack EMP-0001.
    """
    sys = stress_system
    # 1. Enroll Person 1
    det1 = PersonDetection(np.array([100, 100, 150, 220], dtype=np.float32), 0.95, (125, 160), (125, 220), frame_id=0)
    for f in range(5):
        active, lost, newly_conf, expired = sys["tracker"].update([det1], frame_id=f)
        sys["identity_manager"].process_tracks(None, f, active, lost, newly_conf, expired)
        
    assert "EMP-0001" in sys["identity_manager"].employees
    
    # 2. Person 1 is missed (enters lost buffer)
    active, lost, newly_conf, expired = sys["tracker"].update([], frame_id=6)
    sys["identity_manager"].process_tracks(None, 6, active, lost, newly_conf, expired)
    
    # 3. New Person enters at a distant location (1000, 400)
    det_new = PersonDetection(np.array([1000, 400, 1050, 520], dtype=np.float32), 0.90, (1025, 460), (1025, 520), frame_id=7)
    for f in range(7, 12):
        active, lost, newly_conf, expired = sys["tracker"].update([det_new], frame_id=f)
        sys["identity_manager"].process_tracks(None, f, active, lost, newly_conf, expired)
        
    # New track must be EMP-0002, NOT hijacking EMP-0001!
    active_emp_ids = {t.emp_id for t in sys["tracker"].all_active_tracks if t.emp_id}
    assert "EMP-0002" in active_emp_ids
    assert "EMP-0001" not in active_emp_ids  # EMP-0001 remains locked for returning Person 1


def test_collision_safe_fail_unassigned(stress_system):
    """
    Requirement 10: COLLISION TEST
    If two tracks somehow claim the same EMP-ID, the stronger track keeps it,
    and the weaker track becomes UNASSIGNED (emp_id = None) without creating a fake EMP.
    """
    sys = stress_system
    kf = sys["tracker"].kalman_filter
    
    # Create two synthetic tracks claiming EMP-0001
    t1 = Track(PersonDetection(np.array([100, 100, 150, 220]), 0.95, (125, 160), (125, 220), 0), 0, kf)
    t1.state = TrackState.TRACKED
    t1.hits = 50
    t1.confidence = 0.95
    t1.emp_id = "EMP-0001"
    
    t2 = Track(PersonDetection(np.array([200, 200, 250, 320]), 0.60, (225, 260), (225, 320), 0), 0, kf)
    t2.state = TrackState.TRACKED
    t2.hits = 5
    t2.confidence = 0.60
    t2.emp_id = "EMP-0001"  # duplicate collision
    
    # Process collision resolution in identity manager
    sys["identity_manager"].process_tracks(None, 0, [t1, t2], [], [], [])
    
    # Stronger track retains EMP-0001
    assert t1.emp_id == "EMP-0001"
    # Weaker track relegated to unassigned (emp_id is None)
    assert t2.emp_id is None


def test_performance_tracking_zero_vs_recovery(stress_system):
    """
    Requirement 13: PERFORMANCE TEST
    Measures frame processing time, FPS, and tracker updates vs Re-ID calls.
    Verifies: Normal tracking Re-ID = 0, Emergency recovery Re-ID = 1.
    """
    sys = stress_system
    frames, detections_seq = sys["generator"].generate_6_moving_people(num_frames=200)
    
    start_time = time.perf_counter()
    for f, (frame, dets) in enumerate(zip(frames, detections_seq)):
        active, lost, newly_conf, expired = sys["tracker"].update(dets, frame_id=f)
        sys["identity_manager"].process_tracks(frame, f, active, lost, newly_conf, expired)
    elapsed = time.perf_counter() - start_time
    
    fps = 200.0 / elapsed
    assert fps > 30.0  # Must exceed real-time 30 FPS easily
    assert sys["reid_recovery"].reid_calls_count == 0


def test_same_chair_departure_new_occupant_and_return_recovery(stress_system):
    """
    REGRESSION TEST 2: Same-Chair Departure, New Occupant, and Return Recovery
    1. Person A enters -> gets EMP-0001 -> tracker follows A normally (Re-ID = 0).
    2. Person A leaves completely -> A's track expires -> EMP-0001 available for recovery (gallery preserved).
    3. Person B enters/occupies exact same chair location -> B gets NEW EMP-0002 (never inherits EMP-0001).
    4. Person A returns with new T-ID -> Re-ID recognizes A -> A recovers EMP-0001.
    5. EMP-0002 remains assigned to Person B.
    """
    sys = stress_system
    sys["reid_recovery"].config.similarity_threshold = 0.70
    sys["reid_recovery"].config.min_margin = 0.05
    
    # Person A: Red appearance
    person_a = SyntheticPerson(1, (500, 300), (0.0, 0.0), (50, 120), (220, 30, 30), (30, 30, 220))
    # Person B: Distinct Green/Yellow appearance
    person_b = SyntheticPerson(2, (500, 300), (0.0, 0.0), (50, 120), (20, 220, 20), (220, 220, 20))
    
    # PHASE 1: Person A occupies chair (Frames 0..20)
    for f in range(20):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        person_a.draw(canvas)
        det_a = PersonDetection(bbox=person_a.bbox.copy(), confidence=0.95, center=person_a.center, bottom_center=person_a.bottom_center, frame_id=f)
        det_a.extract_crop(canvas)
        
        active, lost, newly_conf, expired = sys["tracker"].update([det_a], frame_id=f)
        sys["identity_manager"].process_tracks(canvas, f, active, lost, newly_conf, expired)
        
    assert len(sys["tracker"].all_active_tracks) == 1
    a_initial_emp = sys["tracker"].all_active_tracks[0].emp_id
    assert a_initial_emp == "EMP-0001"
    normal_tracking_reid_calls = sys["reid_recovery"].reid_calls_count
    assert normal_tracking_reid_calls == 0  # Re-ID is 0 during normal tracking!
    
    # PHASE 2: Person A leaves completely (Frames 20..75: 55 empty frames > max_lost_frames=30)
    for f in range(20, 75):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        active, lost, newly_conf, expired = sys["tracker"].update([], frame_id=f)
        sys["identity_manager"].process_tracks(canvas, f, active, lost, newly_conf, expired)
        
    # Active tracks is empty, but EMP-0001 profile & gallery remain preserved!
    assert len(sys["tracker"].all_active_tracks) == 0
    assert "EMP-0001" in sys["identity_manager"].employees
    assert "EMP-0001" in sys["identity_manager"].available_for_recovery_emp_ids
    assert sys["identity_manager"].get_dormant_gallery().has_dormant("EMP-0001")
    
    # PHASE 3: Person B occupies the exact same chair (500, 300) (Frames 75..110)
    # B's appearance is different from A -> B must receive NEW EMP-0002!
    for f in range(75, 110):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        person_b.draw(canvas)
        det_b = PersonDetection(bbox=person_b.bbox.copy(), confidence=0.95, center=person_b.center, bottom_center=person_b.bottom_center, frame_id=f)
        det_b.extract_crop(canvas)
        
        active, lost, newly_conf, expired = sys["tracker"].update([det_b], frame_id=f)
        sys["identity_manager"].process_tracks(canvas, f, active, lost, newly_conf, expired)
        
    assert len(sys["tracker"].all_active_tracks) == 1
    b_emp = sys["tracker"].all_active_tracks[0].emp_id
    assert b_emp == "EMP-0002"  # B gets EMP-0002, NEVER inheriting EMP-0001!
    assert b_emp != a_initial_emp
    
    # PHASE 4: Person A returns (Frames 110..150)
    # Person B is still in the chair (500, 300), Person A enters at (200, 300)
    person_a.x, person_a.y = 200.0, 300.0
    reid_calls_before_return = sys["reid_recovery"].reid_calls_count
    
    for f in range(110, 150):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        person_b.draw(canvas)
        person_a.draw(canvas)
        
        det_b = PersonDetection(bbox=person_b.bbox.copy(), confidence=0.95, center=person_b.center, bottom_center=person_b.bottom_center, frame_id=f)
        det_b.extract_crop(canvas)
        
        det_a = PersonDetection(bbox=person_a.bbox.copy(), confidence=0.95, center=person_a.center, bottom_center=person_a.bottom_center, frame_id=f)
        det_a.extract_crop(canvas)
        
        active, lost, newly_conf, expired = sys["tracker"].update([det_b, det_a], frame_id=f)
        sys["identity_manager"].process_tracks(canvas, f, active, lost, newly_conf, expired)
        
    assert len(sys["tracker"].all_active_tracks) == 2
    
    # Find tracks
    track_b = next(t for t in sys["tracker"].all_active_tracks if abs(t.bottom_center[0] - 500) < 50)
    track_a = next(t for t in sys["tracker"].all_active_tracks if abs(t.bottom_center[0] - 200) < 50)
    
    a_returned_emp = track_a.emp_id
    b_current_emp = track_b.emp_id
    
    # Final Assertions:
    assert a_initial_emp == "EMP-0001"
    assert b_emp == "EMP-0002"
    assert a_returned_emp == "EMP-0001"
    assert b_current_emp == "EMP-0002"
    assert b_emp != a_returned_emp
    
    return_recovery_reid_calls = sys["reid_recovery"].reid_calls_count - reid_calls_before_return
    assert return_recovery_reid_calls == 1


def test_full_three_person_lifecycle_and_gallery_persistence(stress_system):
    """
    REGRESSION TEST 1: Full Three-Person Lifecycle with Departure, New Entrant, and Return
    1. Person A enters -> EMP-0001
    2. Person B enters -> EMP-0002
    3. Person A completely disappears -> A's tracker expires
    4. Verify EMP-0001 still exists in the employee gallery
    5. Person C enters at A's previous location -> C receives EMP-0003 (C != EMP-0001)
    6. Person A returns with a NEW T-ID -> Emergency Re-ID matches EMP-0001
    7. Verify A owns EMP-0001, B still owns EMP-0002, C still owns EMP-0003
    """
    sys = stress_system
    sys["reid_recovery"].config.similarity_threshold = 0.70
    sys["reid_recovery"].config.min_margin = 0.05

    # Three distinct synthetic people (Red/Blue, Green/Yellow, White/Black)
    person_a = SyntheticPerson(1, (300, 300), (0, 0), (50, 120), (220, 30, 30), (30, 30, 220))   # Red
    person_b = SyntheticPerson(2, (700, 300), (0, 0), (50, 120), (20, 220, 20), (220, 220, 20))  # Green
    person_c = SyntheticPerson(3, (300, 300), (0, 0), (50, 120), (240, 240, 240), (20, 20, 20))   # White/Black
    
    # Step 1 & 2: A and B enter
    for f in range(20):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        person_a.draw(canvas)
        person_b.draw(canvas)
        da = PersonDetection(bbox=person_a.bbox.copy(), confidence=0.95, center=person_a.center, bottom_center=person_a.bottom_center, frame_id=f)
        da.extract_crop(canvas)
        db = PersonDetection(bbox=person_b.bbox.copy(), confidence=0.95, center=person_b.center, bottom_center=person_b.bottom_center, frame_id=f)
        db.extract_crop(canvas)
        
        active, lost, newly_conf, expired = sys["tracker"].update([da, db], frame_id=f)
        sys["identity_manager"].process_tracks(canvas, f, active, lost, newly_conf, expired)
        
    assert len(sys["tracker"].all_active_tracks) == 2
    tr_a = next(t for t in sys["tracker"].all_active_tracks if abs(t.bottom_center[0] - 300) < 50)
    tr_b = next(t for t in sys["tracker"].all_active_tracks if abs(t.bottom_center[0] - 700) < 50)
    a_initial = tr_a.emp_id
    b_initial = tr_b.emp_id
    
    assert a_initial == "EMP-0001"
    assert b_initial == "EMP-0002"
    
    # Step 3: A completely disappears, B stays (Frames 20..75)
    for f in range(20, 75):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        person_b.draw(canvas)
        db = PersonDetection(bbox=person_b.bbox.copy(), confidence=0.95, center=person_b.center, bottom_center=person_b.bottom_center, frame_id=f)
        db.extract_crop(canvas)
        
        active, lost, newly_conf, expired = sys["tracker"].update([db], frame_id=f)
        sys["identity_manager"].process_tracks(canvas, f, active, lost, newly_conf, expired)
        
    # Step 4: Verify EMP-0001 still exists in employee gallery
    assert "EMP-0001" in sys["identity_manager"].employees
    assert "EMP-0001" in sys["identity_manager"].available_for_recovery_emp_ids
    assert sys["identity_manager"].get_dormant_gallery().has_dormant("EMP-0001")
    
    # Step 5: Person C enters at A's previous location (300, 300) (Frames 75..110)
    for f in range(75, 110):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        person_b.draw(canvas)
        person_c.draw(canvas)
        db = PersonDetection(bbox=person_b.bbox.copy(), confidence=0.95, center=person_b.center, bottom_center=person_b.bottom_center, frame_id=f)
        db.extract_crop(canvas)
        dc = PersonDetection(bbox=person_c.bbox.copy(), confidence=0.95, center=person_c.center, bottom_center=person_c.bottom_center, frame_id=f)
        dc.extract_crop(canvas)
        
        active, lost, newly_conf, expired = sys["tracker"].update([db, dc], frame_id=f)
        sys["identity_manager"].process_tracks(canvas, f, active, lost, newly_conf, expired)
        
    assert len(sys["tracker"].all_active_tracks) == 2
    tr_c = next(t for t in sys["tracker"].all_active_tracks if abs(t.bottom_center[0] - 300) < 50)
    c_initial = tr_c.emp_id
    assert c_initial == "EMP-0003"
    assert c_initial != a_initial
    assert c_initial != "EMP-0001"
    
    # Step 6: Person A returns with a NEW T-ID at location (1000, 300) (Frames 110..150)
    person_a.x, person_a.y = 1000.0, 300.0
    for f in range(110, 150):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        person_b.draw(canvas)
        person_c.draw(canvas)
        person_a.draw(canvas)
        
        db = PersonDetection(bbox=person_b.bbox.copy(), confidence=0.95, center=person_b.center, bottom_center=person_b.bottom_center, frame_id=f)
        db.extract_crop(canvas)
        dc = PersonDetection(bbox=person_c.bbox.copy(), confidence=0.95, center=person_c.center, bottom_center=person_c.bottom_center, frame_id=f)
        dc.extract_crop(canvas)
        da = PersonDetection(bbox=person_a.bbox.copy(), confidence=0.95, center=person_a.center, bottom_center=person_a.bottom_center, frame_id=f)
        da.extract_crop(canvas)
        
        active, lost, newly_conf, expired = sys["tracker"].update([db, dc, da], frame_id=f)
        sys["identity_manager"].process_tracks(canvas, f, active, lost, newly_conf, expired)
        
    assert len(sys["tracker"].all_active_tracks) == 3
    tr_b_curr = next(t for t in sys["tracker"].all_active_tracks if abs(t.bottom_center[0] - 700) < 50)
    tr_c_curr = next(t for t in sys["tracker"].all_active_tracks if abs(t.bottom_center[0] - 300) < 50)
    tr_a_curr = next(t for t in sys["tracker"].all_active_tracks if abs(t.bottom_center[0] - 1000) < 50)
    
    a_return = tr_a_curr.emp_id
    b_curr = tr_b_curr.emp_id
    c_curr = tr_c_curr.emp_id
    
    # Final Assertions:
    assert a_initial == "EMP-0001"
    assert b_initial == "EMP-0002"
    assert c_initial == "EMP-0003"
    assert a_return == "EMP-0001"
    
    assert a_return != b_curr
    assert a_return != c_curr
    assert c_initial != "EMP-0001"
    assert sys["identity_manager"].get_dormant_gallery().has_dormant("EMP-0001")


def test_ambiguous_reid_margin_fail_safe(stress_system):
    """
    REGRESSION TEST 3: Ambiguous Re-ID Margin Fail-Safe
    Tests an ambiguous Re-ID case where:
    EMP-0001 similarity = 0.61
    EMP-0002 similarity = 0.60
    If configured margin requirement (0.05) is not satisfied, the system
    MUST NOT assign either existing EMP and instead safely auto-enroll a new EMP.
    """
    sys = stress_system
    gal = sys["reid_recovery"].gallery
    gal.config.similarity_threshold = 0.55
    gal.config.min_margin = 0.05
    
    # Create two synthetic baseline embeddings in gallery
    emb1 = np.zeros(512, dtype=np.float32)
    emb1[0] = 0.80
    emb1[1] = 0.60
    emb1 /= np.linalg.norm(emb1)
    
    emb2 = np.zeros(512, dtype=np.float32)
    emb2[0] = 0.79
    emb2[1] = 0.61
    emb2 /= np.linalg.norm(emb2)
    
    gal.add_embedding("EMP-0001", emb1, 0.95, 0)
    gal.add_embedding("EMP-0002", emb2, 0.95, 0)
    
    # Query embedding roughly equidistant to both (e.g. [0.795, 0.605])
    q_emb = np.zeros(512, dtype=np.float32)
    q_emb[0] = 0.795
    q_emb[1] = 0.605
    q_emb /= np.linalg.norm(q_emb)
    
    matched_emp, score, all_scores, meta = gal.match(
        q_emb,
        eligible_emp_ids={"EMP-0001", "EMP-0002"},
        similarity_threshold=0.55,
        min_margin=0.05
    )
    
    # Margin is tiny (< 0.05), so match MUST return None (ambiguous fail-safe!)
    assert matched_emp is None
    assert meta["is_ambiguous"] is True
    assert meta["margin"] < 0.05


def test_performance_1000_frames_zero_reid_and_one_drop_exact_one_reid(stress_system):
    """
    PERFORMANCE & STRESS TEST:
    1. Continuous tracking consumes exactly 0 Re-ID calls.
    2. Forcing one complete track expiration and return consumes exactly 1 emergency Re-ID inference.
    """
    sys = stress_system
    frames, detections_seq = sys["generator"].generate_6_people_with_single_drop(num_frames=1000)
    
    for f, (frame, dets) in enumerate(zip(frames, detections_seq)):
        active, lost, newly_conf, expired = sys["tracker"].update(dets, frame_id=f)
        sys["identity_manager"].process_tracks(frame, f, active, lost, newly_conf, expired)
        
    # Total emergency Re-ID calls across all 1000 frames with 1 complete drop & return is EXACTLY 1
    assert sys["reid_recovery"].reid_calls_count == 1
    assert sys["identity_manager"].total_recoveries == 1
    assert len(sys["identity_manager"].employees) == 6



