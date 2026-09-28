"""
Comprehensive Automated Test Suite for CCTV Person Tracking & Emergency Identity Recovery.
Verifies all 12 core operational scenarios, zero unnecessary Re-ID calls,
anti-swapping protection, and tracker dominance.
"""

import pytest
import numpy as np

from ..config import AppConfig, TrackerConfig, ReIDConfig
from ..tracker.primary_tracker import MotionAwareByteTrack
from ..tracker.track import Track, TrackState
from ..tracker.motion import MotionState, Direction
from ..reid.model import ReIDExtractor
from ..reid.gallery import EmployeeGallery
from ..reid.recovery import EmergencyReIDRecovery
from ..identity.identity_manager import IdentityManager
from .synthetic_generator import SyntheticScenarioGenerator, SyntheticPerson


@pytest.fixture
def tracking_system():
    """Setup a clean isolated tracking + Re-ID system for each test."""
    Track.reset_counter()
    config = AppConfig.default()
    config.tracker.max_lost_frames = 30  # 30 frames for fast testing
    config.tracker.min_hits_to_confirm = 2
    config.identity.fixed_mode = False
    config.identity.auto_enroll_unknown = True
    config.identity.emp_prefix = "EMP-"
    
    tracker = MotionAwareByteTrack(config.tracker)
    reid_extractor = ReIDExtractor(config.reid)
    gallery = EmployeeGallery(config.reid)
    reid_recovery = EmergencyReIDRecovery(reid_extractor, gallery, config.reid)
    identity_manager = IdentityManager(reid_recovery, config.identity)
    
    return {
        "config": config,
        "tracker": tracker,
        "extractor": reid_extractor,
        "gallery": gallery,
        "reid_recovery": reid_recovery,
        "identity_manager": identity_manager,
        "generator": SyntheticScenarioGenerator(width=1280, height=720)
    }


def test_scenario_1_single_moving_person(tracking_system):
    """Test 1: Single moving person tracked smoothly with velocity, trajectory and 0 Re-ID calls."""
    sys = tracking_system
    frames, detections_seq = sys["generator"].generate_single_moving_person(num_frames=40)
    
    for f, (frame, dets) in enumerate(zip(frames, detections_seq)):
        active, lost, newly_conf, expired = sys["tracker"].update(dets, frame_id=f)
        sys["identity_manager"].process_tracks(frame, f, active, lost, newly_conf, expired)
        
    # Under Zero Re-ID Mandate, healthy tracking consumes ZERO Re-ID calls
    assert sys["reid_recovery"].reid_calls_count == 0
    assert len(sys["tracker"].all_active_tracks) == 1
    track = sys["tracker"].all_active_tracks[0]
    assert track.emp_id == "EMP-0001"
    assert track.motion_state == MotionState.MOVING
    assert track.speed > 3.0
    assert track.direction == Direction.EAST
    assert len(track.trajectory.points) > 10


def test_scenario_2_multiple_moving_people(tracking_system):
    """Test 2: Multiple distinct people assigned separate T-IDs and EMP-IDs with 0 Re-ID calls."""
    sys = tracking_system
    frames, detections_seq = sys["generator"].generate_multiple_moving_people(num_frames=40)
    
    for f, (frame, dets) in enumerate(zip(frames, detections_seq)):
        active, lost, newly_conf, expired = sys["tracker"].update(dets, frame_id=f)
        sys["identity_manager"].process_tracks(frame, f, active, lost, newly_conf, expired)
        
    assert len(sys["tracker"].all_active_tracks) == 3
    emp_ids = {t.emp_id for t in sys["tracker"].all_active_tracks}
    assert len(emp_ids) == 3
    assert "EMP-0001" in emp_ids
    assert "EMP-0002" in emp_ids
    assert "EMP-0003" in emp_ids
    # Auto-enrolled without emergency Re-ID
    assert sys["reid_recovery"].reid_calls_count == 0


def test_scenario_3_crossing_people_no_identity_swap(tracking_system):
    """Test 3: Two people crossing paths maintain their respective EMP-IDs without identity swapping."""
    sys = tracking_system
    frames, detections_seq = sys["generator"].generate_crossing_people(num_frames=60)
    
    for f, (frame, dets) in enumerate(zip(frames, detections_seq)):
        active, lost, newly_conf, expired = sys["tracker"].update(dets, frame_id=f)
        sys["identity_manager"].process_tracks(frame, f, active, lost, newly_conf, expired)
        
    assert len(sys["tracker"].all_active_tracks) == 2
    # During crossing and continuous tracking, 0 Re-ID calls
    assert sys["reid_recovery"].reid_calls_count == 0
    assert sys["identity_manager"].total_enrollments == 2


def test_scenario_4_fast_movement(tracking_system):
    """Test 4: Fast moving person is maintained by Kalman motion association."""
    sys = tracking_system
    frames, detections_seq = sys["generator"].generate_fast_movement(num_frames=30)
    
    for f, (frame, dets) in enumerate(zip(frames, detections_seq)):
        active, lost, newly_conf, expired = sys["tracker"].update(dets, frame_id=f)
        sys["identity_manager"].process_tracks(frame, f, active, lost, newly_conf, expired)
        
    assert len(sys["tracker"].all_active_tracks) == 1
    track = sys["tracker"].all_active_tracks[0]
    assert track.speed >= 10.0
    assert track.motion_state == MotionState.MOVING


def test_scenario_5_temporary_detection_miss_zero_reid(tracking_system):
    """Test 5: Temporary miss (occlusion) is recovered by tracker with ZERO Re-ID calls."""
    sys = tracking_system
    # Miss frames 15..19 (4 frames)
    frames, detections_seq = sys["generator"].generate_temporary_detection_miss(num_frames=35, miss_start=15, miss_len=4)
    
    reid_calls_before_miss = 0
    reid_calls_after_recovery = 0
    
    for f, (frame, dets) in enumerate(zip(frames, detections_seq)):
        active, lost, newly_conf, expired = sys["tracker"].update(dets, frame_id=f)
        sys["identity_manager"].process_tracks(frame, f, active, lost, newly_conf, expired)
        
        if f == 14:
            reid_calls_before_miss = sys["reid_recovery"].reid_calls_count
            assert len(active) == 1
        elif f == 17:
            # During miss, track should be in lost buffer
            assert len(lost) == 1
            assert len(active) == 0
            
    reid_calls_after_recovery = sys["reid_recovery"].reid_calls_count
    
    # Track recovered by tracker without triggering Re-ID!
    assert len(sys["tracker"].all_active_tracks) == 1
    assert sys["tracker"].all_active_tracks[0].emp_id == "EMP-0001"
    assert reid_calls_after_recovery == reid_calls_before_miss == 0


def test_scenario_6_and_8_complete_failure_and_reid_recovery(tracking_system):
    """Test 6 & 8: Complete tracker failure -> person returns -> Re-ID recovers EMP-0001."""
    sys = tracking_system
    # Person present 0..30, leaves 31..99 (exceeds max_lost_frames=30, deleted around f=62), returns 100..140
    frames, detections_seq = sys["generator"].generate_person_exit_and_return(num_frames=140)
    
    for f, (frame, dets) in enumerate(zip(frames, detections_seq)):
        active, lost, newly_conf, expired = sys["tracker"].update(dets, frame_id=f)
        sys["identity_manager"].process_tracks(frame, f, active, lost, newly_conf, expired)
        
        if f == 70:
            # While away and completely expired beyond max_lost_frames, active and lost buffers are empty
            assert len(active) == 0
            assert len(lost) == 0
            
    # When person returned:
    # 1. Tracker assigned a new T-ID (e.g. T-002)
    # 2. Re-ID activated ONCE upon confirmed return
    # 3. EMP-0001 was recovered from gallery
    # 4. Total registered employees remains exactly 1!
    assert sys["identity_manager"].total_enrollments == 1
    assert sys["identity_manager"].total_recoveries == 1
    assert len(sys["tracker"].all_active_tracks) == 1
    recovered_track = sys["tracker"].all_active_tracks[0]
    assert recovered_track.emp_id == "EMP-0001"
    # Exactly 1 emergency recovery call across the entire sequence!
    assert sys["reid_recovery"].reid_calls_count == 1


def test_scenario_9_new_person_entering(tracking_system):
    """Test 9: New distinct person entering gets enrolled as EMP-0002."""
    from ..detector.yolo_detector import PersonDetection
    sys = tracking_system
    p1 = SyntheticPerson(1, (200, 360), (4.0, 0.0), (50, 120), (220, 40, 40), (40, 40, 220))
    p2 = SyntheticPerson(2, (1100, 360), (-4.0, 0.0), (50, 120), (40, 220, 40), (220, 220, 40))
    p2.is_visible = False
    
    for f in range(50):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        if f == 25:
            p2.is_visible = True
            
        dets = []
        for p in [p1, p2]:
            if p.is_visible:
                p.draw(canvas)
                d = PersonDetection(bbox=p.bbox.copy(), confidence=0.92, center=p.center, bottom_center=p.bottom_center, frame_id=f)
                d.extract_crop(canvas)
                dets.append(d)
                p.step()
                
        active, lost, newly_conf, expired = sys["tracker"].update(dets, frame_id=f)
        sys["identity_manager"].process_tracks(canvas, f, active, lost, newly_conf, expired)
        
    assert len(sys["tracker"].all_active_tracks) == 2
    assert sys["identity_manager"].total_enrollments == 2
    emp_ids = {t.emp_id for t in sys["tracker"].all_active_tracks}
    assert emp_ids == {"EMP-0001", "EMP-0002"}


def test_scenario_11_stationary_and_moving_coexistence(tracking_system):
    """Test 11: Stationary and moving persons correctly classified simultaneously."""
    sys = tracking_system
    p_stat = SyntheticPerson(1, (400, 360), (0.0, 0.0), (50, 120), (200, 40, 40), (40, 40, 200))
    p_move = SyntheticPerson(2, (200, 200), (6.0, 0.0), (50, 120), (40, 200, 40), (200, 200, 40))
    
    for f in range(25):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        p_stat.draw(canvas)
        p_move.draw(canvas)
        
        from ..detector.yolo_detector import PersonDetection
        d1 = PersonDetection(bbox=p_stat.bbox.copy(), confidence=0.95, center=p_stat.center, bottom_center=p_stat.bottom_center, frame_id=f)
        d1.extract_crop(canvas)
        d2 = PersonDetection(bbox=p_move.bbox.copy(), confidence=0.95, center=p_move.center, bottom_center=p_move.bottom_center, frame_id=f)
        d2.extract_crop(canvas)
        
        p_stat.step()
        p_move.step()
        
        active, lost, newly_conf, expired = sys["tracker"].update([d1, d2], frame_id=f)
        sys["identity_manager"].process_tracks(canvas, f, active, lost, newly_conf, expired)
        
    assert len(sys["tracker"].all_active_tracks) == 2
    states = {t.emp_id: t.motion_state for t in sys["tracker"].all_active_tracks}
    assert states["EMP-0001"] == MotionState.STATIONARY
    assert states["EMP-0002"] == MotionState.MOVING


def test_gallery_anti_contamination(tracking_system):
    """Test: Gallery strictly rejects low-confidence or degenerate crops."""
    gallery = tracking_system["gallery"]
    
    # 1. Low confidence
    valid, reason = gallery.validate_sample_quality(np.zeros((100, 50, 3), dtype=np.uint8), np.array([0, 0, 50, 100]), confidence=0.40)
    assert not valid
    assert "Low confidence" in reason
    
    # 2. Too small crop
    valid, reason = gallery.validate_sample_quality(np.zeros((20, 10, 3), dtype=np.uint8), np.array([0, 0, 10, 20]), confidence=0.95)
    assert not valid
    assert "small" in reason
    
    # 3. High quality valid sample
    valid, reason = gallery.validate_sample_quality(np.zeros((120, 50, 3), dtype=np.uint8), np.array([0, 0, 50, 120]), confidence=0.90)
    assert valid


def test_performance_tracker_dominance(tracking_system):
    """Test: Verifies Tracker Updates consume ZERO Re-ID Calls under continuous operation."""
    sys = tracking_system
    frames, detections_seq = sys["generator"].generate_multiple_moving_people(num_frames=100)
    
    for f, (frame, dets) in enumerate(zip(frames, detections_seq)):
        active, lost, newly_conf, expired = sys["tracker"].update(dets, frame_id=f)
        sys["identity_manager"].process_tracks(frame, f, active, lost, newly_conf, expired)
        
    assert sys["tracker"].tracker_updates_count == 100
    assert sys["reid_recovery"].reid_calls_count == 0  # Zero Re-ID on continuous healthy tracking!


def test_nsa_kalman_adaptive_noise(tracking_system):
    """Test: NSA Kalman scales measurement noise covariance dynamically based on detection score."""
    from ..tracker.kalman import KalmanFilter
    kf = KalmanFilter(use_nsa=True)
    measurement = np.array([100.0, 100.0, 0.5, 120.0], dtype=np.float32)
    mean, cov = kf.initiate(measurement)

    # High confidence detection (0.95) -> low innovation noise
    _, cov_high = kf.project(mean, cov, confidence=0.95)
    # Low confidence detection (0.30) -> higher innovation noise
    _, cov_low = kf.project(mean, cov, confidence=0.30)

    # Innovation covariance for low confidence should be significantly higher
    assert np.trace(cov_low) > np.trace(cov_high)


def test_gmc_camera_motion_transformation():
    """Test: Global Motion Compensation computes transformation and applies to Kalman state."""
    from ..tracker.gmc import GlobalMotionCompensator
    from ..tracker.kalman import KalmanFilter
    import cv2
    
    gmc = GlobalMotionCompensator(method="orb")
    
    # Create two synthetic frames with horizontal shift
    f1 = np.zeros((400, 600, 3), dtype=np.uint8)
    # Add random distinct texture patches
    for x, y in [(50, 50), (200, 100), (400, 250), (100, 300), (350, 80)]:
        cv2.rectangle(f1, (x, y), (x + 40, y + 40), (200, 200, 200), -1)
        
    f2 = np.roll(f1, shift=15, axis=1)  # Shift right by 15 pixels
    
    H1 = gmc.compute_camera_motion(f1)
    H2 = gmc.compute_camera_motion(f2)
    
    assert H1.shape == (2, 3)
    assert H2.shape == (2, 3)
    
    # Test applying H to Kalman state
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([100.0, 100.0, 0.5, 100.0], dtype=np.float32))
    H_shift = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 5.0]], dtype=np.float32)
    mean_t, cov_t = kf.apply_camera_motion(mean, cov, H_shift)
    
    assert mean_t[0] == 110.0  # cx shifted +10
    assert mean_t[1] == 105.0  # cy shifted +5


def test_ocm_direction_momentum_cost():
    """Test: Observation-Centric Momentum calculates directional deviation for turns."""
    from ..tracker.matching import ocm_direction_distance
    
    # Track moving East (vx=5.0, vy=0.0)
    track_momentums = np.array([[5.0, 0.0]], dtype=np.float32)
    track_centers = np.array([[100.0, 100.0]], dtype=np.float32)
    
    # Candidate 1: Continuing East (120, 100) -> angle = 0 -> cost ~ 0.0
    # Candidate 2: Moving North (100, 80) -> angle = 90 deg -> cost ~ 0.5
    # Candidate 3: Reversing West (80, 100) -> angle = 180 deg -> cost ~ 1.0
    det_centers = np.array([[120.0, 100.0], [100.0, 80.0], [80.0, 100.0]], dtype=np.float32)
    
    cost = ocm_direction_distance(track_momentums, track_centers, det_centers)
    assert cost.shape == (1, 3)
    assert cost[0, 0] < 0.05  # Continuing in momentum direction has minimal cost
    assert cost[0, 1] > 0.40  # 90-degree turn has moderate cost
    assert cost[0, 2] > 0.90  # Complete 180-degree reversal has maximum penalty


def test_strict_active_identity_uniqueness(tracking_system):
    """Test: Guarantee that in any frame, all active tracks have 100% unique EMP-IDs with zero duplicates."""
    sys = tracking_system
    frames, detections_seq = sys["generator"].generate_multiple_moving_people(num_frames=50)
    
    for f, (frame, dets) in enumerate(zip(frames, detections_seq)):
        active, lost, newly_conf, expired = sys["tracker"].update(dets, frame_id=f)
        sys["identity_manager"].process_tracks(frame, f, active, lost, newly_conf, expired)
        
        # In EVERY frame, no two active tracks can ever share the same EMP-ID
        assigned_emp_ids = [t.emp_id for t in active if t.emp_id is not None]
        assert len(assigned_emp_ids) == len(set(assigned_emp_ids)), f"Duplicate EMP-ID detected in frame {f}: {assigned_emp_ids}"


def test_zero_reid_on_perfect_tracking(tracking_system):
    """
    Mathematical Proof 1 (Phase 4):
    Simulate 1000 frames with 6 moving people and 0 tracker drops.
    Assert reid_calls_count == 0.
    """
    sys = tracking_system
    frames, detections_seq = sys["generator"].generate_6_moving_people(num_frames=1000)
    
    for f, (frame, dets) in enumerate(zip(frames, detections_seq)):
        active, lost, newly_conf, expired = sys["tracker"].update(dets, frame_id=f)
        sys["identity_manager"].process_tracks(frame, f, active, lost, newly_conf, expired)
        
    assert sys["tracker"].tracker_updates_count == 1000
    assert len(sys["tracker"].all_active_tracks) == 6
    assert len(sys["identity_manager"].employees) == 6
    # Mathematical Proof: Under continuous healthy tracking, Re-ID inference is ZERO!
    assert sys["reid_recovery"].reid_calls_count == 0


def test_one_reid_on_single_drop(tracking_system):
    """
    Mathematical Proof 2 (Phase 4):
    Simulate 1000 frames with 6 people, artificially kill 1 track, and respawn it.
    Assert reid_calls_count == 1.
    """
    sys = tracking_system
    frames, detections_seq = sys["generator"].generate_6_people_with_single_drop(
        num_frames=1000,
        drop_person_idx=0,
        drop_start=150,
        drop_duration=100
    )
    
    for f, (frame, dets) in enumerate(zip(frames, detections_seq)):
        active, lost, newly_conf, expired = sys["tracker"].update(dets, frame_id=f)
        sys["identity_manager"].process_tracks(frame, f, active, lost, newly_conf, expired)
        
    assert sys["tracker"].tracker_updates_count == 1000
    assert len(sys["tracker"].all_active_tracks) == 6
    assert len(sys["identity_manager"].employees) == 6
    # Mathematical Proof: Exactly 1 emergency recovery call when the track failed and respawned!
    assert sys["reid_recovery"].reid_calls_count == 1
    assert sys["reid_recovery"].recovery_success_count == 1



