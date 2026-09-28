"""
Closed-Set 6-Person Experiment Acceptance Test Suite.

Validates the exact acceptance criteria:
1. Critical Acceptance Test:
   - 6 known people enrolled: A->EMP001, B->EMP002, C->EMP003, D->EMP004, E->EMP005, F->EMP006.
   - A leaves. T001 expires. EMP001 becomes AVAILABLE (dormant).
   - G enters A's exact old location.
   - G is compared via OSNet. Because G's appearance differs (sim < 0.55), G becomes UNKNOWN.
   - G does NOT steal EMP001 (spatial location has ZERO authority).
   - A returns as a NEW T-ID (T008).
   - A is compared via OSNet (sim > 0.55).
   - A recovers EMP001.
   - Final check: A=EMP001, B=EMP002, C=EMP003, D=EMP004, E=EMP005, F=EMP006, G=UNKNOWN.
   - All other 5 people maintain their original identities.
2. Second Test:
   - A leaves and returns. T-ID changes (T001 -> T008), but EMP001 is preserved.
3. Third Test:
   - A and B both leave. Both return. Both recover their original EMP IDs without collision.
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


def make_distinct_crop(person_id: int) -> np.ndarray:
    """Generate distinct appearance crop for each person."""
    crop = np.zeros((128, 64, 3), dtype=np.uint8)
    # Color signature unique to person
    color = (
        int((person_id * 73 + 30) % 255),
        int((person_id * 151 + 60) % 255),
        int((person_id * 223 + 90) % 255)
    )
    crop[:] = color
    return crop


@pytest.fixture
def closed_set_system():
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

    return {
        "config": config,
        "tracker": tracker,
        "recovery": recovery,
        "identity_manager": identity_manager,
    }


def test_critical_acceptance_g_enters_a_returns(closed_set_system):
    """
    CRITICAL ACCEPTANCE TEST:
    A=EMP001, B=EMP002, C=EMP003, D=EMP004, E=EMP005, F=EMP006.
    A leaves. T001 expires.
    G enters A's old location -> G must be UNKNOWN (0 location authority).
    A returns as T008 -> A must recover EMP001.
    Final: A=EMP001, B=EMP002, C=EMP003, D=EMP004, E=EMP005, F=EMP006, G=UNKNOWN.
    """
    im: IdentityManager = closed_set_system["identity_manager"]
    recovery: EmergencyReIDRecovery = closed_set_system["recovery"]
    kf = KalmanFilter()

    # Generate appearance crops for 6 known people (1..6) and unknown G (7)
    crops = {i: make_distinct_crop(i) for i in range(1, 8)}

    # Step 1: Initial Enrollment of all 6 known people
    active_tracks = []
    for i in range(1, 7):
        det = create_detection(100 + i * 60, 200, frame_id=1)
        t = Track(det, 1, kf)
        t.state = TrackState.TRACKED
        t.last_crop = crops[i]
        active_tracks.append(t)
        im.process_tracks(None, frame_id=1, active_tracks=active_tracks, lost_tracks=[], newly_confirmed_tracks=[t], expired_tracks=[])
        # Add high quality frozen gallery sample
        emb = recovery.extractor.extract(crops[i])
        im._dormant_gallery.add_embedding(f"EMP00{i}", emb, quality_score=0.95, frame_id=1)

    assert len(im.employees) == 6
    assert [t.emp_id for t in active_tracks] == ["EMP001", "EMP002", "EMP003", "EMP004", "EMP005", "EMP006"]
    
    t_a = active_tracks[0]  # T-001 (EMP001)
    loc_a = t_a.bottom_center  # Location of A

    # Step 2: A leaves completely and expires
    active_tracks_no_a = active_tracks[1:]  # B, C, D, E, F
    im.process_tracks(None, frame_id=50, active_tracks=active_tracks_no_a, lost_tracks=[], newly_confirmed_tracks=[], expired_tracks=[t_a])

    assert not im.is_emp_active("EMP001")
    assert "EMP001" in im.dormant_emp_ids
    # B through F must strictly keep their EMP IDs
    for idx, emp_name in enumerate(["EMP002", "EMP003", "EMP004", "EMP005", "EMP006"]):
        assert active_tracks_no_a[idx].emp_id == emp_name
        assert im.is_emp_active(emp_name)

    # Step 3: G enters at A's exact old location
    t_g = Track(create_detection(loc_a[0], loc_a[1] - 50, frame_id=80), 80, kf)
    t_g.state = TrackState.TRACKED
    t_g.last_crop = crops[7]  # Appearance of G (differs from A)

    im.process_tracks(None, frame_id=80, active_tracks=active_tracks_no_a + [t_g], lost_tracks=[], newly_confirmed_tracks=[t_g], expired_tracks=[])

    # G MUST BE UNKNOWN and MUST NOT receive EMP001 (Location has ZERO authority)
    assert t_g.emp_id == "UNKNOWN"
    assert t_g.identity_status == "UNKNOWN"
    assert "EMP001" in im.dormant_emp_ids  # EMP001 must remain available
    assert not im.is_emp_active("EMP001")

    # Step 4: A returns as a new track (T-008)
    t_a_returned = Track(create_detection(100, 200, frame_id=120), 120, kf)
    t_a_returned.state = TrackState.TRACKED
    t_a_returned.last_crop = crops[1]  # Appearance of A

    current_active = active_tracks_no_a + [t_g, t_a_returned]
    im.process_tracks(None, frame_id=120, active_tracks=current_active, lost_tracks=[], newly_confirmed_tracks=[t_a_returned], expired_tracks=[])

    # A MUST recover EMP001
    assert t_a_returned.emp_id == "EMP001"
    assert t_a_returned.identity_status == "TRACKED"
    assert im.is_emp_active("EMP001")
    assert "EMP001" not in im.dormant_emp_ids

    # Final Acceptance Check:
    # A = EMP001, B = EMP002, C = EMP003, D = EMP004, E = EMP005, F = EMP006, G = UNKNOWN
    assert t_a_returned.emp_id == "EMP001"
    assert active_tracks_no_a[0].emp_id == "EMP002"
    assert active_tracks_no_a[1].emp_id == "EMP003"
    assert active_tracks_no_a[2].emp_id == "EMP004"
    assert active_tracks_no_a[3].emp_id == "EMP005"
    assert active_tracks_no_a[4].emp_id == "EMP006"
    assert t_g.emp_id == "UNKNOWN"


def test_second_test_single_departure_and_return(closed_set_system):
    """
    SECOND TEST:
    A leaves. A returns. Same EMP restored.
    T-ID changes (T001 -> T008), EMP001 preserved.
    """
    im: IdentityManager = closed_set_system["identity_manager"]
    recovery: EmergencyReIDRecovery = closed_set_system["recovery"]
    kf = KalmanFilter()

    crop_a = make_distinct_crop(1)

    # 1. Enroll A -> EMP001
    t1 = Track(create_detection(100, 100, frame_id=1), 1, kf)
    t1.state = TrackState.TRACKED
    t1.last_crop = crop_a
    im.process_tracks(None, frame_id=1, active_tracks=[t1], lost_tracks=[], newly_confirmed_tracks=[t1], expired_tracks=[])
    emb_a = recovery.extractor.extract(crop_a)
    im._dormant_gallery.add_embedding("EMP001", emb_a, quality_score=0.95, frame_id=1)
    assert t1.emp_id == "EMP001"

    # 2. A leaves and expires
    im.process_tracks(None, frame_id=40, active_tracks=[], lost_tracks=[], newly_confirmed_tracks=[], expired_tracks=[t1])
    assert "EMP001" in im.dormant_emp_ids

    # 3. A returns as T-008
    t8 = Track(create_detection(100, 100, frame_id=100), 100, kf)
    t8.state = TrackState.TRACKED
    t8.last_crop = crop_a

    im.process_tracks(None, frame_id=100, active_tracks=[t8], lost_tracks=[], newly_confirmed_tracks=[t8], expired_tracks=[])
    assert t8.emp_id == "EMP001"
    assert t8.identity_status == "TRACKED"
    assert im.is_emp_active("EMP001")


def test_third_test_dual_departure_and_return(closed_set_system):
    """
    THIRD TEST:
    A and B both leave. Both return.
    Both recover their original EMP IDs without duplicate or cross-over assignments.
    """
    im: IdentityManager = closed_set_system["identity_manager"]
    recovery: EmergencyReIDRecovery = closed_set_system["recovery"]
    kf = KalmanFilter()

    crop_a = make_distinct_crop(1)
    crop_b = make_distinct_crop(2)

    # 1. Enroll A -> EMP001 and B -> EMP002
    t_a1 = Track(create_detection(100, 100, frame_id=1), 1, kf)
    t_a1.state = TrackState.TRACKED
    t_a1.last_crop = crop_a

    t_b1 = Track(create_detection(200, 100, frame_id=1), 1, kf)
    t_b1.state = TrackState.TRACKED
    t_b1.last_crop = crop_b

    im.process_tracks(None, frame_id=1, active_tracks=[t_a1, t_b1], lost_tracks=[], newly_confirmed_tracks=[t_a1, t_b1], expired_tracks=[])
    im._dormant_gallery.add_embedding("EMP001", recovery.extractor.extract(crop_a), quality_score=0.95, frame_id=1)
    im._dormant_gallery.add_embedding("EMP002", recovery.extractor.extract(crop_b), quality_score=0.95, frame_id=1)

    assert t_a1.emp_id == "EMP001"
    assert t_b1.emp_id == "EMP002"

    # 2. Both A and B leave and expire
    im.process_tracks(None, frame_id=50, active_tracks=[], lost_tracks=[], newly_confirmed_tracks=[], expired_tracks=[t_a1, t_b1])
    assert "EMP001" in im.dormant_emp_ids
    assert "EMP002" in im.dormant_emp_ids

    # 3. Both A and B return simultaneously as new tracks T_A2 and T_B2
    t_a2 = Track(create_detection(120, 100, frame_id=100), 100, kf)
    t_a2.state = TrackState.TRACKED
    t_a2.last_crop = crop_a

    t_b2 = Track(create_detection(220, 100, frame_id=100), 100, kf)
    t_b2.state = TrackState.TRACKED
    t_b2.last_crop = crop_b

    im.process_tracks(None, frame_id=100, active_tracks=[t_a2, t_b2], lost_tracks=[], newly_confirmed_tracks=[t_a2, t_b2], expired_tracks=[])

    # Verified: A recovers EMP001, B recovers EMP002
    assert t_a2.emp_id == "EMP001"
    assert t_b2.emp_id == "EMP002"
    assert im.is_emp_active("EMP001")
    assert im.is_emp_active("EMP002")
    assert len(im.dormant_emp_ids) == 0
