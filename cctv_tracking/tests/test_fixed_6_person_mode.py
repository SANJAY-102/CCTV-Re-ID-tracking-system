"""
Formal Test Suite for Hard-Coded 6-Person Identity Mode.

Validates the exact specifications:
- Exactly 6 persistent identities: EMP001, EMP002, EMP003, EMP004, EMP005, EMP006.
- EMP-ID is permanent; T-ID is ephemeral.
- Zero Re-ID during healthy tracking and crossing.
- Complete track expiration + return recovers the original EMP-ID (Re-ID = 1).
- A 7th person entering (even in an absent person's exact chair) is assigned UNKNOWN (NOT EMP003, NOT EMP007).
- When the original person returns, they recover their EMP-ID, and the 7th person remains UNKNOWN.
- Zero spatial / chair identity gravity.
"""

import pytest
import numpy as np

from ..config import AppConfig
from ..tracker.primary_tracker import MotionAwareByteTrack
from ..tracker.track import Track, TrackState
from ..reid.model import ReIDExtractor
from ..reid.gallery import EmployeeGallery
from ..reid.recovery import EmergencyReIDRecovery
from ..identity.identity_manager import IdentityManager
from ..identity.employee import EmployeeState
from ..detector.yolo_detector import PersonDetection
from .synthetic_generator import SyntheticScenarioGenerator, SyntheticPerson


@pytest.fixture
def fixed_6_system():
    """Isolated tracking + identity system configured for Hard-Coded 6-Person Mode."""
    Track.reset_counter()
    config = AppConfig.default()
    config.tracker.max_lost_frames = 25   # Fast 25-frame expiry for deterministic unit tests
    config.tracker.min_hits_to_confirm = 3
    config.reid.similarity_threshold = 0.70
    config.reid.min_margin = 0.05
    config.identity.fixed_mode = True
    config.identity.auto_enroll_unknown = False
    config.identity.max_known_employees = 6
    config.identity.fixed_emp_ids = ["EMP001", "EMP002", "EMP003", "EMP004", "EMP005", "EMP006"]
    config.identity.unmatched_label = "UNKNOWN"
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


def make_det(person: SyntheticPerson, canvas: np.ndarray, frame_id: int) -> PersonDetection:
    """Helper: ensure person is drawn on canvas, create detection, and extract crop."""
    person.draw(canvas)
    d = PersonDetection(
        bbox=person.bbox.copy(),
        confidence=0.95,
        center=person.center,
        bottom_center=person.bottom_center,
        frame_id=frame_id
    )
    d.extract_crop(canvas)
    return d


def create_6_distinct_people():
    """Create 6 synthetic people with distinct appearances and positions."""
    return [
        SyntheticPerson(1, (150, 300), (0, 0), (50, 120), (220, 150, 30), (30, 150, 220)),   # P1 (A)
        SyntheticPerson(2, (300, 300), (0, 0), (50, 120), (150, 220, 30), (220, 30, 150)),   # P2 (B)
        SyntheticPerson(3, (450, 300), (0, 0), (50, 120), (220, 30, 30), (30, 30, 220)),     # P3 (C) Red/Blue
        SyntheticPerson(4, (600, 300), (0, 0), (50, 120), (30, 220, 220), (220, 20, 20)),    # P4 (D)
        SyntheticPerson(5, (750, 300), (0, 0), (50, 120), (220, 20, 220), (20, 20, 120)),   # P5 (E)
        SyntheticPerson(6, (900, 300), (0, 0), (50, 120), (120, 120, 120), (120, 240, 20)), # P6 (F)
    ]


# ==============================================================================
# TEST CASE 1: All 6 people visible for 1000 frames -> Re-ID = 0
# ==============================================================================
def test_case_1_all_6_visible_1000_frames_zero_reid(fixed_6_system):
    """
    TEST CASE 1:
    All 6 people visible for 1000 frames.
    Expected:
      EMP001...EMP006 remain stable.
      Re-ID calls: 0.
    """
    sys = fixed_6_system
    im = sys["identity_manager"]
    tr = sys["tracker"]
    people = create_6_distinct_people()

    for f in range(1000):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        dets = []
        for p in people:
            p.draw(canvas)
            dets.append(make_det(p, canvas, f))

        active, lost, newly_conf, expired = tr.update(dets, frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    # Telemetry Verification
    assert len(im.employees) == 6
    expected_emp_ids = {"EMP001", "EMP002", "EMP003", "EMP004", "EMP005", "EMP006"}
    assert set(im.employees.keys()) == expected_emp_ids

    # Every active track has a valid fixed EMP ID
    active_emp_ids = {t.emp_id for t in tr.all_active_tracks}
    assert active_emp_ids == expected_emp_ids

    # Critical requirement: EXACTLY ZERO Re-ID recovery calls
    assert sys["reid_recovery"].reid_calls_count == 0


# ==============================================================================
# TEST CASE 2: One known person leaves, expires, returns -> same EMP-ID, Re-ID = 1
# ==============================================================================
def test_case_2_one_leaves_expires_returns_same_emp_one_reid(fixed_6_system):
    """
    TEST CASE 2:
    EMP003 leaves. Tracker expires. Returns as new T-ID.
    Expected:
      Returns as EMP003.
      Re-ID calls: 1.
    """
    sys = fixed_6_system
    im = sys["identity_manager"]
    tr = sys["tracker"]
    people = create_6_distinct_people()

    # Step 1: Initial enrollment (frames 0..15)
    for f in range(15):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        dets = [make_det(p, canvas, f) for p in people]
        for p in people:
            p.draw(canvas)
        active, lost, newly_conf, expired = tr.update(dets, frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    assert "EMP003" in im.employees
    assert im.employees["EMP003"].state == EmployeeState.ACTIVE

    # Step 2: Person 3 leaves the scene (frames 15..55)
    # Tracker lost buffer is 25 frames, so by frame 50 Person 3's track is expired.
    remaining_people = [p for p in people if p.person_id != 3]
    for f in range(15, 55):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        dets = [make_det(p, canvas, f) for p in remaining_people]
        for p in remaining_people:
            p.draw(canvas)
        active, lost, newly_conf, expired = tr.update(dets, frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    # EMP003 must now be DORMANT and gallery preserved
    assert im.employees["EMP003"].state == EmployeeState.DORMANT
    assert "EMP003" in im.dormant_emp_ids
    assert sys["reid_recovery"].reid_calls_count == 0  # Still 0 during departure

    # Step 3: Person 3 returns (frames 55..70)
    for f in range(55, 70):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        dets = [make_det(p, canvas, f) for p in people]
        for p in people:
            p.draw(canvas)
        active, lost, newly_conf, expired = tr.update(dets, frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    # Recovery Verification
    p3_track = next((t for t in tr.all_active_tracks if abs(t.bottom_center[0] - 450) < 30), None)
    assert p3_track is not None
    assert p3_track.emp_id == "EMP003"
    assert im.employees["EMP003"].state == EmployeeState.ACTIVE
    assert sys["reid_recovery"].reid_calls_count == 1
    assert im.total_recoveries == 1


# ==============================================================================
# TEST CASE 3: Person 3 leaves, 7th person enters same location -> UNKNOWN
# ==============================================================================
def test_case_3_person_3_leaves_seventh_person_enters_same_location_becomes_unknown(fixed_6_system):
    """
    TEST CASE 3:
    Person 3 leaves. A seventh person enters and occupies Person 3's EXACT physical location.
    Expected:
      Person 7 = UNKNOWN (NOT EMP003).
      Zero chair/location identity authority.
    """
    sys = fixed_6_system
    im = sys["identity_manager"]
    tr = sys["tracker"]
    people = create_6_distinct_people()

    # Step 1: Initial enrollment (frames 0..15)
    for f in range(15):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        dets = [make_det(p, canvas, f) for p in people]
        for p in people:
            p.draw(canvas)
        active, lost, newly_conf, expired = tr.update(dets, frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    # Step 2: Person 3 leaves (frames 15..55)
    remaining_people = [p for p in people if p.person_id != 3]
    for f in range(15, 55):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        dets = [make_det(p, canvas, f) for p in remaining_people]
        for p in remaining_people:
            p.draw(canvas)
        active, lost, newly_conf, expired = tr.update(dets, frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    assert "EMP003" in im.dormant_emp_ids

    # Step 3: Person 7 (G, Green/Yellow) enters at Person 3's EXACT position (450, 300)
    person_7 = SyntheticPerson(7, (450, 300), (0, 0), (50, 120), (20, 220, 20), (220, 220, 20))
    active_pool_step3 = remaining_people + [person_7]

    for f in range(55, 75):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        dets = [make_det(p, canvas, f) for p in active_pool_step3]
        for p in active_pool_step3:
            p.draw(canvas)
        active, lost, newly_conf, expired = tr.update(dets, frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    # Verification: Person 7 MUST be UNKNOWN
    p7_track = next((t for t in tr.all_active_tracks if abs(t.bottom_center[0] - 450) < 30), None)
    assert p7_track is not None
    assert p7_track.emp_id == "UNKNOWN"
    assert p7_track.emp_id != "EMP003"
    assert "EMP007" not in im.employees

    # EMP003 MUST remain dormant for when the real Person 3 returns
    assert "EMP003" in im.dormant_emp_ids


# ==============================================================================
# TEST CASE 4: Person 3 returns while Person 7 is still present -> No Swap
# ==============================================================================
def test_case_4_person_3_returns_while_person_7_present_no_swap(fixed_6_system):
    """
    TEST CASE 4:
    Person 3 returns while Person 7 is still present.
    Expected:
      Person 3 -> EMP003.
      Person 7 -> UNKNOWN.
      No identity swap.
    """
    sys = fixed_6_system
    im = sys["identity_manager"]
    tr = sys["tracker"]
    people = create_6_distinct_people()

    # Step 1: Initial enrollment (frames 0..15)
    for f in range(15):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        dets = [make_det(p, canvas, f) for p in people]
        for p in people:
            p.draw(canvas)
        active, lost, newly_conf, expired = tr.update(dets, frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    # Step 2: Person 3 leaves (frames 15..55)
    remaining_people = [p for p in people if p.person_id != 3]
    for f in range(15, 55):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        dets = [make_det(p, canvas, f) for p in remaining_people]
        for p in remaining_people:
            p.draw(canvas)
        active, lost, newly_conf, expired = tr.update(dets, frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    # Step 3: Person 7 enters at (450, 300) and becomes UNKNOWN (frames 55..75)
    person_7 = SyntheticPerson(7, (450, 300), (0, 0), (50, 120), (20, 220, 20), (220, 220, 20))
    for f in range(55, 75):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        dets = [make_det(p, canvas, f) for p in remaining_people + [person_7]]
        for p in remaining_people + [person_7]:
            p.draw(canvas)
        active, lost, newly_conf, expired = tr.update(dets, frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    # Step 4: Person 3 returns at a different location (1100, 300) while Person 7 is still there
    person_3_returning = SyntheticPerson(3, (1100, 300), (0, 0), (50, 120), (220, 30, 30), (30, 30, 220))
    all_seven = remaining_people + [person_7, person_3_returning]

    for f in range(75, 95):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        dets = [make_det(p, canvas, f) for p in all_seven]
        for p in all_seven:
            p.draw(canvas)
        active, lost, newly_conf, expired = tr.update(dets, frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    # Verification: Person 3 has EMP003, Person 7 remains UNKNOWN
    p7_track = next((t for t in tr.all_active_tracks if abs(t.bottom_center[0] - 450) < 30), None)
    p3_track = next((t for t in tr.all_active_tracks if abs(t.bottom_center[0] - 1100) < 30), None)

    assert p7_track is not None and p7_track.emp_id == "UNKNOWN"
    assert p3_track is not None and p3_track.emp_id == "EMP003"


# ==============================================================================
# TEST CASE 5: Two known people cross each other -> Re-ID = 0
# ==============================================================================
def test_case_5_two_known_people_cross_zero_reid(fixed_6_system):
    """
    TEST CASE 5:
    Two known people cross each other.
    Expected:
      EMP IDs remain stable.
      Re-ID calls: 0.
    """
    sys = fixed_6_system
    im = sys["identity_manager"]
    tr = sys["tracker"]

    # Person 1 moving right (x=200 -> 600)
    p1 = SyntheticPerson(1, (200, 300), (4.0, 0.0), (50, 120), (220, 150, 30), (30, 150, 220))
    # Person 2 moving left (x=600 -> 200)
    p2 = SyntheticPerson(2, (600, 300), (-4.0, 0.0), (50, 120), (150, 220, 30), (220, 30, 150))
    # Others stationary
    other_4 = [
        SyntheticPerson(3, (750, 300), (0, 0), (50, 120), (220, 30, 30), (30, 30, 220)),
        SyntheticPerson(4, (850, 300), (0, 0), (50, 120), (30, 220, 220), (220, 20, 20)),
        SyntheticPerson(5, (950, 300), (0, 0), (50, 120), (220, 20, 220), (20, 20, 120)),
        SyntheticPerson(6, (1050, 300), (0, 0), (50, 120), (120, 120, 120), (120, 240, 20)),
    ]
    all_6 = [p1, p2] + other_4

    for f in range(60):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        p1.step()
        p2.step()
        dets = [make_det(p, canvas, f) for p in all_6]
        for p in all_6:
            p.draw(canvas)
        active, lost, newly_conf, expired = tr.update(dets, frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    # Find tracks post-crossing (P1 is now near x=440, P2 is near x=360)
    track_p1 = next((t for t in tr.all_active_tracks if t.bottom_center[0] > 400 and t.bottom_center[0] < 550), None)
    track_p2 = next((t for t in tr.all_active_tracks if t.bottom_center[0] > 250 and t.bottom_center[0] < 400), None)

    assert track_p1 is not None and track_p1.emp_id == "EMP001"
    assert track_p2 is not None and track_p2.emp_id == "EMP002"
    assert sys["reid_recovery"].reid_calls_count == 0


# ==============================================================================
# TEST CASE 6: All six move quickly -> continuity maintained, Re-ID = 0
# ==============================================================================
def test_case_6_all_six_move_quickly_tracker_maintains_continuity_zero_reid(fixed_6_system):
    """
    TEST CASE 6:
    All six people move quickly.
    Expected:
      Tracker maintains T-ID continuity.
      Re-ID calls: 0.
    """
    sys = fixed_6_system
    im = sys["identity_manager"]
    tr = sys["tracker"]

    # All 6 people moving at moderate speed to maintain association
    people = [
        SyntheticPerson(1, (100, 200), (3.0, 0.0), (50, 120), (220, 150, 30), (30, 150, 220)),
        SyntheticPerson(2, (100, 300), (3.0, 0.0), (50, 120), (150, 220, 30), (220, 30, 150)),
        SyntheticPerson(3, (100, 400), (3.0, 0.0), (50, 120), (220, 30, 30), (30, 30, 220)),
        SyntheticPerson(4, (100, 500), (3.0, 0.0), (50, 120), (30, 220, 220), (220, 20, 20)),
        SyntheticPerson(5, (100, 600), (3.0, 0.0), (50, 120), (220, 20, 220), (20, 20, 120)),
        SyntheticPerson(6, (700, 350), (-3.0, 0.0), (50, 120), (120, 120, 120), (120, 240, 20)),
    ]

    for f in range(60):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        for p in people:
            p.step()
            p.draw(canvas)
        dets = [make_det(p, canvas, f) for p in people]
        active, lost, newly_conf, expired = tr.update(dets, frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    assert len(tr.all_active_tracks) == 6
    emp_ids = {t.emp_id for t in tr.all_active_tracks}
    assert emp_ids == {"EMP001", "EMP002", "EMP003", "EMP004", "EMP005", "EMP006"}
    assert sys["reid_recovery"].reid_calls_count == 0


# ==============================================================================
# FINAL ACCEPTANCE TEST: Exact A-F, C leaves, G enters -> UNKNOWN, C returns -> EMP003
# ==============================================================================
def test_final_acceptance_full_abc_def_g_lifecycle(fixed_6_system):
    """
    FINAL ACCEPTANCE TEST:
    Exact sequence:
      Six people:
        A -> EMP001
        B -> EMP002
        C -> EMP003
        D -> EMP004
        E -> EMP005
        F -> EMP006
      C leaves.
      C's T-ID expires.
      Seventh person G enters C's old location.
      G -> UNKNOWN
      C returns with a new T-ID.
      C -> EMP003

      Final:
        A = EMP001
        B = EMP002
        C = EMP003
        D = EMP004
        E = EMP005
        F = EMP006
        G = UNKNOWN
      There must be NO EMP-ID swap.
    """
    sys = fixed_6_system
    im = sys["identity_manager"]
    tr = sys["tracker"]

    # 1. Initialize Six People (A..F)
    person_a = SyntheticPerson(1, (150, 300), (0, 0), (50, 120), (220, 150, 30), (30, 150, 220))   # A
    person_b = SyntheticPerson(2, (300, 300), (0, 0), (50, 120), (150, 220, 30), (220, 30, 150))   # B
    person_c = SyntheticPerson(3, (450, 300), (0, 0), (50, 120), (220, 30, 30), (30, 30, 220))     # C (Red/Blue)
    person_d = SyntheticPerson(4, (600, 300), (0, 0), (50, 120), (30, 220, 220), (220, 20, 20))    # D
    person_e = SyntheticPerson(5, (750, 300), (0, 0), (50, 120), (220, 20, 220), (20, 20, 120))   # E
    person_f = SyntheticPerson(6, (900, 300), (0, 0), (50, 120), (120, 120, 120), (120, 240, 20)) # F
    initial_six = [person_a, person_b, person_c, person_d, person_e, person_f]

    for f in range(15):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        for p in initial_six:
            p.draw(canvas)
        dets = [make_det(p, canvas, f) for p in initial_six]
        active, lost, newly_conf, expired = tr.update(dets, frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    # Verify initial enrollment
    assert len(im.employees) == 6
    for idx, eid in enumerate(["EMP001", "EMP002", "EMP003", "EMP004", "EMP005", "EMP006"]):
        assert eid in im.employees
        assert im.employees[eid].state == EmployeeState.ACTIVE

    # 2. C leaves the scene (frames 15..55)
    remaining_five = [p for p in initial_six if p.person_id != 3]
    for f in range(15, 55):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        for p in remaining_five:
            p.draw(canvas)
        dets = [make_det(p, canvas, f) for p in remaining_five]
        active, lost, newly_conf, expired = tr.update(dets, frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    # C's T-ID expired, EMP003 is DORMANT
    assert im.employees["EMP003"].state == EmployeeState.DORMANT
    assert "EMP003" in im.dormant_emp_ids

    # 3. Seventh person G enters C's exact former location (450, 300)
    person_g = SyntheticPerson(7, (450, 300), (0, 0), (50, 120), (20, 220, 20), (220, 220, 20)) # Green/Yellow (distinct from C's Red/Blue)
    pool_step3 = remaining_five + [person_g]

    for f in range(55, 75):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        for p in pool_step3:
            p.draw(canvas)
        dets = [make_det(p, canvas, f) for p in pool_step3]
        active, lost, newly_conf, expired = tr.update(dets, frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    # G must receive UNKNOWN (NOT EMP003, NOT EMP007)
    track_g = next((t for t in tr.all_active_tracks if abs(t.bottom_center[0] - 450) < 30), None)
    assert track_g is not None
    assert track_g.emp_id == "UNKNOWN"
    assert "EMP003" in im.dormant_emp_ids
    assert "EMP007" not in im.employees

    # 4. C returns with a new T-ID at (1100, 300)
    person_c_return = SyntheticPerson(3, (1100, 300), (0, 0), (50, 120), (220, 30, 30), (30, 30, 220))
    pool_step4 = remaining_five + [person_g, person_c_return]

    for f in range(75, 95):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        for p in pool_step4:
            p.draw(canvas)
        dets = [make_det(p, canvas, f) for p in pool_step4]
        active, lost, newly_conf, expired = tr.update(dets, frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    # Final Verification:
    # A = EMP001, B = EMP002, C = EMP003, D = EMP004, E = EMP005, F = EMP006, G = UNKNOWN
    track_a = next(t for t in tr.all_active_tracks if abs(t.bottom_center[0] - 150) < 30)
    track_b = next(t for t in tr.all_active_tracks if abs(t.bottom_center[0] - 300) < 30)
    track_g_final = next(t for t in tr.all_active_tracks if abs(t.bottom_center[0] - 450) < 30)
    track_d = next(t for t in tr.all_active_tracks if abs(t.bottom_center[0] - 600) < 30)
    track_e = next(t for t in tr.all_active_tracks if abs(t.bottom_center[0] - 750) < 30)
    track_f = next(t for t in tr.all_active_tracks if abs(t.bottom_center[0] - 900) < 30)
    track_c_final = next(t for t in tr.all_active_tracks if abs(t.bottom_center[0] - 1100) < 30)

    assert track_a.emp_id == "EMP001"
    assert track_b.emp_id == "EMP002"
    assert track_c_final.emp_id == "EMP003"
    assert track_d.emp_id == "EMP004"
    assert track_e.emp_id == "EMP005"
    assert track_f.emp_id == "EMP006"
    assert track_g_final.emp_id == "UNKNOWN"

    assert len(tr.all_active_tracks) == 7
    assert "EMP007" not in im.employees
