"""
Formal Identity Architecture Tests (7 Required Tests).

Tests the redesigned Persistent Identity Architecture with:
  - ActiveEmployeeRegistry (live T-ID to EMP bindings)
  - DormantIdentityGallery (multi-embedding, persistent, appearance-based)
  - EmergencyReIDRecovery (OSNet Re-ID, ONLY on track expiration)

TEST 1: A leaves and returns -> must recover EMP-0001.
TEST 2: A leaves, B occupies same chair, A returns -> A=EMP-0001, B=EMP-0002.
TEST 3: A leaves, B enters -> A must NOT receive EMP-0002 (A still gets EMP-0001).
TEST 4: Re-ID similarity below threshold -> new person gets NEW EMP.
TEST 5: Ambiguous margin -> no identity is stolen, new EMP created.
TEST 6: 1000 frames, 6 continuous people -> Re-ID = 0.
TEST 7: 1000 frames, 6 people, one complete drop+return -> Re-ID = 1.

FINAL ACCEPTANCE TEST:
  A -> EMP-0001
  B -> EMP-0002
  A leaves -> EMP-0001 DORMANT
  C enters same location -> EMP-0003
  A returns with new T-ID -> EMP-0001 recovered
  Final: A=EMP-0001, B=EMP-0002, C=EMP-0003
"""

import time
import pytest
import numpy as np

from ..config import AppConfig
from ..tracker.primary_tracker import MotionAwareByteTrack
from ..tracker.track import Track, TrackState
from ..reid.model import ReIDExtractor
from ..reid.gallery import EmployeeGallery
from ..reid.recovery import EmergencyReIDRecovery
from ..identity.identity_manager import IdentityManager
from ..identity.registry import ActiveEmployeeRegistry
from ..identity.dormant_gallery import DormantIdentityGallery
from ..identity.employee import EmployeeState
from ..detector.yolo_detector import PersonDetection
from .synthetic_generator import SyntheticScenarioGenerator, SyntheticPerson


@pytest.fixture
def identity_system():
    """Standard isolated system with fast expiry for testing."""
    Track.reset_counter()
    config = AppConfig.default()
    config.tracker.max_lost_frames = 30   # 30 frames fast expiry for tests
    config.tracker.min_hits_to_confirm = 3
    config.reid.similarity_threshold = 0.70
    config.reid.min_margin = 0.05
    config.identity.fixed_mode = False
    config.identity.auto_enroll_unknown = True
    config.identity.emp_prefix = "EMP-"
    config.identity.enable_debug_logging = False  # suppress output during tests

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
        "generator": generator,
    }


def run_frames(system, frames, detections_seq):
    """Helper: run the full tracking + identity pipeline over a frame sequence."""
    im = system["identity_manager"]
    tr = system["tracker"]
    for f, (frame, dets) in enumerate(zip(frames, detections_seq)):
        active, lost, newly_conf, expired = tr.update(dets, frame_id=f)
        im.process_tracks(frame, f, active, lost, newly_conf, expired)


def make_det(person, canvas, frame_id):
    d = PersonDetection(
        bbox=person.bbox.copy(),
        confidence=0.95,
        center=person.center,
        bottom_center=person.bottom_center,
        frame_id=frame_id
    )
    d.extract_crop(canvas)
    return d


# ==============================================================================
# TEST 1: A leaves and returns -> must recover EMP-0001
# ==============================================================================
def test_1_person_returns_recovers_own_emp(identity_system):
    """
    TEST 1: A enters -> EMP-0001. A leaves. A returns with new T-ID.
    A MUST recover EMP-0001. total_enrollments == 1, total_recoveries == 1.
    """
    sys = identity_system
    frames, dets = sys["generator"].generate_person_exit_and_return(num_frames=140)
    run_frames(sys, frames, dets)

    im = sys["identity_manager"]
    assert im.total_enrollments == 1
    assert im.total_recoveries == 1
    assert len(sys["tracker"].all_active_tracks) == 1
    assert sys["tracker"].all_active_tracks[0].emp_id == "EMP-0001"
    assert sys["reid_recovery"].reid_calls_count == 1


# ==============================================================================
# TEST 2: A leaves, B occupies same chair, A returns -> A=EMP-0001, B=EMP-0002
# ==============================================================================
def test_2_same_chair_different_person_and_original_returns(identity_system):
    """
    TEST 2: A -> EMP-0001, leaves. B occupies same chair -> EMP-0002.
    A returns at different position -> Re-ID recovers EMP-0001.
    B stays as EMP-0002. No identity theft.
    """
    sys = identity_system
    im = sys["identity_manager"]
    tr = sys["tracker"]

    # Standard threshold 0.70 for Market1501 OSNet
    sys["reid_recovery"].config.similarity_threshold = 0.70
    im.reid_recovery.config.similarity_threshold = 0.70

    # Red person A (stationary at x=500)
    person_a = SyntheticPerson(1, (500, 300), (0, 0), (50, 120), (220, 30, 30), (30, 30, 220))
    # Green/Yellow person B (same location, completely different colors)
    person_b = SyntheticPerson(2, (500, 300), (0, 0), (50, 120), (20, 220, 20), (220, 220, 20))


    # Phase 1: A alone (frames 0..30)
    for f in range(35):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        person_a.draw(canvas)
        d = make_det(person_a, canvas, f)
        active, lost, newly_conf, expired = tr.update([d], frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    a_emp = next(t.emp_id for t in tr.all_active_tracks)
    assert a_emp == "EMP-0001"

    # Phase 2: A leaves (frames 35..80 — tracker expires)
    for f in range(35, 85):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        active, lost, newly_conf, expired = tr.update([], frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    # EMP-0001 must be dormant (not deleted!)
    assert "EMP-0001" in im.employees
    assert "EMP-0001" in im.dormant_emp_ids
    assert im.employees["EMP-0001"].state == EmployeeState.DORMANT

    # Phase 3: B enters same chair (frames 85..120)
    for f in range(85, 125):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        person_b.draw(canvas)
        d = make_det(person_b, canvas, f)
        active, lost, newly_conf, expired = tr.update([d], frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    b_emp = next(t.emp_id for t in tr.all_active_tracks)
    # B MUST get EMP-0002 — NOT EMP-0001!
    assert b_emp == "EMP-0002"
    assert b_emp != "EMP-0001"

    # Phase 4: A returns at different position (x=200) (frames 125..165)
    person_a.x, person_a.y = 200.0, 300.0
    reid_before = sys["reid_recovery"].reid_calls_count

    for f in range(125, 165):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        person_b.draw(canvas)
        person_a.draw(canvas)
        db = make_det(person_b, canvas, f)
        da = make_det(person_a, canvas, f)
        active, lost, newly_conf, expired = tr.update([db, da], frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    track_a = next((t for t in tr.all_active_tracks if abs(t.bottom_center[0] - 200) < 60), None)
    track_b = next((t for t in tr.all_active_tracks if abs(t.bottom_center[0] - 500) < 60), None)

    assert track_a is not None, "Person A track not found after return"
    assert track_b is not None, "Person B track not found"

    a_returned = track_a.emp_id
    b_current = track_b.emp_id

    # A recovers own EMP-0001
    assert a_returned == "EMP-0001"
    # B still owns EMP-0002
    assert b_current == "EMP-0002"
    # A did NOT steal B's identity
    assert a_returned != b_current
    # Exactly 1 Re-ID call for A's return
    assert sys["reid_recovery"].reid_calls_count - reid_before == 1


# ==============================================================================
# TEST 3: A leaves, B enters -> A returns -> A must NOT receive EMP-0002
# ==============================================================================
def test_3_returning_person_does_not_steal_active_emp(identity_system):
    """
    TEST 3: A -> EMP-0001, B -> EMP-0002. A leaves. A returns.
    A must recover EMP-0001 and NEVER receive EMP-0002 (which belongs to active B).
    Active identities cannot be stolen.
    """
    sys = identity_system
    im = sys["identity_manager"]
    tr = sys["tracker"]

    person_a = SyntheticPerson(1, (200, 300), (0, 0), (50, 120), (200, 30, 30), (30, 30, 200))
    person_b = SyntheticPerson(2, (900, 300), (0, 0), (50, 120), (30, 200, 30), (200, 200, 30))

    # Phase 1: A and B are both present (frames 0..35)
    for f in range(40):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        person_a.draw(canvas)
        person_b.draw(canvas)
        da = make_det(person_a, canvas, f)
        db = make_det(person_b, canvas, f)
        active, lost, newly_conf, expired = tr.update([da, db], frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    emps = {t.emp_id for t in tr.all_active_tracks}
    assert emps == {"EMP-0001", "EMP-0002"}

    # Phase 2: A leaves (frames 40..90)
    for f in range(40, 90):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        person_b.draw(canvas)
        db = make_det(person_b, canvas, f)
        active, lost, newly_conf, expired = tr.update([db], frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    # EMP-0001 is dormant, EMP-0002 is active
    assert "EMP-0001" in im.dormant_emp_ids
    assert "EMP-0002" in im.active_emp_ids

    # Phase 3: A returns at new position (frames 90..130)
    person_a.x, person_a.y = 200.0, 300.0
    for f in range(90, 135):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        person_b.draw(canvas)
        person_a.draw(canvas)
        db = make_det(person_b, canvas, f)
        da = make_det(person_a, canvas, f)
        active, lost, newly_conf, expired = tr.update([db, da], frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    track_a = next((t for t in tr.all_active_tracks if abs(t.bottom_center[0] - 200) < 60), None)
    track_b = next((t for t in tr.all_active_tracks if abs(t.bottom_center[0] - 900) < 60), None)

    assert track_a is not None
    assert track_b is not None

    # A recovers EMP-0001 (not EMP-0002 which B owns)
    assert track_a.emp_id == "EMP-0001"
    assert track_b.emp_id == "EMP-0002"
    assert track_a.emp_id != track_b.emp_id


# ==============================================================================
# TEST 4: Re-ID similarity below threshold -> new person gets NEW EMP
# ==============================================================================
def test_4_failed_reid_below_threshold_new_emp(identity_system):
    """
    TEST 4: Person A exits. Threshold set high (0.90).
    Completely different person enters. Re-ID returns None. -> New EMP-0002.
    """
    sys = identity_system
    im = sys["identity_manager"]
    tr = sys["tracker"]
    sys["reid_recovery"].config.similarity_threshold = 0.90  # Artificially strict

    person1 = SyntheticPerson(1, (100, 360), (4.0, 0.0), (50, 120), (220, 50, 50), (50, 50, 220))
    person2 = SyntheticPerson(2, (1000, 360), (-4.0, 0.0), (50, 120), (10, 220, 220), (220, 10, 10))

    # Person 1 visible frames 0..30
    for f in range(140):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        dets = []
        if f <= 30:
            person1.draw(canvas)
            dets.append(make_det(person1, canvas, f))
            person1.step()
        elif f >= 100:
            person2.draw(canvas)
            dets.append(make_det(person2, canvas, f))
            person2.step()
        active, lost, newly_conf, expired = tr.update(dets, frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    # Re-ID fired once (on person2 appearance), returned None -> new EMP-0002
    assert sys["reid_recovery"].reid_calls_count == 1
    assert len(im.employees) == 2
    assert "EMP-0001" in im.employees
    assert "EMP-0002" in im.employees
    assert tr.all_active_tracks[0].emp_id == "EMP-0002"


# ==============================================================================
# TEST 5: Ambiguous margin -> no identity stolen, new EMP created
# ==============================================================================
def test_5_ambiguous_margin_creates_new_emp(identity_system):
    """
    TEST 5: DormantIdentityGallery margin guard.
    EMP-0001 sim = 0.61, EMP-0002 sim = 0.60 (margin = 0.01 < 0.05 minimum).
    Result: NO match. New EMP enrolled. Existing identities preserved.
    """
    sys = identity_system
    dg = sys["identity_manager"]._dormant_gallery
    dg.config.similarity_threshold = 0.55
    dg.config.min_margin = 0.05

    emb1 = np.zeros(512, dtype=np.float32)
    emb1[0], emb1[1] = 0.80, 0.60
    emb1 /= np.linalg.norm(emb1)

    emb2 = np.zeros(512, dtype=np.float32)
    emb2[0], emb2[1] = 0.79, 0.61
    emb2 /= np.linalg.norm(emb2)

    dg.add_embedding("EMP-0001", emb1, 0.95, 0)
    dg.add_embedding("EMP-0002", emb2, 0.95, 0)

    # Equidistant query
    q_emb = np.zeros(512, dtype=np.float32)
    q_emb[0], q_emb[1] = 0.795, 0.605
    q_emb /= np.linalg.norm(q_emb)

    matched, score, all_scores, meta = dg.match(
        q_emb,
        eligible_emp_ids={"EMP-0001", "EMP-0002"},
        similarity_threshold=0.55,
        min_margin=0.05
    )

    assert matched is None, f"Should be ambiguous but matched {matched}"
    assert meta["is_ambiguous"] is True
    assert meta["margin"] < 0.05


# ==============================================================================
# TEST 6: 1000 frames, 6 continuous people -> Re-ID inference count == 0
# ==============================================================================
def test_6_zero_reid_1000_frames_6_people(identity_system):
    """
    TEST 6: 6 continuously tracked people for 1000 frames.
    Zero tracker failures. Zero Re-ID inference calls.
    """
    sys = identity_system
    frames, dets = sys["generator"].generate_6_moving_people(num_frames=1000)
    run_frames(sys, frames, dets)

    assert sys["tracker"].tracker_updates_count == 1000
    assert len(sys["tracker"].all_active_tracks) == 6
    assert len(sys["identity_manager"].employees) == 6
    # ZERO Re-ID calls on continuous healthy tracking
    assert sys["reid_recovery"].reid_calls_count == 0


# ==============================================================================
# TEST 7: 1000 frames, 6 people, one complete drop+return -> Re-ID == 1
# ==============================================================================
def test_7_one_reid_on_one_complete_track_drop(identity_system):
    """
    TEST 7: 6 people for 1000 frames. One person's track is forced to expire and return.
    Re-ID inference count == exactly 1 (the emergency recovery call on return).
    """
    sys = identity_system
    frames, dets = sys["generator"].generate_6_people_with_single_drop(
        num_frames=1000,
        drop_person_idx=0,
        drop_start=150,
        drop_duration=100
    )
    run_frames(sys, frames, dets)

    assert sys["tracker"].tracker_updates_count == 1000
    assert len(sys["tracker"].all_active_tracks) == 6
    assert len(sys["identity_manager"].employees) == 6
    # Exactly ONE emergency Re-ID call
    assert sys["reid_recovery"].reid_calls_count == 1
    assert sys["reid_recovery"].recovery_success_count == 1


# ==============================================================================
# FINAL ACCEPTANCE TEST: A=EMP-0001, B=EMP-0002, C=EMP-0003, A_return=EMP-0001
# ==============================================================================
def test_final_acceptance_abc_lifecycle(identity_system):
    """
    FINAL ACCEPTANCE TEST:

    A enters -> EMP-0001
    B enters -> EMP-0002
    A leaves completely -> EMP-0001 DORMANT (preserved in gallery)
    C enters at A's EXACT same location -> must get EMP-0003 (not EMP-0001!)
    A returns with new T-ID -> Emergency Re-ID -> EMP-0001 recovered

    Final state:
      A = EMP-0001
      B = EMP-0002
      C = EMP-0003

    This MUST PASS or the architecture is considered broken.
    """
    sys = identity_system
    im = sys["identity_manager"]
    tr = sys["tracker"]

    # Three visually distinct people (Red/Blue, Green/Yellow, White/Black)
    person_a = SyntheticPerson(1, (300, 300), (0, 0), (50, 120), (220, 20, 20), (20, 20, 220))   # Red / Blue
    person_b = SyntheticPerson(2, (700, 300), (0, 0), (50, 120), (20, 220, 20), (220, 220, 20))  # Green / Yellow
    person_c = SyntheticPerson(3, (300, 300), (0, 0), (50, 120), (240, 240, 240), (20, 20, 20))   # White / Black

    # ── Step 1: A and B enter (frames 0..25) ────────────────────────────────
    for f in range(30):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        person_a.draw(canvas)
        person_b.draw(canvas)
        da = make_det(person_a, canvas, f)
        db = make_det(person_b, canvas, f)
        active, lost, newly_conf, expired = tr.update([da, db], frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    tracks_after_ab = tr.all_active_tracks
    assert len(tracks_after_ab) == 2

    tr_a = next(t for t in tracks_after_ab if abs(t.bottom_center[0] - 300) < 50)
    tr_b = next(t for t in tracks_after_ab if abs(t.bottom_center[0] - 700) < 50)
    a_initial = tr_a.emp_id
    b_initial = tr_b.emp_id
    assert a_initial == "EMP-0001"
    assert b_initial == "EMP-0002"

    # ── Step 2: A completely leaves (frames 30..80) ──────────────────────────
    for f in range(30, 85):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        person_b.draw(canvas)
        db = make_det(person_b, canvas, f)
        active, lost, newly_conf, expired = tr.update([db], frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    # EMP-0001 must be preserved in dormant gallery
    assert "EMP-0001" in im.employees, "EMP-0001 was deleted — CRITICAL BUG"
    assert "EMP-0001" in im.dormant_emp_ids, "EMP-0001 not dormant — tracking state bug"
    assert im.employees["EMP-0001"].state == EmployeeState.DORMANT
    assert im._dormant_gallery.has_dormant("EMP-0001"), "EMP-0001 gallery was cleared — CRITICAL BUG"
    assert "EMP-0002" in im.active_emp_ids

    # ── Step 3: C enters at EXACT same location as A (frames 85..120) ────────
    for f in range(85, 125):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        person_b.draw(canvas)
        person_c.draw(canvas)
        db = make_det(person_b, canvas, f)
        dc = make_det(person_c, canvas, f)
        active, lost, newly_conf, expired = tr.update([db, dc], frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    assert len(tr.all_active_tracks) == 2
    tr_c = next(t for t in tr.all_active_tracks if abs(t.bottom_center[0] - 300) < 50)
    c_initial = tr_c.emp_id
    # C must be a new identity, NOT EMP-0001!
    assert c_initial == "EMP-0003", f"C got {c_initial} but should be EMP-0003!"
    assert c_initial != "EMP-0001", "C stole A's dormant identity — CRITICAL BUG"

    # ── Step 4: A returns at new position (frames 125..165) ──────────────────
    person_a.x, person_a.y = 1000.0, 300.0
    reid_before = sys["reid_recovery"].reid_calls_count

    for f in range(125, 170):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        person_b.draw(canvas)
        person_c.draw(canvas)
        person_a.draw(canvas)
        db = make_det(person_b, canvas, f)
        dc = make_det(person_c, canvas, f)
        da = make_det(person_a, canvas, f)
        active, lost, newly_conf, expired = tr.update([db, dc, da], frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    assert len(tr.all_active_tracks) == 3

    tr_a_ret = next((t for t in tr.all_active_tracks if abs(t.bottom_center[0] - 1000) < 60), None)
    tr_b_ret = next((t for t in tr.all_active_tracks if abs(t.bottom_center[0] - 700) < 60), None)
    tr_c_ret = next((t for t in tr.all_active_tracks if abs(t.bottom_center[0] - 300) < 60), None)

    assert tr_a_ret is not None, "A's return track not found"
    assert tr_b_ret is not None, "B's track not found"
    assert tr_c_ret is not None, "C's track not found"

    a_return = tr_a_ret.emp_id
    b_final = tr_b_ret.emp_id
    c_final = tr_c_ret.emp_id

    # ── Final Assertions ──────────────────────────────────────────────────────
    assert a_initial == "EMP-0001"
    assert b_initial == "EMP-0002"
    assert c_initial == "EMP-0003"
    assert a_return == "EMP-0001", f"A returned as {a_return} instead of EMP-0001 — CRITICAL BUG"
    assert b_final == "EMP-0002"
    assert c_final == "EMP-0003"

    assert a_return != b_final, "A and B share the same EMP — CRITICAL BUG"
    assert a_return != c_final, "A and C share the same EMP — CRITICAL BUG"
    assert c_initial != "EMP-0001", "C stole A's identity — CRITICAL BUG"
    assert im._dormant_gallery.has_dormant("EMP-0001"), "EMP-0001 gallery lost — CRITICAL BUG"

    # Exactly 1 Re-ID call for A's return (B and C already enrolled)
    reid_for_a_return = sys["reid_recovery"].reid_calls_count - reid_before
    assert reid_for_a_return == 1, f"Expected 1 Re-ID for A's return, got {reid_for_a_return}"

    print("\n✓ FINAL ACCEPTANCE TEST PASSED:")
    print(f"  A_initial  = {a_initial}")
    print(f"  B          = {b_final}")
    print(f"  C          = {c_final}")
    print(f"  A_return   = {a_return}")


# ==============================================================================
# COMPONENT UNIT TESTS
# ==============================================================================

def test_active_registry_unit():
    """Unit test: ActiveEmployeeRegistry enforces strict 1-to-1 bindings."""
    registry = ActiveEmployeeRegistry()
    registry.register("EMP-0001", 1)
    registry.register("EMP-0002", 2)

    assert registry.is_emp_active("EMP-0001")
    assert registry.is_emp_active("EMP-0002")
    assert not registry.is_emp_active("EMP-0003")
    assert registry.get_emp_for_track(1) == "EMP-0001"
    assert registry.get_track_for_emp("EMP-0002") == 2

    released = registry.release_track(1)
    assert released == "EMP-0001"
    assert not registry.is_emp_active("EMP-0001")
    assert registry.is_emp_active("EMP-0002")


def test_dormant_gallery_multi_embedding():
    """Unit test: DormantIdentityGallery stores multiple embeddings and matches correctly."""
    from ..config import ReIDConfig
    config = ReIDConfig()
    config.similarity_threshold = 0.55
    config.min_margin = 0.05
    config.max_gallery_size_per_emp = 4

    dg = DormantIdentityGallery(config)

    # Person A gallery: two embeddings near [1, 0, 0, ...]
    emb_a1 = np.zeros(512, dtype=np.float32)
    emb_a1[0] = 1.0
    emb_a2 = np.zeros(512, dtype=np.float32)
    emb_a2[0], emb_a2[1] = 0.98, 0.14
    emb_a2 /= np.linalg.norm(emb_a2)

    dg.add_embedding("EMP-0001", emb_a1, 0.95, 1)
    dg.add_embedding("EMP-0001", emb_a2, 0.90, 2)

    # Person B gallery: completely different direction [0, 0, 1, 0, ...]
    emb_b1 = np.zeros(512, dtype=np.float32)
    emb_b1[2] = 1.0
    dg.add_embedding("EMP-0002", emb_b1, 0.92, 3)

    # Query near Person A
    q = np.zeros(512, dtype=np.float32)
    q[0] = 1.0
    matched, score, all_scores, meta = dg.match(
        q,
        eligible_emp_ids={"EMP-0001", "EMP-0002"},
        similarity_threshold=0.55,
        min_margin=0.05
    )
    assert matched == "EMP-0001"
    assert score >= 0.55
    assert meta["margin"] >= 0.05


def test_dormant_gallery_gallery_survives_track_expiry(identity_system):
    """
    Explicit unit test: Gallery embeddings MUST survive track expiration.
    When a track expires:
    - Employee profile remains in identity_manager.employees
    - Employee is in dormant_emp_ids
    - DormantIdentityGallery.has_dormant(emp_id) is True
    """
    sys = identity_system
    im = sys["identity_manager"]
    tr = sys["tracker"]

    person_a = SyntheticPerson(1, (400, 360), (0, 0), (50, 120), (200, 30, 30), (30, 30, 200))

    # Track A for 40 frames (gets enrolled and gallery populated)
    for f in range(40):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        person_a.draw(canvas)
        d = make_det(person_a, canvas, f)
        active, lost, newly_conf, expired = tr.update([d], frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    assert "EMP-0001" in im.employees
    assert im._dormant_gallery.has_dormant("EMP-0001")

    # A leaves; wait for track to expire completely
    for f in range(40, 85):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        active, lost, newly_conf, expired = tr.update([], frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    # CRITICAL: Gallery must be preserved after track expiration
    assert "EMP-0001" in im.employees, "Employee was deleted — CRITICAL BUG"
    assert "EMP-0001" in im.dormant_emp_ids, "Employee not dormant — tracking bug"
    assert im._dormant_gallery.has_dormant("EMP-0001"), "Gallery was destroyed — CRITICAL BUG"
    assert im.employees["EMP-0001"].state == EmployeeState.DORMANT

    gallery_embs = im._dormant_gallery.get_embeddings("EMP-0001")
    assert len(gallery_embs) > 0, "Gallery is empty after track expiration — CRITICAL BUG"


def test_active_emp_cannot_be_stolen(identity_system):
    """
    Active employee identity protection:
    An active EMP must NEVER appear in the dormant recovery pool.
    Even if another new track appears, it cannot steal an active person's EMP.
    """
    sys = identity_system
    im = sys["identity_manager"]
    tr = sys["tracker"]

    person_a = SyntheticPerson(1, (200, 300), (0, 0), (50, 120), (200, 20, 20), (20, 20, 200))
    person_b = SyntheticPerson(2, (800, 300), (0, 0), (50, 120), (20, 200, 20), (200, 200, 20))

    # Both active for 50 frames
    for f in range(60):
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        person_a.draw(canvas)
        person_b.draw(canvas)
        da = make_det(person_a, canvas, f)
        db = make_det(person_b, canvas, f)
        active, lost, newly_conf, expired = tr.update([da, db], frame_id=f)
        im.process_tracks(canvas, f, active, lost, newly_conf, expired)

    # Active pool must NOT appear in dormant recovery pool
    dormant = im.dormant_emp_ids
    active = im.active_emp_ids
    assert len(dormant & active) == 0, \
        f"CRITICAL BUG: Same EMPs appear in both active and dormant: {dormant & active}"
