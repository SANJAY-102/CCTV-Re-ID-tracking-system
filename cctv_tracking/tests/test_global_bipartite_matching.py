"""
Unit and Regression Tests for Global Bipartite Identity Recovery.

Validates:
1. Active EMP Lock: Active EMPs owned by healthy visible tracks are excluded from the dormant recovery pool.
2. Global Bipartite Matching: N candidate tracks matched against M dormant EMPs via Hungarian algorithm.
3. Strict 1-to-1 Mapping: No two tracks can be assigned the same EMP.
4. Ambiguity Guard: Candidates with margin < min_margin are rejected and marked UNKNOWN.
5. Strict RECOVERED Label: Only genuinely recovered dormant identities receive identity_status == 'RECOVERED'.
"""

import pytest
import numpy as np

from cctv_tracking.config import AppConfig
from cctv_tracking.identity.dormant_gallery import DormantIdentityGallery
from cctv_tracking.identity.identity_manager import IdentityManager
from cctv_tracking.identity.employee import EmployeeState
from cctv_tracking.tracker.track import Track, TrackState
from cctv_tracking.reid.model import ReIDExtractor
from cctv_tracking.reid.gallery import EmployeeGallery
from cctv_tracking.reid.recovery import EmergencyReIDRecovery


from cctv_tracking.tracker.kalman import KalmanFilter
from cctv_tracking.detector.yolo_detector import PersonDetection


class MockExtractor:
    """Mock extractor returning pre-configured embeddings for predictable testing."""
    def __init__(self, embedding_dim=512):
        self.embedding_dim = embedding_dim

    def extract(self, crop):
        return np.ones(self.embedding_dim, dtype=np.float32) / np.sqrt(self.embedding_dim)


def create_normalized_embedding(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def test_active_emp_lock():
    """Verify that an active EMP cannot be assigned to another candidate track."""
    config = AppConfig.default()
    config.identity.fixed_mode = True
    config.identity.max_known_employees = 6

    reid_extractor = ReIDExtractor(config.reid)
    gallery = EmployeeGallery(config.reid)
    reid_recovery = EmergencyReIDRecovery(reid_extractor, gallery, config.reid)
    im = IdentityManager(reid_recovery, config.identity)
    kf = KalmanFilter()

    # Enroll EMP001 with track 1
    d1 = PersonDetection(np.array([100, 100, 150, 200]), 0.95, (125, 150), (125, 200), 0)
    t1 = Track(d1, 1, kf)
    t1.state = TrackState.TRACKED
    im.register_new_employee(t1, frame_id=1)
    assert t1.emp_id == "EMP001"
    assert im.is_emp_active("EMP001")

    # t1 remains active. Now candidate t2 enters.
    d2 = PersonDetection(np.array([200, 100, 250, 200]), 0.95, (225, 150), (225, 200), 0)
    t2 = Track(d2, 2, kf)
    t2.state = TrackState.TRACKED

    # Process frame with t1 active, t2 newly confirmed
    im.process_tracks(None, frame_id=2, active_tracks=[t1], lost_tracks=[], newly_confirmed_tracks=[t2], expired_tracks=[])

    # t1 must retain EMP001, t2 CANNOT take EMP001
    assert t1.emp_id == "EMP001"
    assert t2.emp_id != "EMP001"
    # In fixed mode, t2 should be enrolled as EMP002 or UNKNOWN, but never EMP001
    assert t2.emp_id == "EMP002"
    assert t2.identity_status == "TRACKED"
    assert t2.identity_event == "NEW"


def test_hungarian_global_bipartite_assignment():
    """Verify that linear_sum_assignment assigns optimal pairs globally."""
    config = AppConfig.default().reid
    gallery = DormantIdentityGallery(config)

    # Create distinct orthogonal-ish embeddings for 3 employees
    e1 = create_normalized_embedding(10)
    e2 = create_normalized_embedding(20)
    e3 = create_normalized_embedding(30)

    gallery.add_embedding("EMP001", e1, quality_score=0.95, frame_id=1)
    gallery.add_embedding("EMP002", e2, quality_score=0.95, frame_id=1)
    gallery.add_embedding("EMP003", e3, quality_score=0.95, frame_id=1)

    # Query embeddings close to EMP001 and EMP002 respectively (~0.95 cosine similarity)
    noise1 = np.random.RandomState(1).randn(512).astype(np.float32)
    noise1 /= np.linalg.norm(noise1)
    q_t10 = 0.95 * e1 + 0.05 * noise1
    q_t10 /= np.linalg.norm(q_t10)

    noise2 = np.random.RandomState(2).randn(512).astype(np.float32)
    noise2 /= np.linalg.norm(noise2)
    q_t20 = 0.95 * e2 + 0.05 * noise2
    q_t20 /= np.linalg.norm(q_t20)

    query_embeddings = {10: q_t10, 20: q_t20}
    dormant_pool = {"EMP001", "EMP002", "EMP003"}

    results = gallery.match_batch_global(query_embeddings, dormant_pool, similarity_threshold=0.70, min_margin=0.05)

    assert results[10][0] == "EMP001"
    assert results[20][0] == "EMP002"
    assert results[10][1] > 0.90
    assert results[20][1] > 0.90


def test_no_duplicate_recovery():
    """Verify that two candidate tracks can NEVER both recover the same dormant EMP."""
    config = AppConfig.default().reid
    gallery = DormantIdentityGallery(config)

    e1 = create_normalized_embedding(42)
    gallery.add_embedding("EMP001", e1, quality_score=0.95, frame_id=1)

    # Both track 1 and track 2 are close to EMP001, but track 1 is slightly closer
    q1 = e1 + 0.01 * np.ones(512, dtype=np.float32)
    q1 /= np.linalg.norm(q1)

    q2 = e1 + 0.05 * np.ones(512, dtype=np.float32)
    q2 /= np.linalg.norm(q2)

    query_embeddings = {1: q1, 2: q2}
    dormant_pool = {"EMP001"}

    results = gallery.match_batch_global(query_embeddings, dormant_pool, similarity_threshold=0.70, min_margin=0.05)

    assigned_emps = [res[0] for res in results.values() if res[0] is not None]
    # Exactly one track gets EMP001, never both!
    assert assigned_emps == ["EMP001"]
    assert results[1][0] == "EMP001"
    assert results[2][0] is None


def test_ambiguity_margin_guard():
    """Verify that candidates with ambiguous scores (margin < min_margin) are rejected."""
    config = AppConfig.default().reid
    gallery = DormantIdentityGallery(config)

    # Create two very similar embeddings for EMP001 and EMP002
    base = create_normalized_embedding(100)
    e1 = base + 0.01 * np.ones(512, dtype=np.float32)
    e1 /= np.linalg.norm(e1)
    e2 = base + 0.015 * np.ones(512, dtype=np.float32)
    e2 /= np.linalg.norm(e2)

    gallery.add_embedding("EMP001", e1, quality_score=0.95, frame_id=1)
    gallery.add_embedding("EMP002", e2, quality_score=0.95, frame_id=1)

    # Query that matches both almost identically
    q = base.copy()
    query_embeddings = {5: q}
    dormant_pool = {"EMP001", "EMP002"}

    # Min margin is 0.05, but difference will be < 0.01
    results = gallery.match_batch_global(query_embeddings, dormant_pool, similarity_threshold=0.70, min_margin=0.05)

    # Must be rejected due to ambiguity
    assert results[5][0] is None
    assert results[5][3]["is_ambiguous"] is True
