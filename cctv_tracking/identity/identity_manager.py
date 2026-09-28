"""
Identity Manager — Thin Orchestrator.

Wires together three independent components:
  1. ActiveEmployeeRegistry  — authoritative live track-to-EMP bindings
  2. DormantIdentityGallery  — persistent appearance gallery for departed persons
  3. EmergencyOSNetRecovery  — deep Re-ID inference (activated ONLY on track expiration)

Decision Pipeline per frame
───────────────────────────

ACTIVE TRACKS:
  → Keep existing EMP binding. Zero Re-ID.

LOST TRACKS (ByteTrack lost buffer — NOT expired):
  → Keep EMP locked. No other track may claim it. Zero Re-ID.

EXPIRED TRACKS (ByteTrack deleted track):
  → Release from ActiveEmployeeRegistry.
  → Register (or update) employee in DormantIdentityGallery.
  → Employee object preserved PERMANENTLY.

NEWLY CONFIRMED TRACKS (ByteTrack confirmed tentative track):
  → Query: are there DORMANT employees in gallery?
    NO  → Auto-enroll as NEW EMP. Re-ID = 0.
    YES → This track becomes a RECOVERY CANDIDATE.
           Extract OSNet embedding.
           Query DormantIdentityGallery.match() with strict margin guard.
           MATCH   → Recover dormant EMP. Move DORMANT → ACTIVE.
           NO MATCH→ Auto-enroll as NEW EMP.
           Either way: Re-ID calls this frame = 1 per new track.

COLLISION SAFE-FAIL:
  → If two tracks somehow claim the same EMP-ID, the weaker track loses.

Structured Logging (per user specification):
  NEW TRACK: T-003
  RECOVERY CANDIDATES: [EMP-0001, EMP-0003]
  ReID: T-003 -> 1 emergency inference
  SIMILARITIES: EMP-0001 = 0.86, EMP-0003 = 0.24
  DECISION: RECOVER EMP-0001
  (or)
  DECISION: NEW EMP-0004
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import time
import numpy as np

from .employee import Employee, EmployeeState
from .registry import ActiveEmployeeRegistry
from .dormant_gallery import DormantIdentityGallery, DormantEmbeddingRecord
from ..tracker.track import Track, TrackState
from ..reid.recovery import EmergencyReIDRecovery
from ..config import IdentityConfig


class IdentityManager:
    """
    Orchestrates persistent physical person identity across track lifetimes.
    Delegates state to ActiveEmployeeRegistry (live) and DormantIdentityGallery (departed).
    """

    def __init__(
        self,
        reid_recovery: EmergencyReIDRecovery,
        config: Optional[IdentityConfig] = None
    ):
        self.reid_recovery = reid_recovery
        self.config = config or IdentityConfig()

        self._emp_counter: int = 0

        # COMPONENT 1: Master persistent employee profiles (NEVER deleted)
        self.employees: Dict[str, Employee] = {}

        # COMPONENT 2: Live bidirectional track <-> EMP bindings
        self._active_registry = ActiveEmployeeRegistry()

        # COMPONENT 3: Dormant appearance gallery (persists after track death)
        self._dormant_gallery = DormantIdentityGallery(self.reid_recovery.config)

        # Telemetry
        self.total_enrollments: int = 0

    @property
    def reid_inference_calls(self) -> int:
        return self.reid_recovery.reid_inference_calls

    @property
    def recovery_attempts(self) -> int:
        return self.reid_recovery.recovery_attempts

    @property
    def successful_identity_recoveries(self) -> int:
        return self.reid_recovery.successful_identity_recoveries

    @property
    def total_recoveries(self) -> int:
        return self.reid_recovery.successful_identity_recoveries

    @total_recoveries.setter
    def total_recoveries(self, val: int) -> None:
        self.reid_recovery.successful_identity_recoveries = val

    def reset(self) -> None:
        """Reset all state for test isolation."""
        self._emp_counter = 0
        self.employees.clear()
        self._active_registry.clear()
        self._dormant_gallery._records.clear()
        self.total_enrollments = 0
        self.reid_recovery.reid_inference_calls = 0
        self.reid_recovery.recovery_attempts = 0
        self.reid_recovery.successful_identity_recoveries = 0
        self.reid_recovery.recovery_failed_count = 0

    # ------------------------------------------------------------------
    # Public Properties
    # ------------------------------------------------------------------

    @property
    def active_emp_ids(self) -> Set[str]:
        return self._active_registry.active_emp_ids

    @property
    def dormant_emp_ids(self) -> Set[str]:
        """
        Returns EMP-IDs eligible for emergency Re-ID recovery.
        An EMP is dormant when it exists in the employee registry but has no active track binding.
        INVARIANT: Active EMPs are NEVER in this set.
        """
        all_emp_ids = set(self.employees.keys())
        return all_emp_ids - self._active_registry.active_emp_ids

    @property
    def available_for_recovery_emp_ids(self) -> Set[str]:
        """Alias for dormant_emp_ids (backward compatibility)."""
        return self.dormant_emp_ids

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _generate_next_emp_id(self) -> str:
        if getattr(self.config, "fixed_mode", False):
            fixed_ids = getattr(self.config, "fixed_emp_ids", ["EMP001", "EMP002", "EMP003", "EMP004", "EMP005", "EMP006"])
            if self._emp_counter < len(fixed_ids):
                emp_id = fixed_ids[self._emp_counter]
                self._emp_counter += 1
                return emp_id
            return getattr(self.config, "unmatched_label", "UNKNOWN")
        self._emp_counter += 1
        return f"{self.config.emp_prefix}{self._emp_counter:04d}"

    def _extract_crop(self, track: Track, frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Slice person crop from frame using track bounding box."""
        if frame is None or frame.size == 0:
            return None
        x1, y1, x2, y2 = [int(round(v)) for v in track.to_tlbr()]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            return frame[y1:y2, x1:x2].copy()
        return None

    def _log(self, msg: str) -> None:
        if self.config.enable_debug_logging:
            print(f"[IdentityManager] {msg}")

    # ------------------------------------------------------------------
    # Employee Registration
    # ------------------------------------------------------------------

    def register_new_employee(
        self,
        track: Track,
        frame_id: int,
        frame: Optional[np.ndarray] = None,
        initial_embedding: Optional[np.ndarray] = None,
        forced_emp_id: Optional[str] = None
    ) -> str:
        """
        Enroll a brand-new physical employee.
        Populates both the employee profile and the dormant gallery baseline.
        """
        if forced_emp_id is not None:
            emp_id = forced_emp_id
        else:
            emp_id = self._generate_next_emp_id()

        # If unmatched/limit reached in fixed mode, assign label without creating persistent employee
        if emp_id == getattr(self.config, "unmatched_label", "UNKNOWN"):
            track.emp_id = emp_id
            self._log(f"TRACK UNMATCHED: {track.tid_str} assigned {emp_id}")
            return emp_id

        # Ensure we have a valid crop
        if track.last_crop is None or track.last_crop.size == 0:
            crop = self._extract_crop(track, frame)
            if crop is not None:
                track.last_crop = crop

        # Extract baseline embedding for gallery if not already provided
        if initial_embedding is None and track.last_crop is not None and track.last_crop.size > 0:
            initial_embedding = self.reid_recovery.extractor.extract(track.last_crop)

        emp = Employee(
            emp_id=emp_id,
            name=f"Person {self._emp_counter}",
            state=EmployeeState.ACTIVE,
            current_track_id=track.track_id,
            current_tid_str=track.tid_str,
            associated_tids=[track.tid_str],
            created_at=time.time(),
            last_seen_timestamp=time.time(),
            first_seen_frame=frame_id,
            last_seen_frame=frame_id,
            total_frames_tracked=1,
            last_known_bbox=track.to_tlbr(),
            last_known_bottom_center=track.bottom_center,
            last_known_velocity=(track.vx, track.vy),
            last_known_direction=track.direction.value,
            last_known_embedding=initial_embedding
        )
        self.employees[emp_id] = emp

        # Bind in active registry
        self._active_registry.register(emp_id, track.track_id)
        track.emp_id = emp_id
        self.total_enrollments += 1

        # Add baseline embedding to dormant gallery (pre-population for future departure)
        if initial_embedding is not None:
            self._dormant_gallery.add_embedding(
                emp_id=emp_id,
                embedding=initial_embedding,
                quality_score=track.confidence,
                frame_id=frame_id
            )

        self._log(f"NEW ENROLLMENT: {emp_id} ← {track.tid_str} (frame {frame_id})")
        return emp_id

    # ------------------------------------------------------------------
    # Core Per-Frame Processing
    # ------------------------------------------------------------------

    def process_tracks(
        self,
        frame: Optional[np.ndarray],
        frame_id: int,
        active_tracks: List[Track],
        lost_tracks: List[Track],
        newly_confirmed_tracks: List[Track],
        expired_tracks: List[Track]
    ) -> None:
        """
        Main per-frame identity coordination entry point.
        Called once per frame with the four track lists from ByteTrack.
        """
        # =============================================================
        # PHASE 1: Process Expired Tracks (Track Death → DORMANT EMP)
        # =============================================================
        for exp_track in expired_tracks:
            if exp_track.emp_id and exp_track.emp_id in self.employees:
                emp = self.employees[exp_track.emp_id]

                # Transition employee state: ACTIVE → DORMANT
                emp.state = EmployeeState.DORMANT
                emp.current_track_id = None
                emp.current_tid_str = None
                emp.last_seen_frame = frame_id
                emp.last_seen_timestamp = time.time()
                emp.last_known_bbox = exp_track.to_tlbr()
                emp.last_known_bottom_center = exp_track.bottom_center

                # Release from active registry (EMP is now dormant — no active T-ID)
                self._active_registry.release_track(exp_track.track_id)

                # Register into DormantIdentityGallery for future long-gap recovery
                # (embeddings are already populated during tracking; this updates metadata)
                dormant_embeddings = [
                    DormantEmbeddingRecord(
                        embedding=emb,
                        quality_score=0.85,
                        frame_id=frame_id,
                        captured_at=time.time()
                    )
                    for emb in self._dormant_gallery.get_embeddings(emp.emp_id)
                ]
                if not dormant_embeddings and emp.last_known_embedding is not None:
                    dormant_embeddings = [DormantEmbeddingRecord(
                        embedding=emp.last_known_embedding,
                        quality_score=0.75,
                        frame_id=frame_id,
                        captured_at=time.time()
                    )]

                self._dormant_gallery.register(
                    emp_id=emp.emp_id,
                    name=emp.name,
                    embeddings=dormant_embeddings,
                    last_seen_frame=frame_id,
                    created_at=emp.created_at
                )

                self._log(
                    f"TRACK EXPIRED: {exp_track.tid_str} → {emp.emp_id} now DORMANT "
                    f"(gallery size: {len(self._dormant_gallery.get_embeddings(emp.emp_id))})"
                )

        # Clean up any stale active bindings
        live_and_lost = {t.track_id for t in active_tracks} | {t.track_id for t in lost_tracks}
        stale = [tid for tid in self._active_registry.active_track_ids if tid not in live_and_lost]
        for tid in stale:
            emp_id = self._active_registry.release_track(tid)
            if emp_id and emp_id in self.employees:
                self.employees[emp_id].state = EmployeeState.DORMANT
                self.employees[emp_id].current_track_id = None
                self.employees[emp_id].current_tid_str = None

        # =============================================================
        # PHASE 2: Maintain Healthy Active Tracks (Zero Re-ID)
        # =============================================================
        for track in active_tracks:
            if track.emp_id is not None and track.emp_id in self.employees:
                # Ensure binding is correct
                self._active_registry.register(track.emp_id, track.track_id)

                emp = self.employees[track.emp_id]
                emp.state = EmployeeState.ACTIVE
                emp.current_track_id = track.track_id
                emp.current_tid_str = track.tid_str
                emp.last_seen_frame = frame_id
                emp.last_seen_timestamp = time.time()
                emp.total_frames_tracked += 1
                emp.last_known_bbox = track.to_tlbr()
                emp.last_known_bottom_center = track.bottom_center
                emp.last_known_velocity = (track.vx, track.vy)
                emp.last_known_direction = track.direction.value

                # Update identity_status: normal tracked tracks are TRACKED, UNKNOWN is UNKNOWN
                if track.emp_id == getattr(self.config, "unmatched_label", "UNKNOWN"):
                    track.identity_status = "UNKNOWN"
                else:
                    track.identity_status = "TRACKED"

                # Refresh crop (no neural inference — just pixel slice)
                if frame is not None and frame.size > 0 and (frame_id % 30 == 0 or track.last_crop is None):
                    crop = self._extract_crop(track, frame)
                    if crop is not None:
                        track.last_crop = crop

                # Multi-Sample Gallery Collection:
                # If disable_dynamic_gallery is True, gallery is 100% frozen after initial enrollment.
                # CRITICAL INVARIANTS:
                # 1. NEVER learn from UNKNOWN tracks.
                # 2. NEVER learn from recovered tracks (post-recovery).
                # 3. NEVER update galleries when disable_dynamic_gallery is True.
                if not getattr(self.config, "disable_dynamic_gallery", True):
                    if (frame is not None and frame.size > 0 and track.confidence >= 0.70
                            and track.emp_id and track.emp_id in getattr(self.config, "fixed_emp_ids", [f"EMP{i:03d}" for i in range(1, 7)])
                            and not getattr(track, "was_recovered", False)
                            and getattr(track, "identity_event", "") != "RECOVERED"):
                        current_samples = len(self._dormant_gallery.get_embeddings(track.emp_id))
                        max_samples = getattr(self.config, "min_gallery_samples", 5)
                        if current_samples < max_samples:
                            crop = self._extract_crop(track, frame)
                            if crop is not None and crop.size > 0:
                                emb = self.reid_recovery.extractor.extract(crop)
                                self._dormant_gallery.add_embedding(
                                    emp_id=track.emp_id,
                                    embedding=emb,
                                    quality_score=track.confidence,
                                    frame_id=frame_id
                                )

        # =============================================================
        # PHASE 3: Maintain Temporarily Lost Tracks (Zero Re-ID, Lock Identity)
        # =============================================================
        for lost_trk in lost_tracks:
            if lost_trk.emp_id and lost_trk.emp_id in self.employees:
                emp = self.employees[lost_trk.emp_id]
                # Keep ACTIVE state — EMP is temporarily occluded, not departed
                emp.state = EmployeeState.ACTIVE

        # =============================================================
        # PHASE 4: Global Bipartite Recovery & Enrollment
        # =============================================================
        # First, preserve already-identified tracks (carried over from tentative phase)
        for track in newly_confirmed_tracks:
            if track.emp_id is not None and track.emp_id in self.employees:
                self._active_registry.register(track.emp_id, track.track_id)
                track.identity_status = "TRACKED"

        # STEP 1: Freeze active EMP ownership (ACTIVE_EMP_IDS)
        # Any EMP currently owned by a live/healthy track is strictly LOCKED.
        active_emp_ids = {
            t.emp_id for t in active_tracks
            if t.emp_id is not None and t.emp_id in self.employees
        } | {
            t.emp_id for t in newly_confirmed_tracks
            if t.emp_id is not None and t.emp_id in self.employees
        }

        # STEP 2: Collect all candidate tracks needing identity assignment
        # Strict eligibility (Section 6 & 7):
        # 1. Track does not already have an enrolled EMP-ID
        # 2. Track has NOT already attempted recovery (single one-shot session per track)
        candidate_tracks = [
            t for t in newly_confirmed_tracks
            if (t.emp_id is None or t.emp_id not in self.employees)
            and not getattr(t, "recovery_attempted", False)
        ]

        # Deterministic ordering for initial enrollment in fixed mode
        if getattr(self.config, "fixed_mode", False) and len(self.employees) < getattr(self.config, "max_known_employees", 6):
            candidate_tracks = sorted(candidate_tracks, key=lambda t: t.bottom_center[0])

        # STEP 3: Lock Active EMPs — Dormant pool excludes ALL active EMPs
        dormant_pool = (self.dormant_emp_ids & self._dormant_gallery.get_dormant_emp_ids()) - active_emp_ids

        # STEP 4: Process candidates globally
        if candidate_tracks:
            # CASE A: If no dormant candidates exist
            if len(dormant_pool) == 0:
                for track in candidate_tracks:
                    track.recovery_attempted = True
                    self._log(f"NEW TRACK: {track.tid_str}")
                    if getattr(self.config, "fixed_mode", False):
                        if len(self.employees) < getattr(self.config, "max_known_employees", 6):
                            new_emp_id = self.register_new_employee(track, frame_id, frame=frame)
                            track.identity_status = "TRACKED"
                            track.identity_event = "NEW"
                            track.identity_event_frame = frame_id
                            track.identity_confidence = 1.0
                            track.was_recovered = False
                            self._log(f"DECISION: FIXED ENROLLMENT {new_emp_id} ← {track.tid_str}")
                        else:
                            # 7th person enters -> UNKNOWN!
                            track.emp_id = getattr(self.config, "unmatched_label", "UNKNOWN")
                            track.identity_status = "UNKNOWN"
                            track.identity_event = "UNKNOWN"
                            track.identity_event_frame = frame_id
                            track.identity_confidence = 0.0
                            track.was_recovered = False
                            track.recovered_locked = True
                            self._log(f"DECISION: {track.tid_str} → UNKNOWN (no dormant candidates; all {self.config.max_known_employees} known EMPs occupied)")
                    else:
                        if self.config.auto_enroll_unknown:
                            new_emp_id = self.register_new_employee(track, frame_id, frame=frame)
                            track.identity_status = "TRACKED"
                            track.identity_event = "NEW"
                            track.identity_event_frame = frame_id
                            track.identity_confidence = 1.0
                            track.was_recovered = False
                            self._log(f"DECISION: NEW {new_emp_id} (no dormant candidates)")
                        else:
                            track.emp_id = getattr(self.config, "unmatched_label", "UNKNOWN")
                            track.identity_status = "UNKNOWN"
                            track.identity_event = "UNKNOWN"
                            track.identity_event_frame = frame_id
                            track.identity_confidence = 0.0
                            track.was_recovered = False
                            track.recovered_locked = True
            else:
                # CASE B: Dormant candidates exist -> Run Global Emergency Re-ID
                # Extract embeddings for ALL candidate tracks simultaneously
                # Mark recovery_attempted = True IMMEDIATELY (one-shot session, Section 7 & 8)
                query_embeddings: Dict[int, np.ndarray] = {}
                for track in candidate_tracks:
                    track.recovery_attempted = True
                    self.reid_recovery.recovery_attempts += 1
                    crop = self._extract_crop(track, frame)
                    if crop is None:
                        crop = track.last_crop if track.last_crop is not None else np.zeros((64, 32, 3), dtype=np.uint8)
                    track.last_crop = crop
                    emb = self.reid_recovery.extractor.extract(crop)
                    self.reid_recovery.reid_inference_calls += 1
                    query_embeddings[track.track_id] = emb

                self._log(f"GLOBAL RECOVERY CANDIDATES: {sorted(dormant_pool)}")

                # Solve global Hungarian bipartite assignment
                batch_results = self._dormant_gallery.match_batch_global(
                    query_embeddings=query_embeddings,
                    eligible_emp_ids=dormant_pool,
                    similarity_threshold=self.reid_recovery.config.similarity_threshold,
                    min_margin=self.reid_recovery.config.min_margin,
                )

                # Apply ALL assignments atomically
                for track in candidate_tracks:
                    recovered_emp_id, score, all_scores, meta = batch_results.get(
                        track.track_id, (None, 0.0, {}, {})
                    )
                    best_emp = meta.get("best_emp", "None")
                    margin_val = meta.get("margin", float("inf"))
                    margin_str = f"{margin_val:.4f}" if margin_val != float("inf") else "inf"
                    
                    # Format candidate score list (all 6 known EMPs)
                    score_lines = []
                    for eid in getattr(self.config, "fixed_emp_ids", [f"EMP{i:03d}" for i in range(1, 7)]):
                        if eid in all_scores:
                            score_lines.append(f"  Candidate {eid} = {all_scores[eid]:.4f}")
                        elif eid in active_emp_ids:
                            score_lines.append(f"  Candidate {eid} = LOCKED (Active)")
                        else:
                            score_lines.append(f"  Candidate {eid} = N/A (Not in gallery)")
                    cand_block = "\n".join(score_lines)

                    if recovered_emp_id is not None and recovered_emp_id in dormant_pool:
                        decision_str = f"RECOVERED ({recovered_emp_id})"
                    elif meta.get("is_ambiguous"):
                        decision_str = "UNKNOWN (ambiguous margin)"
                    elif meta.get("best_score", 0.0) < self.reid_recovery.config.similarity_threshold:
                        decision_str = "UNKNOWN (below similarity threshold)"
                    else:
                        decision_str = "UNKNOWN (Hungarian conflict or rejected)"

                    self._log(
                        f"\n==================================================\n"
                        f"FRAME {frame_id}\n"
                        f"{track.tid_str}\n"
                        f"{cand_block}\n"
                        f"BEST = {best_emp} (score: {meta.get('best_score', 0):.4f}, margin: {margin_str})\n"
                        f"DECISION = {decision_str}\n"
                        f"=================================================="
                    )

                    if recovered_emp_id is not None and recovered_emp_id in dormant_pool:
                        # RECOVERY SUCCESS
                        emp = self.employees[recovered_emp_id]
                        emp.state = EmployeeState.ACTIVE
                        emp.current_track_id = track.track_id
                        emp.current_tid_str = track.tid_str
                        if track.tid_str not in emp.associated_tids:
                            emp.associated_tids.append(track.tid_str)
                        emp.last_seen_frame = frame_id
                        emp.last_seen_timestamp = time.time()
                        
                        # Lock active registry immediately
                        self._active_registry.register(recovered_emp_id, track.track_id)
                        track.emp_id = recovered_emp_id
                        track.identity_status = "TRACKED"
                        track.identity_event = "RECOVERED"
                        track.identity_event_frame = frame_id
                        track.recovered_locked = True
                        track.recovered_from_emp = recovered_emp_id
                        track.identity_confidence = score
                        track.was_recovered = True
                        self.reid_recovery.successful_identity_recoveries += 1
                    else:
                        # RECOVERY FAILED / AMBIGUOUS / UNKNOWN
                        self.reid_recovery.recovery_failed_count += 1
                        if getattr(self.config, "fixed_mode", False):
                            if len(self.employees) < getattr(self.config, "max_known_employees", 6):
                                # Still enrolling initial 6
                                new_emp_id = self.register_new_employee(track, frame_id, frame=frame)
                                track.identity_status = "TRACKED"
                                track.identity_event = "NEW"
                                track.identity_event_frame = frame_id
                                track.identity_confidence = 1.0
                                track.was_recovered = False
                            else:
                                # All 6 known identities filled, no match to dormant EMP -> UNKNOWN!
                                track.emp_id = getattr(self.config, "unmatched_label", "UNKNOWN")
                                track.identity_status = "UNKNOWN"
                                track.identity_event = "UNKNOWN"
                                track.identity_event_frame = frame_id
                                track.recovered_locked = True
                                track.identity_confidence = meta.get("best_score", 0.0)
                                track.was_recovered = False
                        elif self.config.auto_enroll_unknown:
                            new_emp_id = self.register_new_employee(
                                track, frame_id, frame=frame, initial_embedding=query_embeddings.get(track.track_id)
                            )
                            track.identity_status = "TRACKED"
                            track.identity_event = "NEW"
                            track.identity_event_frame = frame_id
                            track.identity_confidence = 1.0
                            track.was_recovered = False
                        else:
                            track.emp_id = getattr(self.config, "unmatched_label", "UNKNOWN")
                            track.identity_status = "UNKNOWN"
                            track.identity_event = "UNKNOWN"
                            track.identity_event_frame = frame_id
                            track.recovered_locked = True
                            track.identity_confidence = meta.get("best_score", 0.0)
                            track.was_recovered = False

        # =============================================================
        # PHASE 5: Collision Safe-Fail (Enforce Strict 1-to-1)
        # =============================================================
        seen_emp_ids: Dict[str, Track] = {}
        for track in sorted(active_tracks, key=lambda t: (t.hits, t.confidence), reverse=True):
            if track.emp_id is None or track.emp_id == getattr(self.config, "unmatched_label", "UNKNOWN"):
                continue
            if track.emp_id in seen_emp_ids:
                # Weaker track loses its EMP binding
                self._active_registry.release_track(track.track_id)
                track.emp_id = getattr(self.config, "unmatched_label", "UNKNOWN") if getattr(self.config, "fixed_mode", False) else None
                track.identity_status = "UNKNOWN"
            else:
                seen_emp_ids[track.emp_id] = track
                self._active_registry.register(track.emp_id, track.track_id)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_employee(self, emp_id: str) -> Optional[Employee]:
        return self.employees.get(emp_id)

    def is_emp_active(self, emp_id: str) -> bool:
        return self._active_registry.is_emp_active(emp_id)

    def get_active_registry(self) -> ActiveEmployeeRegistry:
        return self._active_registry

    def get_dormant_gallery(self) -> DormantIdentityGallery:
        return self._dormant_gallery
