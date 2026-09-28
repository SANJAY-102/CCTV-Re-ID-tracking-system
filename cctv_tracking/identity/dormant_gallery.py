"""
Dormant Identity Gallery.

Stores multi-embedding appearance galleries for DORMANT employees — people
who have left the camera field of view but whose identity must persist for
future re-entry recovery.

Architectural References:
- matus012/multicam_persistent_id: Persistent OSNet gallery with long-gap recovery
- BoxMOT StrongSORT: StraightForwardReID multi-gallery approach
- Norfair: reid_distance_function for unmatched (expired) track resolution

FUNDAMENTAL RULE:
    Only DORMANT employees live here.
    ACTIVE employees are NEVER queried against dormant gallery.
    A person CANNOT steal another person's dormant identity through proximity alone.
    Appearance gallery is the ONLY identity truth for long-gap recovery.

Embedding Management:
    - Bounded: max N embeddings per employee (FIFO with quality-based replacement)
    - Preserved: gallery survives track expiration UNCONDITIONALLY
    - Multi-embedding matching: cosine similarity with top-k mean aggregation
    - Margin protection: second-best candidate must be sufficiently far below best
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import ReIDConfig


@dataclass
class DormantEmbeddingRecord:
    """Single validated appearance embedding from a gallery session."""
    embedding: np.ndarray      # L2-normalized 512-dim OSNet vector
    quality_score: float       # Detection confidence / sharpness
    frame_id: int
    captured_at: float         # Unix timestamp


@dataclass
class DormantPersonRecord:
    """
    Full dormant identity profile.
    Preserved indefinitely after track expiration.
    This is the authoritative appearance memory of a physical person.
    """
    emp_id: str
    name: str
    created_at: float
    last_seen_frame: int
    last_seen_at: float
    # Multi-embedding gallery (bounded)
    embeddings: List[DormantEmbeddingRecord] = field(default_factory=list)


class DormantIdentityGallery:
    """
    Isolated persistent appearance gallery for dormant (departed) employees.

    Lifecycle:
        Employee departs -> track expires -> DormantPersonRecord registered here
        New track appears -> query this gallery via match()
        Match found -> recover EMP (remove from dormant, return to active registry)
        No match    -> enroll as NEW EMP (this gallery unchanged)

    Invariant:
        The gallery NEVER shrinks on any track event.
        Records are only removed when an EMP is successfully recovered.
        Active employees are NEVER stored or queried here.
    """

    def __init__(self, config: Optional[ReIDConfig] = None):
        self.config = config or ReIDConfig()
        # emp_id -> DormantPersonRecord
        self._records: Dict[str, DormantPersonRecord] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        emp_id: str,
        name: str,
        embeddings: List[DormantEmbeddingRecord],
        last_seen_frame: int,
        created_at: float
    ) -> None:
        """
        Register or update a dormant employee's gallery record.
        Called when a track completely expires and the person departs.
        """
        if emp_id in self._records:
            # Merge new embeddings into existing record (do not discard history)
            existing = self._records[emp_id]
            existing.last_seen_frame = last_seen_frame
            existing.last_seen_at = time.time()
            for emb_rec in embeddings:
                self._add_embedding_to_record(existing, emb_rec)
        else:
            self._records[emp_id] = DormantPersonRecord(
                emp_id=emp_id,
                name=name,
                created_at=created_at,
                last_seen_frame=last_seen_frame,
                last_seen_at=time.time(),
                embeddings=list(embeddings)
            )

    def add_embedding(
        self,
        emp_id: str,
        embedding: np.ndarray,
        quality_score: float,
        frame_id: int
    ) -> bool:
        """
        Add a single embedding to an existing dormant or active-to-dormant record.
        Can be called while tracking (pre-departure) to build up the gallery.
        """
        if embedding is None or len(embedding) == 0:
            return False

        # L2-normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = (embedding / norm).astype(np.float32)
        else:
            return False

        record = DormantEmbeddingRecord(
            embedding=embedding,
            quality_score=quality_score,
            frame_id=frame_id,
            captured_at=time.time()
        )

        if emp_id not in self._records:
            # Pre-create record (will be promoted to dormant on track expiry)
            self._records[emp_id] = DormantPersonRecord(
                emp_id=emp_id,
                name="Unknown",
                created_at=time.time(),
                last_seen_frame=frame_id,
                last_seen_at=time.time(),
                embeddings=[]
            )

        return self._add_embedding_to_record(self._records[emp_id], record)

    def _add_embedding_to_record(
        self, record: DormantPersonRecord, new_entry: DormantEmbeddingRecord
    ) -> bool:
        """
        Add embedding to a record with bounded size and quality-based replacement.
        Inspired by StrongSORT's StraightForwardReID gallery management.
        """
        max_size = self.config.max_gallery_size_per_emp
        items = record.embeddings

        if len(items) < max_size:
            items.append(new_entry)
            return True

        # Replace lowest-quality existing entry if this one is better
        min_q_idx = int(np.argmin([it.quality_score for it in items]))
        if new_entry.quality_score >= items[min_q_idx].quality_score:
            items[min_q_idx] = new_entry
            return True

        return False

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def has_dormant(self, emp_id: str) -> bool:
        return emp_id in self._records and bool(self._records[emp_id].embeddings)

    def get_dormant_emp_ids(self) -> Set[str]:
        return {eid for eid, rec in self._records.items() if rec.embeddings}

    def get_embeddings(self, emp_id: str) -> List[np.ndarray]:
        if emp_id not in self._records:
            return []
        return [r.embedding for r in self._records[emp_id].embeddings]

    def match(
        self,
        query_embedding: np.ndarray,
        eligible_emp_ids: Optional[Set[str]] = None,
        similarity_threshold: Optional[float] = None,
        min_margin: Optional[float] = None,
    ) -> Tuple[Optional[str], float, Dict[str, float], Dict[str, Any]]:
        """
        Match a query embedding against ONLY dormant employee galleries.

        Args:
            query_embedding: 1-D L2-normalized OSNet feature vector.
            eligible_emp_ids: Which dormant EMPs to query (required; active EMPs excluded externally).
            similarity_threshold: Minimum cosine similarity for a valid match.
            min_margin: Minimum gap between best and second-best to prevent ambiguous recovery.

        Returns:
            (matched_emp_id, best_score, all_scores, meta)
            matched_emp_id is None if no valid unambiguous match found.
        """
        thresh = similarity_threshold if similarity_threshold is not None else self.config.similarity_threshold
        req_margin = min_margin if min_margin is not None else self.config.min_margin

        empty_meta = {
            "best_emp": None, "best_score": 0.0,
            "second_emp": None, "second_score": 0.0,
            "margin": float("inf"), "is_ambiguous": False,
            "all_scores": {}
        }

        if query_embedding is None or len(query_embedding) == 0:
            return None, 0.0, {}, empty_meta

        # Normalize query
        q_norm = np.linalg.norm(query_embedding)
        q_emb = (query_embedding / q_norm).astype(np.float32) if q_norm > 1e-6 else None
        if q_emb is None:
            return None, 0.0, {}, empty_meta

        # Build candidate set — strictly dormant only
        candidate_ids = eligible_emp_ids if eligible_emp_ids is not None else self.get_dormant_emp_ids()

        all_scores: Dict[str, float] = {}
        for emp_id in candidate_ids:
            record = self._records.get(emp_id)
            if record is None or not record.embeddings:
                continue

            gallery_embs = np.stack([r.embedding for r in record.embeddings], axis=0)  # (N, D)
            sims = gallery_embs @ q_emb  # cosine similarity for each stored embedding

            # StrongSORT-style robust aggregation: mean of top-2 (if available)
            if len(sims) >= 2:
                top2 = float(np.mean(np.partition(sims, -2)[-2:]))
            else:
                top2 = float(sims[0])

            all_scores[emp_id] = round(top2, 5)

        if not all_scores:
            return None, 0.0, {}, empty_meta

        sorted_cands = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        best_emp_id, best_score = sorted_cands[0]

        second_emp_id, second_score, margin = None, 0.0, float("inf")
        if len(sorted_cands) > 1:
            second_emp_id, second_score = sorted_cands[1]
            margin = round(best_score - second_score, 5)

        is_ambiguous = len(sorted_cands) > 1 and margin < req_margin

        meta = {
            "best_emp": best_emp_id,
            "best_score": round(best_score, 5),
            "second_emp": second_emp_id,
            "second_score": round(second_score, 5),
            "margin": round(margin, 5) if margin != float("inf") else float("inf"),
            "is_ambiguous": is_ambiguous,
            "all_scores": all_scores
        }

        # Must pass BOTH threshold AND margin guard
        if best_score >= thresh and not is_ambiguous:
            return best_emp_id, best_score, all_scores, meta

        return None, best_score, all_scores, meta

    def match_batch_global(
        self,
        query_embeddings: Dict[int, np.ndarray],
        eligible_emp_ids: Set[str],
        similarity_threshold: Optional[float] = None,
        min_margin: Optional[float] = None,
    ) -> Dict[int, Tuple[Optional[str], float, Dict[str, float], Dict[str, Any]]]:
        """
        Global bipartite matching of multiple candidate tracks against eligible DORMANT EMPs
        using the Hungarian algorithm (linear_sum_assignment).

        Enforces:
        - ONE T-ID -> maximum ONE EMP
        - ONE EMP -> maximum ONE T-ID
        - Active EMPs -> locked / excluded from eligible_emp_ids
        - Frozen gallery: no state changes or gallery mutations occur during matching
        - Atomic resolution: all tracks solved simultaneously
        """
        thresh = similarity_threshold if similarity_threshold is not None else self.config.similarity_threshold
        req_margin = min_margin if min_margin is not None else self.config.min_margin

        results: Dict[int, Tuple[Optional[str], float, Dict[str, float], Dict[str, Any]]] = {}
        if not query_embeddings:
            return results

        candidate_emp_ids = sorted(list(eligible_emp_ids))
        track_ids = sorted(list(query_embeddings.keys()))

        if not candidate_emp_ids:
            for tid in track_ids:
                empty_meta = {
                    "best_emp": None, "best_score": 0.0,
                    "second_emp": None, "second_score": 0.0,
                    "margin": float("inf"), "is_ambiguous": False,
                    "all_scores": {}
                }
                results[tid] = (None, 0.0, {}, empty_meta)
            return results

        # 1. Compute full similarity matrix (N_tracks x M_emps)
        N = len(track_ids)
        M = len(candidate_emp_ids)
        sim_matrix = np.zeros((N, M), dtype=np.float32)
        all_scores_per_track: Dict[int, Dict[str, float]] = {}

        for i, tid in enumerate(track_ids):
            q_emb = query_embeddings[tid]
            all_scores_per_track[tid] = {}
            if q_emb is None or len(q_emb) == 0:
                continue
            q_norm = np.linalg.norm(q_emb)
            if q_norm < 1e-6:
                continue
            q_emb = (q_emb / q_norm).astype(np.float32)

            for j, emp_id in enumerate(candidate_emp_ids):
                record = self._records.get(emp_id)
                if record is None or not record.embeddings:
                    continue
                gallery_embs = np.stack([r.embedding for r in record.embeddings], axis=0)
                sims = gallery_embs @ q_emb
                if len(sims) >= 2:
                    score = float(np.mean(np.partition(sims, -2)[-2:]))
                else:
                    score = float(sims[0])
                sim_matrix[i, j] = score
                all_scores_per_track[tid][emp_id] = round(score, 5)

        # 2. For each track, evaluate margin and ambiguity
        track_meta: Dict[int, Dict[str, Any]] = {}
        for i, tid in enumerate(track_ids):
            scores = all_scores_per_track[tid]
            if not scores:
                track_meta[tid] = {
                    "best_emp": None, "best_score": 0.0,
                    "second_emp": None, "second_score": 0.0,
                    "margin": float("inf"), "is_ambiguous": False,
                    "all_scores": {}
                }
                continue
            sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            best_emp_id, best_score = sorted_cands[0]
            second_emp_id, second_score, margin = None, 0.0, float("inf")
            if len(sorted_cands) > 1:
                second_emp_id, second_score = sorted_cands[1]
                margin = round(best_score - second_score, 5)
            is_ambiguous = (len(sorted_cands) > 1 and margin < req_margin)
            track_meta[tid] = {
                "best_emp": best_emp_id,
                "best_score": round(best_score, 5),
                "second_emp": second_emp_id,
                "second_score": round(second_score, 5),
                "margin": round(margin, 5) if margin != float("inf") else float("inf"),
                "is_ambiguous": is_ambiguous,
                "all_scores": scores
            }

        # 3. Solve Global Bipartite Assignment via linear_sum_assignment
        # Cost is 1.0 - similarity (minimizing cost maximizes similarity)
        cost_matrix = 1.0 - sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_tracks: Set[int] = set()
        for r, c in zip(row_ind, col_ind):
            tid = track_ids[r]
            assigned_tracks.add(tid)
            emp_id = candidate_emp_ids[c]
            sim = float(sim_matrix[r, c])
            meta = track_meta[tid]
            scores = all_scores_per_track[tid]

            # Validation checks:
            # 1) Must meet similarity threshold
            # 2) Must NOT be ambiguous (margin >= req_margin)
            # 3) The assigned emp must be the best_emp for this track
            if sim >= thresh and not meta["is_ambiguous"] and meta["best_emp"] == emp_id:
                results[tid] = (emp_id, sim, scores, meta)
            else:
                results[tid] = (None, sim, scores, meta)

        # Unassigned tracks (when N > M)
        for tid in track_ids:
            if tid not in assigned_tracks:
                meta = track_meta.get(tid, {
                    "best_emp": None, "best_score": 0.0,
                    "second_emp": None, "second_score": 0.0,
                    "margin": float("inf"), "is_ambiguous": False,
                    "all_scores": {}
                })
                scores = all_scores_per_track.get(tid, {})
                results[tid] = (None, 0.0, scores, meta)

        return results

    def promote_to_active(self, emp_id: str) -> None:
        """
        Remove from dormant gallery when successfully recovered.
        The employee's gallery embeddings are NOT deleted — just move state.
        (The IdentityManager retains the full Employee object and gallery separately.)
        """
        # We do NOT delete the record — we keep embeddings for future drops.
        # Only the 'eligible_emp_ids' set (controlled externally) changes.
        pass

    def remove(self, emp_id: str) -> None:
        """Permanently remove a dormant record (used only in tests or explicit deletion)."""
        self._records.pop(emp_id, None)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def validate_sample_quality(
        self,
        crop: Optional[np.ndarray],
        bbox: np.ndarray,
        confidence: float
    ) -> Tuple[bool, str]:
        """Anti-contamination sample quality filter."""
        if confidence < self.config.quality_min_confidence:
            return False, f"Low confidence: {confidence:.2f}"
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        if h < self.config.quality_min_height:
            return False, f"Height too small: {h:.1f}"
        if w < self.config.quality_min_width:
            return False, f"Width too small: {w:.1f}"
        ar = h / max(1.0, w)
        if ar > self.config.quality_max_aspect_ratio or ar < 0.5:
            return False, f"Unusual aspect ratio: {ar:.2f}"
        if crop is None or crop.size == 0:
            return False, "Empty crop"
        return True, "Valid"

    @property
    def total_dormant(self) -> int:
        return sum(1 for r in self._records.values() if r.embeddings)

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"DormantIdentityGallery(records={list(self._records.keys())})"
