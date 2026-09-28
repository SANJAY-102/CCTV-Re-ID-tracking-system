"""
Multi-Embedding Employee Gallery & Contamination Guard (StrongSORT-inspired).
Stores multiple high-quality feature vectors per employee across time and pose variations,
filters out low-quality/occluded samples, and performs robust multi-sample similarity matching.

Separates:
- Active identities (currently on screen, excluded from recovery)
- Dormant identities (departed, eligible for emergency Re-ID matching)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np

from ..config import ReIDConfig


@dataclass
class GalleryItem:
    """Individual validated high-quality feature vector with quality metadata."""
    embedding: np.ndarray      # L2-normalized 512-dim vector
    quality_score: float       # Detection confidence / sharpness score
    frame_id: int
    created_at: float


class EmployeeGallery:
    """
    Multi-embedding gallery manager.
    Maintains bounded appearance history for each persistent employee.
    Survives track expiration; embeddings are NEVER deleted on track death.
    """

    def __init__(self, config: Optional[ReIDConfig] = None):
        self.config = config or ReIDConfig()
        # Master mapping: emp_id -> List[GalleryItem]
        self._gallery: Dict[str, List[GalleryItem]] = {}

    @property
    def total_employees(self) -> int:
        return len(self._gallery)

    def has_employee(self, emp_id: str) -> bool:
        return emp_id in self._gallery and len(self._gallery[emp_id]) > 0

    def get_employee_ids(self) -> List[str]:
        return list(self._gallery.keys())

    def get_embeddings(self, emp_id: str) -> List[np.ndarray]:
        if emp_id not in self._gallery:
            return []
        return [item.embedding for item in self._gallery[emp_id]]

    def validate_sample_quality(
        self,
        crop: Optional[np.ndarray],
        bbox: np.ndarray,
        confidence: float
    ) -> Tuple[bool, str]:
        """
        Anti-contamination filter.
        Rejects occluded, blurred, or low-confidence samples.
        """
        if confidence < self.config.quality_min_confidence:
            return False, f"Low confidence: {confidence:.2f} < {self.config.quality_min_confidence}"

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        if height < self.config.quality_min_height:
            return False, f"Crop height too small: {height:.1f} < {self.config.quality_min_height}"
        if width < self.config.quality_min_width:
            return False, f"Crop width too small: {width:.1f} < {self.config.quality_min_width}"

        aspect_ratio = height / max(1.0, width)
        if aspect_ratio > self.config.quality_max_aspect_ratio or aspect_ratio < 0.5:
            return False, f"Unusual aspect ratio: {aspect_ratio:.2f}"

        if crop is None or crop.size == 0:
            return False, "Empty image crop"

        return True, "Valid"

    def add_embedding(
        self,
        emp_id: str,
        embedding: np.ndarray,
        quality_score: float,
        frame_id: int,
        timestamp: float = 0.0
    ) -> bool:
        """
        Add a new validated embedding to an employee's gallery.
        Maintains at most max_gallery_size_per_emp diverse high-quality items.
        """
        if embedding is None or len(embedding) == 0:
            return False

        # Ensure unit L2 normalization
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm

        item = GalleryItem(
            embedding=embedding.astype(np.float32),
            quality_score=quality_score,
            frame_id=frame_id,
            created_at=timestamp
        )

        if emp_id not in self._gallery:
            self._gallery[emp_id] = [item]
            return True

        current_items = self._gallery[emp_id]

        # If not full, simply append
        if len(current_items) < self.config.max_gallery_size_per_emp:
            current_items.append(item)
            return True

        # If gallery is full, replace lowest quality existing sample if new sample is better
        min_idx = int(np.argmin([it.quality_score for it in current_items]))
        if quality_score >= current_items[min_idx].quality_score:
            current_items[min_idx] = item
            return True

        return False

    def match(
        self,
        query_embedding: np.ndarray,
        eligible_emp_ids: Optional[Set[str]] = None,
        similarity_threshold: Optional[float] = None,
        min_margin: Optional[float] = None,
        exclude_emp_ids: Optional[Set[str]] = None
    ) -> Tuple[Optional[str], float, Dict[str, float], Dict[str, Any]]:
        """
        Compare query embedding against eligible dormant employees in the gallery.
        
        Args:
            query_embedding: 1D normalized feature vector.
            eligible_emp_ids: Set of DORMANT EMP-IDs eligible for recovery (preferred).
            similarity_threshold: Minimum cosine similarity required.
            min_margin: Minimum margin over second-best candidate to prevent ambiguity.
            exclude_emp_ids: Set of ACTIVE EMP-IDs (cannot be recovered/stolen).
            
        Returns:
            Tuple of:
                - matched_emp_id: Matched EMP-XXXX or None
                - best_score: Top cosine similarity score
                - all_scores: Dictionary of candidate scores {emp_id: score}
                - meta: Decision metadata for structured logging
        """
        thresh = similarity_threshold if similarity_threshold is not None else self.config.similarity_threshold
        req_margin = min_margin if min_margin is not None else self.config.min_margin
        
        # Determine candidate set
        exclude = exclude_emp_ids or set()
        
        default_meta = {
            "best_emp": None,
            "best_score": 0.0,
            "second_emp": None,
            "second_score": 0.0,
            "margin": float("inf"),
            "is_ambiguous": False,
            "all_scores": {}
        }

        if query_embedding is None or len(query_embedding) == 0 or not self._gallery:
            return None, 0.0, {}, default_meta

        # Ensure query is unit normalized
        q_norm = np.linalg.norm(query_embedding)
        if q_norm > 1e-6:
            q_emb = query_embedding / q_norm
        else:
            return None, 0.0, {}, default_meta

        all_scores: Dict[str, float] = {}

        for emp_id, gallery_items in self._gallery.items():
            # If explicit eligible set is provided, filter strictly
            if eligible_emp_ids is not None and emp_id not in eligible_emp_ids:
                continue
            # If excluded (e.g. actively tracked), skip
            if emp_id in exclude:
                continue
            if not gallery_items:
                continue

            # Compute cosine similarity against all gallery embeddings for this employee
            emp_embeddings = np.array([it.embedding for it in gallery_items], dtype=np.float32)
            sims = np.dot(emp_embeddings, q_emb)
            
            # Robust aggregate metric: top-2 mean or max (StrongSORT style)
            if len(sims) >= 2:
                top2 = np.partition(sims, -2)[-2:]
                score = float(np.mean(top2))
            else:
                score = float(np.max(sims))

            all_scores[emp_id] = round(score, 4)

        if not all_scores:
            return None, 0.0, {}, default_meta

        # Sort candidate scores in descending order
        sorted_candidates = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        best_emp_id, best_score = sorted_candidates[0]
        
        second_emp_id = None
        second_score = 0.0
        margin = float("inf")
        
        if len(sorted_candidates) > 1:
            second_emp_id, second_score = sorted_candidates[1]
            margin = round(best_score - second_score, 4)

        is_ambiguous = (margin < req_margin) if len(sorted_candidates) > 1 else False

        meta = {
            "best_emp": best_emp_id,
            "best_score": best_score,
            "second_emp": second_emp_id,
            "second_score": second_score,
            "margin": margin,
            "is_ambiguous": is_ambiguous,
            "all_scores": all_scores
        }

        # Validation Conditions:
        # 1. Best score >= similarity threshold
        # 2. Margin over second candidate >= min_margin (fail-safe for ambiguous matches)
        if best_score >= thresh and margin >= req_margin and best_emp_id is not None:
            return best_emp_id, best_score, all_scores, meta

        return None, best_score, all_scores, meta
