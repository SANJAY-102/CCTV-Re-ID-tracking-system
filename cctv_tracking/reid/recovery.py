"""
Emergency Re-ID Recovery Engine.

ONLY ACTIVATES when a track is completely confirmed as NEW by ByteTrack
and dormant identities exist in the gallery.

Uses:
- OSNet extractor (proper person Re-ID model, not ImageNet MobileNet)
- DormantIdentityGallery for candidate set (active EMPs excluded absolutely)

Re-ID Frequency Contract:
  Healthy tracked person:        0 Re-ID calls
  Temporarily lost (LOST state): 0 Re-ID calls
  Tracker recovery:              0 Re-ID calls
  Track expires + returns:       1 Re-ID call on new track confirmation
"""

from typing import Optional, Tuple, Set, Dict, Any
import time
import numpy as np

from .model import ReIDExtractor
from .gallery import EmployeeGallery  # Legacy compatibility alias
from ..config import ReIDConfig


class EmergencyReIDRecovery:
    """
    Emergency OSNet Re-ID Recovery Service.
    Answers the single question: 'Is this new confirmed track a returning dormant employee?'

    The gallery parameter is kept for backward compatibility with existing tests;
    the IdentityManager now uses DormantIdentityGallery directly for matching,
    but this class still owns the OSNet extractor and the counter telemetry.
    """

    def __init__(
        self,
        extractor: ReIDExtractor,
        gallery: EmployeeGallery,
        config: Optional[ReIDConfig] = None
    ):
        self.extractor = extractor
        self.gallery = gallery  # Legacy EmployeeGallery kept for test compatibility
        self.config = config or ReIDConfig()

        # Strict separated performance telemetry (Section 1)
        self.reid_inference_calls: int = 0          # Actual OSNet forward passes
        self.recovery_attempts: int = 0             # Candidate tracks entering recovery session
        self.successful_identity_recoveries: int = 0 # Genuine successful restorations of dormant EMPs
        self.recovery_failed_count: int = 0

    @property
    def reid_calls_count(self) -> int:
        return self.reid_inference_calls

    @reid_calls_count.setter
    def reid_calls_count(self, val: int) -> None:
        self.reid_inference_calls = val

    @property
    def recovery_success_count(self) -> int:
        return self.successful_identity_recoveries

    @recovery_success_count.setter
    def recovery_success_count(self, val: int) -> None:
        self.successful_identity_recoveries = val

    def recover_identity(
        self,
        person_crop: np.ndarray,
        bbox: np.ndarray,
        confidence: float,
        frame_id: int,
        dormant_emp_ids: Set[str],
        active_emp_ids: Optional[Set[str]] = None,
        lost_memory: Optional[Dict[str, dict]] = None
    ) -> Tuple[Optional[str], float, bool, np.ndarray, Dict[str, Any]]:
        """
        Run ONE emergency OSNet Re-ID inference against dormant employees.

        Args:
            person_crop: Person image crop.
            bbox: Bounding box [x1, y1, x2, y2].
            confidence: Detection confidence.
            frame_id: Current frame number.
            dormant_emp_ids: ONLY these EMPs are queried (the dormant set).
            active_emp_ids: These EMPs are EXCLUDED from matching (protected active IDs).
            lost_memory: Unused; preserved for API compatibility.

        Returns:
            (matched_emp_id, sim_score, is_new_enrollment, embedding, meta)
        """
        # Increment strictly monitored counter
        self.reid_calls_count += 1

        # Extract deep OSNet person Re-ID embedding
        embedding = self.extractor.extract(person_crop)

        # Query ONLY the provided dormant candidates
        # Exclude any active EMPs for absolute safety
        eligible = dormant_emp_ids - (active_emp_ids or set())

        matched_emp_id, sim_score, all_scores, meta = self.gallery.match(
            query_embedding=embedding,
            eligible_emp_ids=eligible,
            similarity_threshold=self.config.similarity_threshold,
            min_margin=self.config.min_margin,
            exclude_emp_ids=active_emp_ids
        )

        if matched_emp_id is not None:
            self.recovery_success_count += 1
            # Add embedding to gallery for future robustness
            is_valid, _ = self.gallery.validate_sample_quality(person_crop, bbox, confidence)
            if is_valid:
                self.gallery.add_embedding(
                    emp_id=matched_emp_id,
                    embedding=embedding,
                    quality_score=confidence,
                    frame_id=frame_id,
                    timestamp=time.time()
                )
        else:
            self.recovery_failed_count += 1

        return matched_emp_id, sim_score, False, embedding, meta
