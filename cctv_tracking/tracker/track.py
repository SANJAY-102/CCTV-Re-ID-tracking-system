"""
Track Representation and Lifecycle State Machine with OCM and Camera Motion.
Encapsulates NSA Kalman state, T-ID, persistent EMP-ID linkage, motion kinematics,
observation-centric momentum, and trajectory history.
"""

from collections import deque
from enum import Enum
from typing import Optional, Tuple, List
import numpy as np

from .kalman import KalmanFilter
from .motion import MotionAnalyzer, MotionState, Direction
from .trajectory import Trajectory
from ..detector.yolo_detector import PersonDetection


class TrackState(str, Enum):
    TENTATIVE = "TENTATIVE"    # Fresh track awaiting confirmation (min_hits)
    TRACKED = "TRACKED"        # Actively tracked in current frame
    LOST = "LOST"              # Missed in current frame, held in lost buffer
    DELETED = "DELETED"        # Expired beyond max_lost_frames (tracker completely failed)


class Track:
    """
    Core track entity with NSA Kalman filtering, OCM momentum vector,
    and camera motion compensation.
    """

    _count = 0

    def __init__(
        self,
        detection: PersonDetection,
        frame_id: int,
        kalman_filter: KalmanFilter,
        max_trajectory_len: int = 45,
        stationary_thresh: float = 2.5
    ):
        Track._count += 1
        self.track_id: int = Track._count
        self.tid_str: str = f"T-{self.track_id:03d}"
        
        # Persistent Identity Linkage (Assigned and governed strictly by IdentityManager)
        self.emp_id: Optional[str] = None
        
        self.state: TrackState = TrackState.TENTATIVE
        self.kalman_filter = kalman_filter
        self.stationary_thresh = stationary_thresh
        
        # Initialize Kalman state with [cx, cy, a, h]
        cx, cy = detection.center
        w, h = detection.width, detection.height
        a = w / max(1.0, h)
        measurement = np.array([cx, cy, a, h], dtype=np.float32)
        self.mean, self.covariance = self.kalman_filter.initiate(measurement)
        
        # Current geometry & detection info
        self.bbox: np.ndarray = detection.bbox.copy()
        self.confidence: float = detection.confidence
        self.center: Tuple[float, float] = detection.center
        self.bottom_center: Tuple[float, float] = detection.bottom_center
        
        # Timing & Lifecycle counters
        self.start_frame: int = frame_id
        self.frame_id: int = frame_id
        self.age: int = 1
        self.hits: int = 1
        self.lost_frames: int = 0
        self.was_recovered: bool = False
        self.tracker_recovered: bool = False
        self.identity_status: str = "NEW"  # "NEW", "TRACKED", "UNKNOWN"
        self.identity_confidence: float = 1.0
        
        # Identity Session & One-Time Event Lifecycle (Section 2, 3, 4, 7)
        self.recovery_attempted: bool = False
        self.identity_event: Optional[str] = "NEW"  # "NEW", "RECOVERED", "UNKNOWN", or None
        self.identity_event_frame: int = frame_id
        self.recovered_locked: bool = False
        self.recovered_from_emp: Optional[str] = None
        
        # Raw Observation History for OCM (Observation-Centric Momentum)
        self.observations: deque = deque(maxlen=10)
        self.observations.append(self.center)

        # Motion & Trajectory
        self.trajectory = Trajectory(max_length=max_trajectory_len)
        self.trajectory.add_point(self.bottom_center, frame_id)
        
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.speed: float = 0.0
        self.direction_deg: float = 0.0
        self.direction: Direction = Direction.STATIONARY
        self.motion_state: MotionState = MotionState.STATIONARY
        
        # Last known high-quality crop
        self.last_crop: Optional[np.ndarray] = detection.crop

    @classmethod
    def reset_counter(cls):
        """Reset internal track ID counter (for tests)."""
        cls._count = 0

    @property
    def is_confirmed(self) -> bool:
        return self.state in (TrackState.TRACKED, TrackState.LOST) and self.hits >= 3

    @property
    def is_active(self) -> bool:
        return self.state == TrackState.TRACKED

    @property
    def is_lost(self) -> bool:
        return self.state == TrackState.LOST

    @property
    def is_deleted(self) -> bool:
        return self.state == TrackState.DELETED

    def get_ocm_momentum_vector(self, delta_t: int = 3) -> Tuple[float, float, float]:
        """Return Observation-Centric Momentum (OCM) velocity vector (vx, vy, speed)."""
        return MotionAnalyzer.compute_ocm_momentum_vector(list(self.observations), delta_t=delta_t)

    def apply_camera_motion(self, H: np.ndarray):
        """Compensate Kalman filter and geometry for camera background shift."""
        if H is None:
            return
        self.mean, self.covariance = self.kalman_filter.apply_camera_motion(self.mean, self.covariance, H)
        self.bbox = self.to_tlbr()
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        self.bottom_center = ((x1 + x2) / 2.0, float(y2))

    def to_tlbr(self) -> np.ndarray:
        """Get bounding box [x1, y1, x2, y2] from Kalman state mean."""
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]  # w = a * h
        ret[:2] -= ret[2:] / 2  # top-left x, y
        ret[2:] += ret[:2]  # bottom-right x, y
        return ret.astype(np.float32)

    def to_tlwh(self) -> np.ndarray:
        """Get bounding box [x, y, w, h] from Kalman state mean."""
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret.astype(np.float32)

    def predict(self):
        """Run Kalman motion prediction with adaptive velocity damping."""
        if self.lost_frames > 2:
            # Zero out velocities ONLY when missing for multiple frames to prevent runaway drift
            self.mean[4:8] = 0.0
        else:
            # Gentle velocity damping to maintain smooth walking momentum
            self.mean[4:8] *= 0.88

        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
        self.age += 1

    def update(
        self,
        detection: "PersonDetection",
        frame_id: int,
        camera_affine: Optional[np.ndarray] = None
    ):
        """
        Update track state with matched detection.
        """
        # Apply GMC (Global Motion Compensation) transformation
        if camera_affine is not None:
            self.mean, self.covariance = self.kalman_filter.apply_camera_motion(
                self.mean, self.covariance, camera_affine
            )

        # Kalman measurement update with NSA adaptive measurement noise
        cx, cy = detection.center
        w, h = detection.width, detection.height
        a = w / max(1.0, h)
        measurement = np.array([cx, cy, a, h], dtype=np.float32)
        
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, measurement, confidence=detection.confidence
        )
        
        # Update geometry
        self.bbox = detection.bbox.copy()
        self.confidence = detection.confidence
        self.center = detection.center
        self.bottom_center = detection.bottom_center
        if detection.crop is not None:
            self.last_crop = detection.crop
            
        self.frame_id = frame_id
        self.hits += 1
        
        # Append observation for OCM
        self.observations.append(self.center)
        
        # Transient 1-frame tracker recovery flag (for internal tracker logic, not identity)
        self.tracker_recovered = (self.state == TrackState.LOST or self.lost_frames > 0)
        
        self.lost_frames = 0
        self.state = TrackState.TRACKED
        
        # Add tracking point to trajectory
        self.trajectory.add_point(self.bottom_center, frame_id)
        
        # Update motion kinematics
        recent_pts = self.trajectory.get_recent_points(6)
        self.vx, self.vy, self.speed = MotionAnalyzer.calculate_velocity(recent_pts)
        self.direction_deg, self.direction = MotionAnalyzer.calculate_direction(
            self.vx, self.vy, self.speed, self.stationary_thresh
        )
        self.motion_state = MotionAnalyzer.classify_state(
            self.speed,
            is_lost=False,
            stationary_thresh=self.stationary_thresh
        )

    def mark_missed(self, max_lost_frames: int):
        """
        Mark track as missing detection in current frame.
        """
        self.lost_frames += 1
        if self.lost_frames > max_lost_frames:
            self.state = TrackState.DELETED
        else:
            self.state = TrackState.LOST
            
        self.motion_state = MotionState.LOST
        self.tracker_recovered = False
