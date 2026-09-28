"""
Configuration module for CCTV Person Tracking & Emergency Identity Recovery System.
Contains strongly typed dataclasses and sensible defaults optimized for surveillance feeds.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import os
import torch


@dataclass
class DetectorConfig:
    """YOLO Person Detector Configuration."""
    model_name: str = "yolov8n.pt"  # Lightweight nano model by default (can be yolov8s.pt, yolov8m.pt)
    conf_thresh: float = 0.45       # Minimum detection confidence for primary tracker
    iou_thresh: float = 0.45        # NMS IoU threshold
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    img_size: int = 640
    person_class_id: int = 0        # COCO class 0 is 'person'


@dataclass
class TrackerConfig:
    """MotionAwareByteTrack Primary Tracker Configuration."""
    # 2-stage association thresholds
    high_det_thresh: float = 0.50          # High-confidence detection threshold (Stage 1)
    low_det_thresh: float = 0.20           # Low-confidence detection threshold (Stage 2)
    new_track_thresh: float = 0.60         # Minimum confidence to initialize a tentative track
    
    # Matching cost thresholds (1.0 - metric)
    match_thresh_first: float = 0.85       # Maximum allowed cost for high-conf match (accommodates standing/walking)
    match_thresh_second: float = 0.70      # Maximum allowed IoU distance for low-conf match (occlusions/motion blur)
    match_thresh_unconfirmed: float = 0.75 # Matching threshold for tentative tracks
    
    # Motion & Kalman settings
    max_lost_frames: int = 120             # Number of frames to hold lost tracks before deletion (4 sec @ 30fps)
    min_hits_to_confirm: int = 4           # Consecutive frames to confirm tentative track (filters out noise)
    
    # Motion classification
    stationary_speed_thresh: float = 2.5   # Pixels/frame below which track is classified as STATIONARY
    motion_history_len: int = 15           # Number of frames used to calculate smoothed velocity/direction
    
    # Advanced Tracking & Motion Enhancements
    use_gmc: bool = False                  # Disabled by default for static CCTV cameras to prevent ORB drift
    gmc_method: str = "orb"                # "orb" or "sparse_optflow"
    use_nsa: bool = True                   # Noise Scale Adaptive Kalman filtering (scales by 1 - conf)
    ocm_weight: float = 0.15               # Observation-Centric Momentum directional cost weight
    duplicate_iou_thresh: float = 0.50     # Overlap threshold to suppress duplicate tracks on same person
    
    # Trajectory
    max_trajectory_len: int = 30           # Clean concise trajectory trail


@dataclass
class ReIDConfig:
    """Emergency Re-ID Recovery Configuration."""
    enabled: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_dim: int = 512
    backbone: str = "osnet_x1_0"           # OSNet deep Person Re-ID architecture
    weights_path: Optional[str] = None
    
    # Matching parameters
    similarity_threshold: float = 0.55     # Cosine similarity required to recover EMP-ID using Market1501 OSNet
    min_margin: float = 0.05               # Minimum score difference over second-best candidate to prevent ambiguity
    max_gallery_size_per_emp: int = 8      # Maximum diverse high-quality embeddings per employee
    
    # Sample Quality Filter (Prevents Gallery Contamination)
    quality_min_confidence: float = 0.70   # Only store gallery samples from confident detections
    quality_min_height: int = 60           # Ignore blurry/tiny person crops
    quality_min_width: int = 30
    quality_max_aspect_ratio: float = 4.0  # Max height/width ratio


@dataclass
class IdentityConfig:
    """Identity Ownership and Employee Management Configuration."""
    emp_prefix: str = "EMP"
    tid_prefix: str = "T-"
    auto_enroll_unknown: bool = False      # In fixed mode: False (assigns UNKNOWN instead of EMP007)
    min_track_frames_for_enrollment: int = 3 # Prevent enrolling noise/glitches as employees
    enable_debug_logging: bool = True      # Explicit identity decision trace logging
    
    # Hard-Coded 6-Person Identity Mode
    fixed_mode: bool = True
    max_known_employees: int = 6
    fixed_emp_ids: List[str] = field(default_factory=lambda: [
        "EMP001", "EMP002", "EMP003", "EMP004", "EMP005", "EMP006"
    ])
    unmatched_label: str = "UNKNOWN"
    min_gallery_samples: int = 5           # Accumulate multiple gallery embeddings per known employee
    disable_dynamic_gallery: bool = True   # Section 14: Freeze gallery after enrollment to prevent contamination


@dataclass
class DatabaseConfig:
    """Persistence Database Configuration."""
    db_type: str = "json"                  # "json" or "sqlite"
    storage_dir: str = os.path.join(os.path.expanduser("~"), ".cctv_tracking_db")
    db_name: str = "employee_registry.json"


@dataclass
class VisualizationConfig:
    """Display and CCTV Annotations Configuration."""
    draw_box: bool = True
    draw_tracking_point: bool = True
    draw_trajectory: bool = True
    draw_hud: bool = True
    event_display_frames: int = 45         # Show one-time events (RECOVERED, NEW) for N frames then transition to TRACKED
    
    # Colors in BGR format
    color_moving: Tuple[int, int, int] = (0, 235, 120)     # Neon Green
    color_stationary: Tuple[int, int, int] = (0, 215, 255) # Bright Yellow / Amber
    color_lost: Tuple[int, int, int] = (40, 100, 255)      # Orange / Warning
    color_recovered: Tuple[int, int, int] = (255, 50, 200) # Bright Magenta / Violet
    color_unknown: Tuple[int, int, int] = (0, 140, 255)    # Deep Orange for UNKNOWN persons
    color_hud_bg: Tuple[int, int, int] = (25, 25, 30)      # Dark translucent background
    color_text: Tuple[int, int, int] = (255, 255, 255)     # Crisp White


@dataclass
class AppConfig:
    """Master Application Configuration."""
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    @classmethod
    def default(cls) -> 'AppConfig':
        return cls()
    
    @classmethod
    def fast_cpu(cls) -> 'AppConfig':
        """Preset optimized for CPU-only or edge devices."""
        cfg = cls()
        cfg.detector.model_name = "yolov8n.pt"
        cfg.detector.device = "cpu"
        cfg.detector.img_size = 480
        cfg.reid.device = "cpu"
        cfg.tracker.max_lost_frames = 45
        return cfg
