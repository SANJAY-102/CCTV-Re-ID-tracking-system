"""
Employee Entity and Identity Data Structures.
Represents persistent physical personnel identities (EMP-XXXX), distinct from
ephemeral tracker IDs (T-XXX).

Separates:
- Track State (ephemeral, owned by ByteTrack)
- Employee Identity State (persistent, owned by IdentityManager / Gallery)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import time
import numpy as np


class EmployeeState(Enum):
    """Lifecycle state of an Employee identity."""
    ACTIVE = "ACTIVE"       # Currently visible on screen or in temporary tracker lost buffer
    DORMANT = "DORMANT"     # Departed from camera / track expired; gallery preserved for future recovery


@dataclass
class Employee:
    """
    Persistent physical employee profile.
    Survives track expiration and camera departures.
    """
    emp_id: str                                      # 'EMP-0001'
    name: str = "Unknown"
    state: EmployeeState = EmployeeState.ACTIVE      # ACTIVE or DORMANT
    
    # 1-to-1 Track Binding
    current_track_id: Optional[int] = None           # e.g. 1 (None when DORMANT)
    current_tid_str: Optional[str] = None            # e.g. 'T-001' (None when DORMANT)
    associated_tids: List[str] = field(default_factory=list)  # Historical tracks: ['T-001', 'T-003']
    
    # Lifecycle Timings
    created_at: float = field(default_factory=time.time)
    last_seen_timestamp: float = field(default_factory=time.time)
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    total_frames_tracked: int = 0
    
    # Contextual Spatial / Kinematic Metadata (Secondary context only, NOT identity truth)
    last_known_bbox: Optional[np.ndarray] = None
    last_known_bottom_center: Optional[Tuple[float, float]] = None
    last_known_velocity: Optional[Tuple[float, float]] = None
    last_known_direction: Optional[str] = None
    last_known_embedding: Optional[np.ndarray] = None

    @property
    def is_active(self) -> bool:
        """True if person is currently visible and bound to an active track."""
        return self.state == EmployeeState.ACTIVE

    @property
    def is_dormant(self) -> bool:
        """True if person has departed and is currently in the dormant recovery gallery."""
        return self.state == EmployeeState.DORMANT

    @property
    def active_tid(self) -> Optional[str]:
        """Backward compatibility alias for current_tid_str."""
        return self.current_tid_str

    @active_tid.setter
    def active_tid(self, value: Optional[str]):
        self.current_tid_str = value

    def to_dict(self) -> Dict[str, Any]:
        """Serialize employee metadata for database persistence."""
        return {
            "emp_id": self.emp_id,
            "name": self.name,
            "state": self.state.value,
            "current_track_id": self.current_track_id,
            "current_tid_str": self.current_tid_str,
            "associated_tids": self.associated_tids,
            "created_at": self.created_at,
            "last_seen_timestamp": self.last_seen_timestamp,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "total_frames_tracked": self.total_frames_tracked,
            "last_known_bottom_center": list(self.last_known_bottom_center) if self.last_known_bottom_center else None,
        }
