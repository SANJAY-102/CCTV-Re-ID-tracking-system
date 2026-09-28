"""
Active Employee Registry.

Manages the live, 1-to-1 bidirectional mapping between EMP-IDs and Track IDs
for all currently visible (ACTIVE) persons.

Inspired by:
- Norfair: tracked_objects list as authoritative live state
- BoxMOT StrongSORT: per-frame active track set with strict identity ownership

Rules:
- Each EMP-ID maps to exactly ONE Track ID in any given frame.
- Each Track ID maps to exactly ONE EMP-ID.
- Only ACTIVE (visible or LOST buffer) employees appear here.
- DORMANT employees are NEVER in this registry.
"""

from typing import Dict, Optional, Set


class ActiveEmployeeRegistry:
    """
    Bidirectional map: emp_id <-> track_id for currently active employees.
    Expiry of a track removes from this registry, but does NOT delete the EMP.
    """

    def __init__(self):
        # emp_id -> track_id
        self._emp_to_track: Dict[str, int] = {}
        # track_id -> emp_id
        self._track_to_emp: Dict[int, str] = {}

    def register(self, emp_id: str, track_id: int) -> None:
        """Bind an EMP-ID to a Track ID. Enforces strict 1-to-1."""
        self._emp_to_track[emp_id] = track_id
        self._track_to_emp[track_id] = emp_id

    def release_track(self, track_id: int) -> Optional[str]:
        """
        Releases a Track ID from active registry when track expires or goes dormant.
        Returns the EMP-ID that was released, or None.
        """
        emp_id = self._track_to_emp.pop(track_id, None)
        if emp_id is not None:
            self._emp_to_track.pop(emp_id, None)
        return emp_id

    def release_emp(self, emp_id: str) -> Optional[int]:
        """Releases an EMP from active registry. Returns the Track ID or None."""
        track_id = self._emp_to_track.pop(emp_id, None)
        if track_id is not None:
            self._track_to_emp.pop(track_id, None)
        return track_id

    def get_emp_for_track(self, track_id: int) -> Optional[str]:
        return self._track_to_emp.get(track_id)

    def get_track_for_emp(self, emp_id: str) -> Optional[int]:
        return self._emp_to_track.get(emp_id)

    def is_emp_active(self, emp_id: str) -> bool:
        return emp_id in self._emp_to_track

    def is_track_assigned(self, track_id: int) -> bool:
        return track_id in self._track_to_emp

    @property
    def active_emp_ids(self) -> Set[str]:
        return set(self._emp_to_track.keys())

    @property
    def active_track_ids(self) -> Set[int]:
        return set(self._track_to_emp.keys())

    def get_all_bindings(self) -> Dict[str, int]:
        return dict(self._emp_to_track)

    def clear(self) -> None:
        self._emp_to_track.clear()
        self._track_to_emp.clear()

    def __len__(self) -> int:
        return len(self._emp_to_track)

    def __repr__(self) -> str:
        return f"ActiveEmployeeRegistry({dict(self._emp_to_track)})"
