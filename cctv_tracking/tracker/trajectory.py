"""
Trajectory and Tracking Point Module.
Manages bottom-center tracking points and historical movement trails.
"""

from collections import deque
from typing import List, Tuple, Optional
import math
import numpy as np


class Trajectory:
    """
    Maintains a temporal sequence of bottom-center tracking points:
    P1 -> P2 -> P3 -> P4 -> P5 ...
    """

    def __init__(self, max_length: int = 45):
        self.max_length = max_length
        self._points: deque = deque(maxlen=max_length)  # List of (x, y)
        self._frames: deque = deque(maxlen=max_length)  # List of frame_ids
        self._total_distance: float = 0.0
        self._last_point: Optional[Tuple[float, float]] = None

    def add_point(self, point: Tuple[float, float], frame_id: int):
        """
        Add a new bottom-center coordinate.
        """
        x, y = float(point[0]), float(point[1])
        if self._last_point is not None:
            dx = x - self._last_point[0]
            dy = y - self._last_point[1]
            dist = math.sqrt(dx * dx + dy * dy)
            self._total_distance += dist

        self._points.append((x, y))
        self._frames.append(frame_id)
        self._last_point = (x, y)

    @property
    def points(self) -> List[Tuple[float, float]]:
        """Return list of points from oldest to newest."""
        return list(self._points)

    @property
    def latest_point(self) -> Optional[Tuple[float, float]]:
        if self._points:
            return self._points[-1]
        return None

    @property
    def total_distance(self) -> float:
        """Total cumulative path distance traveled in pixels."""
        return self._total_distance

    def get_recent_points(self, count: int) -> List[Tuple[float, float]]:
        """Return the N most recent points."""
        if not self._points:
            return []
        pts = list(self._points)
        return pts[-count:] if len(pts) > count else pts

    def clear(self):
        self._points.clear()
        self._frames.clear()
        self._total_distance = 0.0
        self._last_point = None
