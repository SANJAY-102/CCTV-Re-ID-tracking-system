"""
Motion Analysis & Observation-Centric Momentum (OCM) Module.
Calculates continuous velocity vectors, speed, 8-compass direction angles,
observation-centric momentum vectors, and classifies motion states.
"""

from enum import Enum
from typing import Tuple, List, Optional
import math
import numpy as np


class MotionState(str, Enum):
    STATIONARY = "STATIONARY"
    MOVING = "MOVING"
    LOST = "LOST"
    RECOVERED = "RECOVERED"


class Direction(str, Enum):
    NORTH = "N"
    NORTH_EAST = "NE"
    EAST = "E"
    SOUTH_EAST = "SE"
    SOUTH = "S"
    SOUTH_WEST = "SW"
    WEST = "W"
    NORTH_WEST = "NW"
    STATIONARY = "STATIONARY"


class MotionAnalyzer:
    """
    Computes velocity, heading angle, smoothed speed, OCM momentum vectors,
    and movement state classification.
    """

    @staticmethod
    def calculate_velocity(
        points: List[Tuple[float, float]],
        window_size: int = 5
    ) -> Tuple[float, float, float]:
        """
        Calculates (vx, vy, speed) in pixels/frame over a sliding window.
        """
        if len(points) < 2:
            return 0.0, 0.0, 0.0

        n = min(len(points), window_size)
        sub_pts = points[-n:]
        
        dx = sub_pts[-1][0] - sub_pts[0][0]
        dy = sub_pts[-1][1] - sub_pts[0][1]
        dt = max(1, len(sub_pts) - 1)

        vx = dx / dt
        vy = dy / dt
        speed = math.sqrt(vx * vx + vy * vy)
        return vx, vy, speed

    @staticmethod
    def calculate_direction(
        vx: float,
        vy: float,
        speed: float,
        stationary_thresh: float = 2.0
    ) -> Tuple[float, Direction]:
        """
        Computes the angle in degrees [0, 360) and 8-cardinal compass heading.
        """
        if speed < stationary_thresh:
            return 0.0, Direction.STATIONARY

        angle_rad = math.atan2(vy, vx)
        angle_deg = (math.degrees(angle_rad) + 360.0) % 360.0

        if angle_deg >= 337.5 or angle_deg < 22.5:
            direction = Direction.EAST
        elif 22.5 <= angle_deg < 67.5:
            direction = Direction.SOUTH_EAST
        elif 67.5 <= angle_deg < 112.5:
            direction = Direction.SOUTH
        elif 112.5 <= angle_deg < 157.5:
            direction = Direction.SOUTH_WEST
        elif 157.5 <= angle_deg < 202.5:
            direction = Direction.WEST
        elif 202.5 <= angle_deg < 247.5:
            direction = Direction.NORTH_WEST
        elif 247.5 <= angle_deg < 292.5:
            direction = Direction.NORTH
        else:
            direction = Direction.NORTH_EAST

        return angle_deg, direction

    @staticmethod
    def compute_ocm_momentum_vector(
        observations: List[Tuple[float, float]],
        delta_t: int = 3
    ) -> Tuple[float, float, float]:
        """
        Compute Observation-Centric Momentum (OCM) velocity vector from actual observations:
        v_ocm = (z_t - z_{t - delta_t}) / delta_t
        
        Returns:
            (vx_ocm, vy_ocm, speed_ocm)
        """
        if len(observations) < 2:
            return 0.0, 0.0, 0.0

        k = min(len(observations), delta_t + 1)
        sub = observations[-k:]
        dx = sub[-1][0] - sub[0][0]
        dy = sub[-1][1] - sub[0][1]
        dt = max(1, len(sub) - 1)

        vx = dx / dt
        vy = dy / dt
        speed = math.sqrt(vx * vx + vy * vy)
        return vx, vy, speed

    @staticmethod
    def classify_state(
        speed: float,
        is_lost: bool,
        is_freshly_recovered: bool = False,
        stationary_thresh: float = 2.5
    ) -> MotionState:
        """
        Classify the current physical motion state (STATIONARY, MOVING, LOST).
        Identity recovery status is governed orthogonally by IdentityManager.
        """
        if is_lost:
            return MotionState.LOST
        if speed >= stationary_thresh:
            return MotionState.MOVING
        return MotionState.STATIONARY
