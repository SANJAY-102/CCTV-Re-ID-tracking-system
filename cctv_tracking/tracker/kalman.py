"""
Noise Scale Adaptive (NSA) Kalman Filter with Camera Motion Transformation.
State vector: [cx, cy, a, h, vx, vy, va, vh].
Dynamically scales measurement noise covariance based on detection confidence,
and compensates for background camera movement via affine transformation matrices.
"""

from typing import Tuple
import numpy as np


class KalmanFilter:
    """
    8-dimensional NSA Kalman filter with camera ego-motion transformation.
    """

    def __init__(self, use_nsa: bool = True):
        self.use_nsa = use_nsa
        ndim, dt = 4, 1.0

        # State transition matrix (Constant Velocity model)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Measurement matrix (Observing cx, cy, a, h)
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and measurement standard deviation weights
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create track from unassociated measurement.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter prediction step.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        sqr = np.square(np.r_[std_pos, std_vel])
        motion_cov = np.diag(sqr)

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T
        )) + motion_cov

        return mean, covariance

    def project(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        confidence: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution to measurement space.
        If NSA is enabled, scales measurement covariance R dynamically:
        R_nsa = (1 - confidence) * R_base + eps
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))

        # NSA Kalman scaling: scale down noise for high confidence, scale up for low confidence
        if self.use_nsa:
            scale = max(0.05, 1.0 - float(confidence))
            innovation_cov = innovation_cov * (scale * 2.0)

        mean_proj = np.dot(self._update_mat, mean)
        cov_proj = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T
        )) + innovation_cov
        return mean_proj, cov_proj

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
        confidence: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter correction step with new observation.
        """
        projected_mean, projected_cov = self.project(mean, covariance, confidence=confidence)

        # Kalman gain computation
        kalman_gain = np.linalg.solve(
            projected_cov.T,
            np.dot(covariance, self._update_mat.T).T
        ).T
        
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T
        ))
        return new_mean, new_covariance

    def apply_camera_motion(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        H: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply 2x3 affine camera transformation matrix H = [R | t] to Kalman state.
        Compensates for background camera translation, rotation, and scale.
        """
        if H is None or H.shape != (2, 3):
            return mean, covariance

        R = H[:2, :2]
        t = H[:2, 2]

        # 1. Transform Position (cx, cy)
        pos = mean[:2]
        new_pos = np.dot(R, pos) + t
        mean[:2] = new_pos

        # 2. Transform Velocity (vx, vy)
        vel = mean[4:6]
        new_vel = np.dot(R, vel)
        mean[4:6] = new_vel

        # 3. Transform Covariance Matrix
        # Transform (cx, cy) sub-block and (vx, vy) sub-block
        try:
            covariance[:2, :2] = np.linalg.multi_dot((R, covariance[:2, :2], R.T))
            covariance[4:6, 4:6] = np.linalg.multi_dot((R, covariance[4:6, 4:6], R.T))
        except Exception:
            pass

        return mean, covariance
