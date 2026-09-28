"""
Global Motion Compensation (GMC) Module.
Estimates camera ego-motion (pan, tilt, zoom, wind vibration) using background feature
tracking with foreground person masking, producing an affine transformation matrix H.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np


class GlobalMotionCompensator:
    """
    Computes camera background shift between consecutive video frames
    to decouple camera ego-motion from human velocity.
    """

    def __init__(self, method: str = "orb", max_features: int = 500):
        self.method = method.lower()
        self.max_features = max_features
        self.prev_frame_gray: Optional[np.ndarray] = None
        self.prev_keypoints: Optional[np.ndarray] = None
        
        # Initialize ORB detector
        if self.method == "orb":
            self.detector = cv2.ORB_create(nfeatures=self.max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def reset(self):
        """Reset temporal state."""
        self.prev_frame_gray = None
        self.prev_keypoints = None

    def _create_foreground_mask(self, frame_shape: Tuple[int, int], bboxes: List[np.ndarray]) -> np.ndarray:
        """
        Create binary mask where foreground persons are masked out (0),
        ensuring keypoints are extracted strictly from the static background.
        """
        h, w = frame_shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255

        for bbox in bboxes:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            # Dilate bounding box slightly by 10% to remove person edges
            pad_w = int((x2 - x1) * 0.1)
            pad_h = int((y2 - y1) * 0.1)
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

        return mask

    def compute_camera_motion(
        self,
        frame: np.ndarray,
        active_bboxes: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """
        Estimate 2x3 affine transformation matrix H between previous and current frame.
        H maps coordinates in previous frame to current frame: x_curr = H * x_prev
        
        Returns:
            H: (2, 3) affine transformation matrix (Identity if no motion / first frame).
        """
        identity_h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        if frame is None or frame.size == 0:
            return identity_h

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        if self.prev_frame_gray is None:
            self.prev_frame_gray = curr_gray
            return identity_h

        # Downscale for high speed processing (e.g. max width 640)
        h, w = curr_gray.shape[:2]
        scale = 1.0
        if w > 640:
            scale = 640.0 / w
            proc_curr = cv2.resize(curr_gray, (640, int(h * scale)))
            proc_prev = cv2.resize(self.prev_frame_gray, (640, int(h * scale)))
        else:
            proc_curr = curr_gray
            proc_prev = self.prev_frame_gray

        # Scale bboxes to match processing resolution
        scaled_bboxes = []
        if active_bboxes:
            for b in active_bboxes:
                scaled_bboxes.append(b * scale)

        mask = self._create_foreground_mask(proc_curr.shape, scaled_bboxes)

        H = identity_h.copy()

        try:
            if self.method == "sparse_optflow":
                # Good features to track + Lucas-Kanade Optical Flow
                prev_pts = cv2.goodFeaturesToTrack(proc_prev, mask=mask, maxCorners=300, qualityLevel=0.01, minDistance=10)
                if prev_pts is not None and len(prev_pts) >= 6:
                    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(proc_prev, proc_curr, prev_pts, None)
                    valid_prev = prev_pts[status.flatten() == 1]
                    valid_curr = curr_pts[status.flatten() == 1]
                    if len(valid_prev) >= 6:
                        model, inliers = cv2.estimateAffinePartial2D(valid_prev, valid_curr, method=cv2.RANSAC)
                        if model is not None:
                            H = model
            else:
                # Default: ORB feature matching with RANSAC
                kp_prev, des_prev = self.detector.detectAndCompute(proc_prev, mask=mask)
                kp_curr, des_curr = self.detector.detectAndCompute(proc_curr, mask=mask)

                if des_prev is not None and des_curr is not None and len(kp_prev) >= 8 and len(kp_curr) >= 8:
                    matches = self.matcher.match(des_prev, des_curr)
                    # Filter top matches
                    matches = sorted(matches, key=lambda x: x.distance)[:100]
                    if len(matches) >= 8:
                        src_pts = np.float32([kp_prev[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp_curr[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                        model, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                        if model is not None:
                            H = model

            # Rescale translation components if frame was downscaled
            if scale != 1.0 and H is not None:
                H[0, 2] /= scale
                H[1, 2] /= scale

        except Exception:
            H = identity_h.copy()

        self.prev_frame_gray = curr_gray
        return H.astype(np.float32) if H is not None else identity_h
