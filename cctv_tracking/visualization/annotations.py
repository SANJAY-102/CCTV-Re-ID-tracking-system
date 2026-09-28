"""
CCTV Visual Annotations and Real-Time Telemetry HUD.
Renders stylized bounding boxes, bottom-center tracking points (●), trajectory trails,
and performance metrics comparing Primary Tracker Updates vs Rare Re-ID Invocations.
"""

from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np

from ..tracker.track import Track, TrackState
from ..tracker.motion import MotionState
from ..config import VisualizationConfig


class Annotator:
    """
    Renders high-contrast CCTV surveillance annotations and live HUD overlay.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()

    def get_state_color(self, track: Track, current_frame: Optional[int] = None) -> Tuple[int, int, int]:
        """Return BGR color corresponding to track motion and lifecycle state."""
        cur_f = current_frame if current_frame is not None else getattr(track, "frame_id", 0)
        event = getattr(track, "identity_event", None)
        event_frame = getattr(track, "identity_event_frame", -1)
        event_window = getattr(self.config, "event_display_frames", 45)
        is_event_active = (cur_f - event_frame) <= event_window if (event and event_frame >= 0) else False

        if track.emp_id == "UNKNOWN" or getattr(track, "identity_status", "") == "UNKNOWN":
            return getattr(self.config, "color_unknown", (0, 140, 255))
        if track.state == TrackState.LOST:
            return self.config.color_lost
        if event == "RECOVERED" and is_event_active:
            return self.config.color_recovered
        if track.motion_state == MotionState.MOVING:
            return self.config.color_moving
        return self.config.color_stationary

    def draw_tracking_point(self, frame: np.ndarray, point: Tuple[float, float], color: Tuple[int, int, int]):
        """
        Draw bottom-center tracking point (●) with high-contrast inner dot and glowing outer ring.
        """
        px, py = int(round(point[0])), int(round(point[1]))
        h, w = frame.shape[:2]
        if 0 <= px < w and 0 <= py < h:
            # Outer ring
            cv2.circle(frame, (px, py), 6, color, 1, lineType=cv2.LINE_AA)
            # Inner solid circle ●
            cv2.circle(frame, (px, py), 4, color, -1, lineType=cv2.LINE_AA)
            # Core white highlight
            cv2.circle(frame, (px, py), 1, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    def draw_trajectory_trail(self, frame: np.ndarray, points: List[Tuple[float, float]], color: Tuple[int, int, int], is_moving: bool = True):
        """
        Draw smooth trajectory path: P1 -> P2 -> P3 -> ... with temporal fading.
        Only draws if person is actively moving to keep seated people clean.
        """
        if len(points) < 2 or not is_moving:
            return

        num_points = len(points)
        for i in range(1, num_points):
            pt1 = (int(round(points[i - 1][0])), int(round(points[i - 1][1])))
            pt2 = (int(round(points[i][0])), int(round(points[i][1])))
            
            # Skip unrealistic large jumps (e.g. > 150px in single step)
            dist_step = (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2
            if dist_step > 22500:
                continue

            # Fade thickness and intensity towards the past
            progress = i / float(num_points)
            thickness = max(1, int(round(1 + 2 * progress)))
            cv2.line(frame, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
            
            # Draw tiny trail point on intermediate steps
            if i % 3 == 0:
                cv2.circle(frame, pt2, 2, color, -1, lineType=cv2.LINE_AA)

    def draw_track(self, frame: np.ndarray, track: Track, current_frame: Optional[int] = None):
        """
        Draw individual person bounding box, labels, tracking point, and trajectory.
        """
        cur_f = current_frame if current_frame is not None else getattr(track, "frame_id", 0)
        color = self.get_state_color(track, current_frame=cur_f)
        x1, y1, x2, y2 = [int(round(v)) for v in track.to_tlbr()]
        h_img, w_img = frame.shape[:2]
        
        # Clamp to frame boundary
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)

        # 1. Draw Trajectory Trail (only if moving)
        if self.config.draw_trajectory:
            self.draw_trajectory_trail(frame, track.trajectory.points, color, is_moving=(track.motion_state == MotionState.MOVING))

        # 2. Draw Bottom-Center Tracking Point ●
        if self.config.draw_tracking_point:
            self.draw_tracking_point(frame, track.bottom_center, color)

        # 3. Draw Bounding Box with Corner Accents
        if self.config.draw_box:
            # Main rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
            
            # Corner brackets for CCTV aesthetic
            corner_len = min(16, max(6, int((x2 - x1) * 0.2)))
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), 2)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), 2)
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), (255, 255, 255), 2)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), (255, 255, 255), 2)
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), (255, 255, 255), 2)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), (255, 255, 255), 2)
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), 2)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), 2)

        # 4. Header Label: EMP-ID | T-ID
        emp_str = track.emp_id or "UNASSIGNED"
        header_text = f"{emp_str} | {track.tid_str}"
        
        # Subtitle Label: One-time Identity Event (Section 3, 4, 16)
        motion_str = f"MOVING {track.speed:.1f}px/f {track.direction.value}" if track.motion_state == MotionState.MOVING else "STATIONARY"
        
        event = getattr(track, "identity_event", None)
        event_frame = getattr(track, "identity_event_frame", -1)
        event_window = getattr(self.config, "event_display_frames", 45)
        is_event_active = (cur_f - event_frame) <= event_window if (event and event_frame >= 0) else False

        if event == "RECOVERED" and is_event_active:
            conf = getattr(track, "identity_confidence", 1.0)
            motion_text = f"RECOVERED ({conf:.2f}) | {motion_str}"
        elif event == "NEW" and is_event_active:
            motion_text = f"NEW | {motion_str}"
        elif emp_str == "UNKNOWN" or getattr(track, "identity_status", "") == "UNKNOWN":
            motion_text = f"UNKNOWN | {motion_str}"
        else:
            motion_text = f"TRACKED | {motion_str}"

        # Render Label Badges
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale_hdr = 0.48
        scale_sub = 0.38
        thick = 1

        (w_hdr, h_hdr), _ = cv2.getTextSize(header_text, font, scale_hdr, thick)
        (w_sub, h_sub), _ = cv2.getTextSize(motion_text, font, scale_sub, thick)
        badge_w = max(w_hdr, w_sub) + 12
        badge_h = h_hdr + h_sub + 12

        # Draw badge background
        badge_y1 = max(0, y1 - badge_h - 4)
        badge_y2 = badge_y1 + badge_h
        badge_x2 = min(w_img - 1, x1 + badge_w)

        # Translucent badge background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, badge_y1), (badge_x2, badge_y2), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, 1)

        # Draw text
        cv2.putText(frame, header_text, (x1 + 6, badge_y1 + h_hdr + 3), font, scale_hdr, (255, 255, 255), thick, cv2.LINE_AA)
        cv2.putText(frame, motion_text, (x1 + 6, badge_y1 + h_hdr + h_sub + 8), font, scale_sub, color, thick, cv2.LINE_AA)

    def draw_hud(self, frame: np.ndarray, stats: Dict[str, Any]):
        """
        Draw comprehensive performance HUD overlay.
        Specifically showcases Tracker Updates >> Re-ID Calls and separates:
        1. Tracker Occlusion Recoveries (Kalman/IoU)
        2. Re-ID Inference Calls (Forward passes)
        3. Recovery Attempts (Sessions initiated)
        4. Identity Recoveries (Genuine restored dormant EMPs)
        """
        if not self.config.draw_hud:
            return

        h, w = frame.shape[:2]
        hud_w = 340
        hud_h = 260
        
        # Semi-transparent HUD box
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), self.config.color_hud_bg, -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (10, 10), (10 + hud_w, 10 + hud_h), (80, 80, 90), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # HUD Header
        cv2.putText(frame, "CCTV TRACKING & RE-ID SYSTEM", (20, 32), font, 0.50, (0, 235, 120), 1, cv2.LINE_AA)
        cv2.line(frame, (20, 38), (10 + hud_w - 10, 38), (80, 80, 90), 1)

        lines = [
            (f"FPS: {stats.get('fps', 0.0):.1f} | Frame: {stats.get('frame_id', 0)}", (220, 220, 220)),
            (f"Active Tracks: {stats.get('active_tracks', 0)} (Moving: {stats.get('moving_tracks', 0)})", (0, 235, 120)),
            (f"Stationary: {stats.get('stationary_tracks', 0)} | Lost: {stats.get('lost_tracks', 0)}", (0, 215, 255)),
            (f"Tracker Associations: {stats.get('tracker_updates', 0)}", (0, 255, 128)),
            (f"Tracker Occlusions: {stats.get('tracker_recoveries', 0)}", (0, 235, 120)),
            (f"Re-ID Inferences: {stats.get('reid_inference_calls', stats.get('reid_calls', 0))}", (0, 140, 255) if (stats.get('reid_inference_calls', 0) or stats.get('reid_calls', 0)) > 0 else (180, 180, 180)),
            (f"Recovery Attempts: {stats.get('recovery_attempts', 0)}", (255, 180, 50)),
            (f"Identity Recoveries: {stats.get('successful_identity_recoveries', stats.get('reid_recoveries', 0))}", (255, 50, 200)),
            (f"Total EMP Enrolled: {stats.get('total_employees', 0)}", (255, 255, 255)),
        ]

        y = 56
        for text, col in lines:
            cv2.putText(frame, text, (22, y), font, 0.38, col, 1, cv2.LINE_AA)
            y += 21

    def render(
        self,
        frame: np.ndarray,
        active_tracks: List[Track],
        lost_tracks: List[Track],
        stats: Dict[str, Any]
    ) -> np.ndarray:
        """
        Composite full frame with annotations.
        Renders confirmed active tracks ONLY (no ghost lost boxes on screen).
        """
        annotated = frame.copy()
        current_frame = stats.get("frame_id", 0)

        # Render confirmed active tracks ONLY
        for track in active_tracks:
            self.draw_track(annotated, track, current_frame=current_frame)

        # Render HUD
        self.draw_hud(annotated, stats)

        return annotated
