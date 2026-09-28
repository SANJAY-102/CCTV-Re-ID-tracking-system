"""
MotionAwareByteTrack Primary Tracking Engine.
Performs Global Motion Compensation (GMC), NSA Kalman state forecasting,
Observation-Centric Momentum (OCM) multi-factor association, lost buffer management,
and motion state updates.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np

from .kalman import KalmanFilter
from .gmc import GlobalMotionCompensator
from .matching import (
    bbox_ious,
    iou_distance,
    center_distance,
    ocm_direction_distance,
    fuse_motion_cost_matrix,
    linear_assignment
)
from .track import Track, TrackState
from .motion import MotionState
from ..detector.yolo_detector import PersonDetection
from ..config import TrackerConfig


class MotionAwareByteTrack:
    """
    MotionAwareByteTrack CCTV multi-person tracking engine.
    Carries the heavy, high-frequency frame-to-frame tracking workload.
    """

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        self.kalman_filter = KalmanFilter(use_nsa=self.config.use_nsa)
        
        # Global Motion Compensator (GMC)
        self.gmc: Optional[GlobalMotionCompensator] = None
        if self.config.use_gmc:
            self.gmc = GlobalMotionCompensator(method=self.config.gmc_method)

        self.tracked_tracks: List[Track] = []     # Active confirmed tracks
        self.lost_tracks: List[Track] = []        # Tracks currently in lost buffer
        self.unconfirmed_tracks: List[Track] = [] # Tentative tracks awaiting confirmation
        
        self.frame_id: int = 0
        self.tracker_updates_count: int = 0
        # Kalman / IoU tracker occlusion matches (NOT Re-ID identity recoveries)
        self.tracker_occlusion_recoveries: int = 0
        self.tracker_recoveries_count: int = 0  # Backwards compatibility alias

    @property
    def all_active_tracks(self) -> List[Track]:
        """Return all tracks currently being tracked in the scene."""
        return [t for t in self.tracked_tracks if t.state == TrackState.TRACKED]

    @property
    def moving_tracks(self) -> List[Track]:
        return [t for t in self.all_active_tracks if t.motion_state == MotionState.MOVING]

    @property
    def stationary_tracks(self) -> List[Track]:
        return [t for t in self.all_active_tracks if t.motion_state == MotionState.STATIONARY]

    def update(
        self,
        detections: List[PersonDetection],
        frame_id: int,
        frame: Optional[np.ndarray] = None
    ) -> Tuple[List[Track], List[Track], List[Track], List[Track]]:
        """
        Main tracking step.
        
        Args:
            detections: Filtered person detections from YOLO.
            frame_id: Current frame sequence index.
            frame: Optional current video frame for Global Motion Compensation (GMC).
            
        Returns:
            Tuple of:
                - active_tracks: Currently confirmed & actively tracked persons on screen
                - lost_tracks: Confirmed persons temporarily missing detections
                - newly_confirmed_tracks: Tracks that transitioned from TENTATIVE to CONFIRMED
                - expired_tracks: Tracks that exceeded max_lost_frames (CONFIRMED COMPLETE FAILURE)
        """
        self.frame_id = frame_id
        self.tracker_updates_count += 1
        
        # 1. Global Motion Compensation (GMC) - Compensate background camera ego-motion
        if self.gmc is not None and frame is not None:
            active_boxes = [t.to_tlbr() for t in self.tracked_tracks]
            H = self.gmc.compute_camera_motion(frame, active_bboxes=active_boxes)
            
            for track in self.tracked_tracks:
                track.apply_camera_motion(H)
            for track in self.lost_tracks:
                track.apply_camera_motion(H)
            for track in self.unconfirmed_tracks:
                track.apply_camera_motion(H)

        # 2. Kalman prediction for all existing tracks
        for track in self.tracked_tracks:
            track.predict()
        for track in self.lost_tracks:
            track.predict()
        for track in self.unconfirmed_tracks:
            track.predict()

        # 3. Split detections by confidence (2-stage ByteTrack principle)
        high_dets: List[PersonDetection] = []
        low_dets: List[PersonDetection] = []
        
        for det in detections:
            if det.confidence >= self.config.high_det_thresh:
                high_dets.append(det)
            elif det.confidence >= self.config.low_det_thresh:
                low_dets.append(det)

        # Candidate tracks for first association: active confirmed + lost confirmed
        pool_tracks = self.tracked_tracks + self.lost_tracks
        
        # =========================================================================
        # STAGE 1: Match high-confidence detections with existing confirmed tracks
        # (Multi-factor fusion: IoU Distance + Center Distance + OCM Momentum Cost)
        # =========================================================================
        matched_tracks_1: List[Track] = []
        unmatched_tracks_1: List[Track] = []
        unmatched_dets_high: List[PersonDetection] = []
        
        if pool_tracks and high_dets:
            track_boxes = np.array([t.to_tlbr() for t in pool_tracks], dtype=np.float32)
            det_boxes = np.array([d.to_tlbr() for d in high_dets], dtype=np.float32)
            
            # Extract OCM momentum vectors
            track_momentums = np.array([t.get_ocm_momentum_vector()[:2] for t in pool_tracks], dtype=np.float32)
            
            cost_matrix = fuse_motion_cost_matrix(
                track_boxes=track_boxes,
                det_boxes=det_boxes,
                track_momentums=track_momentums,
                iou_weight=0.70,
                center_weight=0.15,
                ocm_weight=self.config.ocm_weight
            )
            
            matches, u_tracks, u_dets = linear_assignment(cost_matrix, threshold=self.config.match_thresh_first)
            
            for t_idx, d_idx in matches:
                track = pool_tracks[t_idx]
                det = high_dets[d_idx]
                was_lost = track.state == TrackState.LOST or track.lost_frames > 0
                track.update(det, frame_id)
                if was_lost:
                    self.tracker_recoveries_count += 1
                    self.tracker_occlusion_recoveries += 1
                matched_tracks_1.append(track)
                
            unmatched_tracks_1 = [pool_tracks[i] for i in u_tracks]
            unmatched_dets_high = [high_dets[i] for i in u_dets]
        else:
            unmatched_tracks_1 = pool_tracks
            unmatched_dets_high = high_dets

        # =========================================================================
        # STAGE 2: Match remaining unmatched tracks with low-confidence detections
        # (Crucial for occlusions, crossing, and motion blur recovery)
        # =========================================================================
        matched_tracks_2: List[Track] = []
        unmatched_tracks_2: List[Track] = []
        
        if unmatched_tracks_1 and low_dets:
            track_boxes = np.array([t.to_tlbr() for t in unmatched_tracks_1], dtype=np.float32)
            det_boxes = np.array([d.to_tlbr() for d in low_dets], dtype=np.float32)
            
            cost_matrix = iou_distance(track_boxes, det_boxes)
            matches, u_tracks, _ = linear_assignment(cost_matrix, threshold=self.config.match_thresh_second)
            
            for t_idx, d_idx in matches:
                track = unmatched_tracks_1[t_idx]
                det = low_dets[d_idx]
                was_lost = track.state == TrackState.LOST or track.lost_frames > 0
                track.update(det, frame_id)
                if was_lost:
                    self.tracker_recoveries_count += 1
                    self.tracker_occlusion_recoveries += 1
                matched_tracks_2.append(track)
                
            unmatched_tracks_2 = [unmatched_tracks_1[i] for i in u_tracks]
        else:
            unmatched_tracks_2 = unmatched_tracks_1

        # =========================================================================
        # STAGE 3: Match remaining unconfirmed (tentative) tracks with high-conf dets
        # =========================================================================
        newly_confirmed_tracks: List[Track] = []
        unmatched_dets_final: List[PersonDetection] = []
        
        matched_tentative: List[Track] = []
        if self.unconfirmed_tracks and unmatched_dets_high:
            unconf_boxes = np.array([t.to_tlbr() for t in self.unconfirmed_tracks], dtype=np.float32)
            det_boxes = np.array([d.to_tlbr() for d in unmatched_dets_high], dtype=np.float32)
            
            cost_matrix = iou_distance(unconf_boxes, det_boxes)
            matches, u_unconf, u_dets = linear_assignment(cost_matrix, threshold=self.config.match_thresh_unconfirmed)
            
            for t_idx, d_idx in matches:
                track = self.unconfirmed_tracks[t_idx]
                det = unmatched_dets_high[d_idx]
                track.update(det, frame_id)
                if track.hits >= self.config.min_hits_to_confirm:
                    track.state = TrackState.TRACKED
                    newly_confirmed_tracks.append(track)
                else:
                    matched_tentative.append(track)
                    
            unconfirmed_unmatched = [self.unconfirmed_tracks[i] for i in u_unconf]
            unmatched_dets_final = [unmatched_dets_high[i] for i in u_dets]
        else:
            unconfirmed_unmatched = self.unconfirmed_tracks
            unmatched_dets_final = unmatched_dets_high

        # Mark unmatched unconfirmed tracks as missed/deleted
        temp_unconf: List[Track] = list(matched_tentative)
        for track in unconfirmed_unmatched:
            track.mark_missed(max_lost_frames=2)  # Tentative tracks expire quickly
            if not track.is_deleted:
                temp_unconf.append(track)
        self.unconfirmed_tracks = temp_unconf

        # =========================================================================
        # STAGE 4: Initialize new tentative tracks from leftover high-conf dets
        # =========================================================================
        for det in unmatched_dets_final:
            if det.confidence >= self.config.new_track_thresh:
                # Strictly prevent spawning duplicate tracks on top of existing active/lost people
                is_duplicate_detection = False
                det_bbox = np.array([det.to_tlbr()], dtype=np.float32)
                for trk in self.tracked_tracks + self.lost_tracks:
                    trk_bbox = np.array([trk.to_tlbr()], dtype=np.float32)
                    if bbox_ious(det_bbox, trk_bbox)[0, 0] > 0.35:
                        is_duplicate_detection = True
                        break
                
                if not is_duplicate_detection:
                    new_track = Track(
                        detection=det,
                        frame_id=frame_id,
                        kalman_filter=self.kalman_filter,
                        max_trajectory_len=self.config.max_trajectory_len,
                        stationary_thresh=self.config.stationary_speed_thresh
                    )
                    self.unconfirmed_tracks.append(new_track)

        # =========================================================================
        # STAGE 5: Update state of all matched and lost tracks
        # =========================================================================
        all_matched = matched_tracks_1 + matched_tracks_2 + newly_confirmed_tracks
        
        # Tracks that were missed in this frame
        expired_tracks: List[Track] = []
        new_lost_tracks: List[Track] = []
        
        for track in unmatched_tracks_2:
            track.mark_missed(max_lost_frames=self.config.max_lost_frames)
            if track.is_deleted:
                expired_tracks.append(track)
            else:
                new_lost_tracks.append(track)

        # Segregate into current active vs lost pools
        active_candidates = [t for t in all_matched if t.state == TrackState.TRACKED]
        
        # Deduplicate active tracks (suppress double boxes overlapping on same seated person)
        # Suppressed duplicate tracks become LOST (preserving identity ownership in lost buffer)
        kept_active, suppressed_active = self._deduplicate_tracks(
            active_candidates,
            iou_thresh=self.config.duplicate_iou_thresh
        )
        self.tracked_tracks = kept_active
        self.lost_tracks = new_lost_tracks + suppressed_active

        return (
            self.tracked_tracks,
            self.lost_tracks,
            newly_confirmed_tracks,
            expired_tracks
        )

    def _deduplicate_tracks(self, tracks: List[Track], iou_thresh: float = 0.70) -> Tuple[List[Track], List[Track]]:
        """
        Suppress duplicate overlapping bounding boxes on the same person.
        Retains the track with the longest history (hits) and higher confidence.
        Suppressed tracks transition to LOST to protect identity ownership.
        Does NOT suppress two different people crossing in opposite directions.
        """
        if len(tracks) <= 1:
            return tracks, []

        # Sort by (hits, confidence) descending
        sorted_tracks = sorted(tracks, key=lambda t: (t.hits, t.confidence), reverse=True)
        kept_tracks: List[Track] = []
        suppressed_tracks: List[Track] = []

        for track in sorted_tracks:
            box_t = track.to_tlbr()
            is_dup = False
            for kept in kept_tracks:
                box_k = kept.to_tlbr()
                iou = bbox_ious(np.array([box_t]), np.array([box_k]))[0, 0]
                if iou > iou_thresh:
                    # If both are moving in significantly different directions (crossing), keep both!
                    if track.motion_state == MotionState.MOVING and kept.motion_state == MotionState.MOVING:
                        angle_diff = abs(track.direction_deg - kept.direction_deg)
                        if angle_diff > 180:
                            angle_diff = 360 - angle_diff
                        if angle_diff > 45.0:
                            continue
                    is_dup = True
                    break
            if not is_dup:
                kept_tracks.append(track)
            else:
                track.state = TrackState.LOST
                suppressed_tracks.append(track)

        return kept_tracks, suppressed_tracks


# Alias for backward compatibility
PrimaryTracker = MotionAwareByteTrack
