"""
Advanced Multi-Factor Association & Matching with Observation-Centric Momentum (OCM).
Provides IoU computation, center distance penalty, OCM directional momentum gating,
and Hungarian bipartite linear sum assignment.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment


def bbox_ious(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise IoU between two sets of bounding boxes [x1, y1, x2, y2].
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.ascontiguousarray(boxes_a, dtype=np.float32)
    boxes_b = np.ascontiguousarray(boxes_b, dtype=np.float32)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])

    wh = np.clip(rb - lt, a_min=0, a_max=None)
    intersection = wh[:, :, 0] * wh[:, :, 1]

    union = area_a[:, None] + area_b[None, :] - intersection
    union = np.maximum(union, 1e-6)

    return intersection / union


def iou_distance(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Computes IoU distance matrix: 1.0 - IoU."""
    return 1.0 - bbox_ious(boxes_a, boxes_b)


def center_distance(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Computes Euclidean center distance normalized by the average diagonal of boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    c_a = np.column_stack([(boxes_a[:, 0] + boxes_a[:, 2]) / 2.0, (boxes_a[:, 1] + boxes_a[:, 3]) / 2.0])
    c_b = np.column_stack([(boxes_b[:, 0] + boxes_b[:, 2]) / 2.0, (boxes_b[:, 1] + boxes_b[:, 3]) / 2.0])

    diff = c_a[:, None, :] - c_b[None, :, :]
    dist = np.sqrt(np.sum(np.square(diff), axis=2))

    scale_a = np.sqrt((boxes_a[:, 2] - boxes_a[:, 0])**2 + (boxes_a[:, 3] - boxes_a[:, 1])**2)
    scale_b = np.sqrt((boxes_b[:, 2] - boxes_b[:, 0])**2 + (boxes_b[:, 3] - boxes_b[:, 1])**2)
    scale = (scale_a[:, None] + scale_b[None, :]) / 2.0
    scale = np.maximum(scale, 1e-5)

    return dist / scale


def ocm_direction_distance(
    track_momentum_vectors: np.ndarray,  # (N, 2) -> (vx, vy)
    track_centers: np.ndarray,           # (N, 2) -> (cx, cy)
    det_centers: np.ndarray,             # (M, 2) -> (cx, cy)
    stationary_thresh: float = 2.0
) -> np.ndarray:
    """
    Computes Observation-Centric Momentum (OCM) directional deviation matrix.
    Penalizes candidates that violate existing momentum direction during turns/stops.
    
    Returns:
        cost_matrix: (N, M) matrix with values in [0.0, 1.0].
    """
    N = len(track_momentum_vectors)
    M = len(det_centers)
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)

    # Displacements from track center to candidate detection centers
    disp = det_centers[None, :, :] - track_centers[:, None, :]  # [N, M, 2]
    disp_norm = np.sqrt(np.sum(np.square(disp), axis=2))        # [N, M]
    disp_norm = np.maximum(disp_norm, 1e-5)
    disp_unit = disp / disp_norm[:, :, None]                    # [N, M, 2]

    # Track momentum speed
    speeds = np.sqrt(np.sum(np.square(track_momentum_vectors), axis=1))  # [N,]
    speeds_norm = np.maximum(speeds, 1e-5)
    mom_unit = track_momentum_vectors / speeds_norm[:, None]             # [N, 2]

    # Dot product between unit momentum vector and unit displacement vector
    # cos(theta) in [-1.0, 1.0]
    cos_theta = np.sum(mom_unit[:, None, :] * disp_unit, axis=2)        # [N, M]

    # Map cos(theta) to cost in [0.0, 1.0]: 0.0 for identical heading, 1.0 for reverse heading
    dir_cost = (1.0 - cos_theta) / 2.0

    # If track is stationary or candidate displacement is tiny, neutral cost (0.0)
    is_stat = (speeds < stationary_thresh)[:, None]
    is_close = (disp_norm < 5.0)
    dir_cost[is_stat | is_close] = 0.0

    return dir_cost.astype(np.float32)


def fuse_motion_cost_matrix(
    track_boxes: np.ndarray,
    det_boxes: np.ndarray,
    track_momentums: Optional[np.ndarray] = None,
    iou_weight: float = 0.85,
    center_weight: float = 0.15,
    ocm_weight: float = 0.0,
    scale_weight: float = 0.0
) -> np.ndarray:
    """
    Robust ByteTrack cost fusion: IoU distance + normalized center distance.
    Weights are strictly normalized so maximum distance produces cost = 1.0.
    """
    if len(track_boxes) == 0 or len(det_boxes) == 0:
        return np.zeros((len(track_boxes), len(det_boxes)), dtype=np.float32)

    iou_dist = iou_distance(track_boxes, det_boxes)
    c_dist = center_distance(track_boxes, det_boxes)
    c_dist = np.clip(c_dist, 0.0, 1.0)

    w_total = iou_weight + center_weight
    if w_total <= 0:
        w_total = 1.0
    cost = (iou_weight / w_total) * iou_dist + (center_weight / w_total) * c_dist
    return np.clip(cost, 0.0, 1.0).astype(np.float32)


def linear_assignment(cost_matrix: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Global optimal bipartite matching via Hungarian algorithm (linear sum assignment).
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0], dtype=int),
            np.arange(cost_matrix.shape[1], dtype=int)
        )

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_a = list(range(cost_matrix.shape[0]))
    unmatched_b = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_indices, col_indices):
        if cost_matrix[r, c] <= threshold:
            matches.append([r, c])
            if r in unmatched_a:
                unmatched_a.remove(r)
            if c in unmatched_b:
                unmatched_b.remove(c)

    matches = np.array(matches, dtype=int) if matches else np.empty((0, 2), dtype=int)
    return matches, np.array(unmatched_a, dtype=int), np.array(unmatched_b, dtype=int)
