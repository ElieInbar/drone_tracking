"""Link tracks to detections: each track gets at most one detection, the best overall assignment."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from dronetrack.geometry import iou_matrix

FORBIDDEN = 1e6  # cost of a pair that must never be chosen


def associate(
    track_boxes: np.ndarray,
    track_cls: np.ndarray,
    det_boxes: np.ndarray,
    det_cls: np.ndarray,
    min_iou: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian matching of tracks (N, 4) to detections (M, 4) on cost = 1 - IoU.

    Pairs of different classes, or overlapping less than `min_iou`, are forbidden.
    Returns (matched (track, detection) index pairs, unmatched track indices, unmatched detection indices).
    """
    n_tracks, n_dets = len(track_boxes), len(det_boxes)
    if n_tracks == 0 or n_dets == 0:
        return [], list(range(n_tracks)), list(range(n_dets))
    iou = iou_matrix(track_boxes, det_boxes)
    allowed = (iou >= min_iou) & (track_cls[:, None] == det_cls[None, :])
    cost = np.where(allowed, 1.0 - iou, FORBIDDEN)
    # Minimises the total cost over all pairs at once, unlike "each track takes its nearest box".
    rows, cols = linear_sum_assignment(cost)
    matches = [(int(r), int(c)) for r, c in zip(rows, cols) if allowed[r, c]]
    matched_tracks = {r for r, _ in matches}
    matched_dets = {c for _, c in matches}
    return (
        matches,
        [i for i in range(n_tracks) if i not in matched_tracks],
        [j for j in range(n_dets) if j not in matched_dets],
    )
