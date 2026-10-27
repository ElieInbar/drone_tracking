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
    det_weight: np.ndarray | None = None,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian matching of tracks (N, 4) to detections (M, 4) on cost = 1 - similarity.

    Similarity = IoU, multiplied by `det_weight` (the detection confidence, ByteTrack's "fuse score")
    when given: a shaky detection must then overlap the track much better to be accepted.
    Pairs of different classes, or with a similarity below `min_iou`, are forbidden.
    Returns (matched (track, detection) index pairs, unmatched track indices, unmatched detection indices).
    """
    n_tracks, n_dets = len(track_boxes), len(det_boxes)
    if n_tracks == 0 or n_dets == 0:
        return [], list(range(n_tracks)), list(range(n_dets))
    sim = iou_matrix(track_boxes, det_boxes)
    if det_weight is not None:
        sim = sim * det_weight[None, :]
    allowed = (sim >= min_iou) & (track_cls[:, None] == det_cls[None, :])
    cost = np.where(allowed, 1.0 - sim, FORBIDDEN)
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
