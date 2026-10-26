"""Box geometry shared by evaluation, tiling and tracking."""

from collections.abc import Sequence

import numpy as np

from dronetrack.types import Detection


def boxes_to_array(dets: Sequence[Detection]) -> np.ndarray:
    """Stack the boxes of `dets` into an (N, 4) float array of x1, y1, x2, y2."""
    return np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=float).reshape(-1, 4)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU (intersection over union) of every box in `a` (N, 4) with every box in `b` (M, 4) -> (N, M)."""
    a = a[:, None, :]  # (N, 1, 4): broadcasting compares each box of a with each box of b
    b = b[None, :, :]  # (1, M, 4)
    inter_w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    inter_h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = inter_w * inter_h
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
