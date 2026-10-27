"""Remove duplicate detections of the same object, e.g. from overlapping tiles."""

from collections.abc import Sequence

import numpy as np

from dronetrack.geometry import boxes_to_array, ios_matrix, iou_matrix
from dronetrack.types import Detection

OVERLAP_METRICS = {"iou": iou_matrix, "ios": ios_matrix}


def nms(dets: Sequence[Detection], threshold: float = 0.5, metric: str = "ios", merge: bool = True) -> list[Detection]:
    """Greedy duplicate removal, class by class, most confident box first.

    The most confident remaining box takes every other box of its class that overlaps it by at least
    `threshold` (IoU or IoS). With merge=False they are simply dropped (classic NMS); with merge=True
    the kept box grows to cover them all (NMM, as in SAHI), so pieces of a cut object rebuild the whole object.
    """
    overlap = OVERLAP_METRICS[metric]
    kept = []
    for cls in sorted({d.cls for d in dets}):
        group = sorted((d for d in dets if d.cls == cls), key=lambda d: -d.score)
        boxes = boxes_to_array(group)
        ov = overlap(boxes, boxes)
        alive = np.ones(len(group), dtype=bool)
        for i, det in enumerate(group):
            if not alive[i]:
                continue
            taken = alive & (ov[i] >= threshold)
            taken[i] = True  # a box always takes itself, even a zero-area one (overlap 0 with itself)
            alive &= ~taken
            if merge:
                x1, y1 = boxes[taken, :2].min(axis=0)
                x2, y2 = boxes[taken, 2:].max(axis=0)
                det = Detection(float(x1), float(y1), float(x2), float(y2), det.score, cls)
            kept.append(det)
    return sorted(kept, key=lambda d: -d.score)
