"""Draw boxes and labels on images."""

from collections.abc import Sequence

import cv2
import numpy as np

from dronetrack.types import Detection

# One fixed BGR color per class, so a class keeps its color across frames.
PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
    (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
]


def draw_detections(image: np.ndarray, dets: Sequence[Detection], class_names: Sequence[str]) -> np.ndarray:
    """Return a copy of `image` with one box and a 'class score' label per detection."""
    out = image.copy()
    for d in dets:
        color = PALETTE[d.cls % len(PALETTE)]
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, color, 2)
        name = class_names[d.cls] if d.cls < len(class_names) else str(d.cls)
        cv2.putText(out, f"{name} {d.score:.2f}", (p1[0], max(p1[1] - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out
