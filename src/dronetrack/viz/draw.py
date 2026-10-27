"""Draw boxes and labels on images."""

import colorsys
from collections.abc import Sequence

import cv2
import numpy as np

from dronetrack.types import Detection, Track

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


def id_color(track_id: int) -> tuple[int, int, int]:
    """A stable BGR color per identity: hues spread by the golden ratio so neighbours look different."""
    r, g, b = colorsys.hsv_to_rgb((track_id * 0.618034) % 1.0, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_tracks(image: np.ndarray, tracks: Sequence[Track], class_names: Sequence[str]) -> np.ndarray:
    """Return a copy of `image` with one box per track, colored by its number, labelled '#id class'."""
    out = image.copy()
    for t in tracks:
        color = id_color(t.track_id)
        p1, p2 = (int(t.x1), int(t.y1)), (int(t.x2), int(t.y2))
        cv2.rectangle(out, p1, p2, color, 2)
        name = class_names[t.cls] if 0 <= t.cls < len(class_names) else str(t.cls)
        cv2.putText(out, f"#{t.track_id} {name}", (p1[0], max(p1[1] - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out
