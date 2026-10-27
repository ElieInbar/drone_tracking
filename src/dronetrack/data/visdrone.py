"""VisDrone dataset: class names and annotation reading."""

from pathlib import Path

from collections import defaultdict

from dronetrack.types import Detection, Track

# YOLO class index i = VisDrone category i + 1 (VisDrone 0 = ignored regions, 11 = others: both excluded)
CLASS_NAMES: tuple[str, ...] = (
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
)
# Classes scored by the VisDrone-MOT protocol: pedestrian, car, van, truck, bus
EVAL_CLASSES: tuple[int, ...] = (0, 3, 4, 5, 8)
IGNORED_CLASS = -1  # class given to "ignored region" boxes (crowds or blur nobody annotated object by object)


def parse_det_annotation(path: Path, img_w: int, img_h: int) -> list[Detection]:
    """Read one VisDrone-DET annotation file into ground-truth Detections.

    Each line is: left, top, width, height, score, category, truncation, occlusion.
    In ground truth, score 0 marks a box to ignore. Boxes are clipped to the image and
    dropped if nothing is left.
    """
    dets = []
    for line in Path(path).read_text().splitlines():
        fields = [f for f in line.strip().split(",") if f != ""]
        if len(fields) < 6:
            continue
        left, top, w, h, score, category = (int(f) for f in fields[:6])
        if score == 0 or not 1 <= category <= 10:
            continue
        x1, y1 = max(left, 0), max(top, 0)
        x2, y2 = min(left + w, img_w), min(top + h, img_h)
        if x2 <= x1 or y2 <= y1:
            continue
        dets.append(Detection(x1, y1, x2, y2, 1.0, category - 1))
    return dets


def parse_mot_annotation(path: Path) -> tuple[dict[int, list[Track]], dict[int, list[Detection]]]:
    """Read one VisDrone-MOT annotation file: ground-truth tracks and ignored regions, keyed by frame number.

    Each line is: frame, id, left, top, width, height, score, category, truncation, occlusion.
    Category 0 marks an ignored region, 11 ("others") is dropped, score 0 marks a box to ignore.
    """
    tracks: dict[int, list[Track]] = defaultdict(list)
    ignored: dict[int, list[Detection]] = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        fields = [f for f in line.strip().split(",") if f != ""]
        if len(fields) < 8:
            continue
        frame, track_id, left, top, w, h, score, category = (int(f) for f in fields[:8])
        if w <= 0 or h <= 0:
            continue
        if category == 0:
            ignored[frame].append(Detection(left, top, left + w, top + h, 1.0, IGNORED_CLASS))
        elif score == 1 and 1 <= category <= 10:
            tracks[frame].append(Track(left, top, left + w, top + h, 1.0, category - 1, track_id=track_id))
    return dict(tracks), dict(ignored)
