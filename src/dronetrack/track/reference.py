"""Ultralytics' ByteTrack behind our tracker interface, as the reference for experiment 3."""

from collections.abc import Sequence

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from dronetrack.types import Detection, Track

NO_IMAGE_SIZE = (10_000, 10_000)  # Boxes wants an image size; ByteTrack never uses it for plain boxes


class UltralyticsByteTrack:
    """Same interface as ByteTracker: update(detections of one frame) -> tracked objects with numbers.

    Uses the default bytetrack.yaml settings, which our ByteTracker copies.
    """

    def __init__(self, config: str = "bytetrack.yaml"):
        self.tracker = BYTETracker(args=IterableSimpleNamespace(**YAML.load(check_yaml(config))))

    def update(self, dets: Sequence[Detection]) -> list[Track]:
        data = np.array([[d.x1, d.y1, d.x2, d.y2, d.score, d.cls] for d in dets], dtype=np.float32).reshape(-1, 6)
        out = self.tracker.update(Boxes(data, NO_IMAGE_SIZE))  # rows: x1, y1, x2, y2, id, score, cls, index
        return [
            Track(float(x1), float(y1), float(x2), float(y2), float(score), int(cls), track_id=int(tid))
            for x1, y1, x2, y2, tid, score, cls, _ in out
        ]
