"""Multi-object tracker in the style of ByteTrack: Kalman prediction + Hungarian matching in two passes."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from dronetrack.geometry import boxes_to_array
from dronetrack.track.kalman import KalmanBoxFilter
from dronetrack.track.matching import associate
from dronetrack.types import Detection, Track

TENTATIVE, ACTIVE, LOST = "tentative", "active", "lost"


@dataclass
class _Track:
    """Internal state of one object being followed."""

    kf: KalmanBoxFilter
    cls: int
    score: float
    state: str
    track_id: int | None = None  # given only once the track is confirmed
    frames_unseen: int = 0
    seen_now: bool = field(default=False)

    def see(self, det: Detection) -> None:
        self.kf.update(np.array([det.x1, det.y1, det.x2, det.y2], float))
        self.score = det.score
        self.frames_unseen = 0
        self.seen_now = True


def _boxes_and_classes(items) -> tuple[np.ndarray, np.ndarray]:
    """Boxes (N, 4) and classes (N,) of a list of _Track (predicted boxes) or Detection."""
    if items and isinstance(items[0], _Track):
        boxes = np.array([t.kf.box for t in items], float).reshape(-1, 4)
    else:
        boxes = boxes_to_array(items)
    return boxes, np.array([x.cls for x in items])


class ByteTracker:
    """Give each object a number it keeps from frame to frame. Call update() once per frame.

    Defaults match Ultralytics' bytetrack.yaml (including fuse_score), so the comparison is between
    algorithms, not settings. Like Ultralytics, the second pass (unsure detections) uses plain IoU.
    """

    def __init__(
        self,
        high: float = 0.25,
        low: float = 0.1,
        new_track: float = 0.25,
        max_lost: int = 30,
        first_min_iou: float = 0.2,
        second_min_iou: float = 0.5,
        tentative_min_iou: float = 0.3,
        fuse_score: bool = True,
    ):
        self.high, self.low, self.new_track = high, low, new_track
        self.max_lost = max_lost
        self.first_min_iou, self.second_min_iou = first_min_iou, second_min_iou
        self.tentative_min_iou = tentative_min_iou
        self.fuse_score = fuse_score  # weigh overlaps by detection confidence (experiment 3, docs/DECISIONS.md D8)
        self.tracks: list[_Track] = []
        self.frame = 0
        self.next_id = 1

    def _match(self, tracks: list[_Track], dets: list[Detection], min_iou: float, fuse: bool = False):
        """Feed matched detections to their tracks; return the tracks and detections left alone."""
        t_boxes, t_cls = _boxes_and_classes(tracks)
        d_boxes, d_cls = _boxes_and_classes(dets)
        weight = np.array([d.score for d in dets]) if fuse and self.fuse_score else None
        matches, free_tracks, free_dets = associate(t_boxes, t_cls, d_boxes, d_cls, min_iou, weight)
        for i, j in matches:
            tracks[i].see(dets[j])
        return [tracks[i] for i in free_tracks], [dets[j] for j in free_dets]

    def _confirm(self, track: _Track) -> None:
        track.state = ACTIVE
        if track.track_id is None:
            track.track_id = self.next_id
            self.next_id += 1

    def update(self, dets: Sequence[Detection]) -> list[Track]:
        self.frame += 1
        for t in self.tracks:
            t.kf.predict()  # 1. where should each object be now?
            t.seen_now = False
        strong = [d for d in dets if d.score >= self.high]
        weak = [d for d in dets if self.low <= d.score < self.high]
        confirmed = [t for t in self.tracks if t.state in (ACTIVE, LOST)]
        tentative = [t for t in self.tracks if t.state == TENTATIVE]

        # 2. First pass: confident detections against every confirmed track, even recently lost ones.
        left_tracks, strong = self._match(confirmed, strong, self.first_min_iou, fuse=True)
        # 3. Second pass (ByteTrack's idea): unsure detections only extend tracks that were active,
        #    e.g. a car half hidden by a tree. They never resurrect lost tracks nor start new ones.
        still_active = [t for t in left_tracks if t.state == ACTIVE]
        self._match(still_active, weak, self.second_min_iou)
        # 4. New objects seen last frame get confirmed if seen again now.
        _, strong = self._match(tentative, strong, self.tentative_min_iou, fuse=True)

        for t in self.tracks:
            if t.seen_now:
                self._confirm(t)
            else:
                t.frames_unseen += 1
                if t.state == ACTIVE:
                    t.state = LOST
        # A tentative track not seen again was a one-frame flash; a lost track is forgotten after max_lost frames.
        self.tracks = [
            t for t in self.tracks if not (t.state == TENTATIVE and not t.seen_now) and t.frames_unseen <= self.max_lost
        ]

        # 5. Confident detections left alone start new tracks (confirmed at once on the very first frame).
        for d in strong:
            if d.score >= self.new_track:
                track = _Track(KalmanBoxFilter(np.array([d.x1, d.y1, d.x2, d.y2], float)), d.cls, d.score, TENTATIVE)
                track.seen_now = True
                if self.frame == 1:
                    self._confirm(track)
                self.tracks.append(track)

        return [
            Track(*(float(v) for v in t.kf.box), t.score, t.cls, track_id=t.track_id)
            for t in self.tracks
            if t.state == ACTIVE and t.seen_now
        ]
