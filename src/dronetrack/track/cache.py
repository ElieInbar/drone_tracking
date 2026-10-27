"""Run the detector once per video and keep its boxes on disk, so trackers can be compared on the same boxes."""

import json
from pathlib import Path

from dronetrack.io import iter_frames
from dronetrack.types import Detection


def save_detections(path: Path, dets_per_frame: list[list[Detection]]) -> None:
    rows = [[[d.x1, d.y1, d.x2, d.y2, d.score, d.cls] for d in frame] for frame in dets_per_frame]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rows))


def load_detections(path: Path) -> list[list[Detection]]:
    rows = json.loads(Path(path).read_text())
    return [[Detection(x1, y1, x2, y2, score, int(cls)) for x1, y1, x2, y2, score, cls in frame] for frame in rows]


def detect_sequence(detector, source: Path, cache: Path | None) -> list[list[Detection]]:
    """Detections of every frame of `source`, read from `cache` if it exists, otherwise computed and saved."""
    if cache is not None and Path(cache).exists():
        return load_detections(cache)
    dets = [detector.detect(frame) for frame in iter_frames(source)]
    if cache is not None:
        save_detections(cache, dets)
    return dets
