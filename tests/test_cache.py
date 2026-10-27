import numpy as np

from dronetrack.track.cache import detect_sequence, load_detections, save_detections
from dronetrack.types import Detection


def test_detections_round_trip(tmp_path):
    frames = [[Detection(1, 2, 3, 4, 0.5, 3)], []]
    save_detections(tmp_path / "d.json", frames)
    assert load_detections(tmp_path / "d.json") == frames


class CountingDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return [Detection(0, 0, 1, 1, 0.9, 3)]


def test_detect_sequence_uses_the_cache_the_second_time(tmp_path, monkeypatch):
    frames = [np.zeros((4, 4, 3), np.uint8)] * 3
    monkeypatch.setattr("dronetrack.track.cache.iter_frames", lambda source: iter(frames))
    det = CountingDetector()
    first = detect_sequence(det, tmp_path / "seq", tmp_path / "cache.json")
    second = detect_sequence(det, tmp_path / "seq", tmp_path / "cache.json")
    assert det.calls == 3 and first == second and len(first) == 3
