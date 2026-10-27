import cv2
import imageio.v2 as imageio
import numpy as np
import pytest

from dronetrack.demo import resize_for_demo, track_video
from dronetrack.track.byte_tracker import ByteTracker
from dronetrack.types import Detection


class FakeDetector:
    """One car moving 4 px right per frame (no model to load)."""

    def __init__(self):
        self.n = 0

    def detect(self, image):
        self.n += 1
        return [Detection(10 + 4 * self.n, 20, 40 + 4 * self.n, 40, 0.9, 3)]


class EmptyDetector:
    def detect(self, image):
        return []


def write_video(path, n, w=320, h=240):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for _ in range(n):
        writer.write(np.zeros((h, w, 3), np.uint8))
    writer.release()
    return path


def test_tracks_one_object_and_writes_browser_mp4(tmp_path):
    src, out = write_video(tmp_path / "in.mp4", 20), tmp_path / "out.mp4"
    r = track_video(src, out, FakeDetector(), ByteTracker())
    assert r.frames == 20 and not r.truncated and r.objects_per_class == {"car": 1}
    reader = imageio.get_reader(out)
    assert reader.get_meta_data()["codec"] == "h264" and reader.count_frames() == 20


def test_stops_at_max_frames(tmp_path):
    src = write_video(tmp_path / "in.mp4", 30)
    r = track_video(src, tmp_path / "o.mp4", FakeDetector(), ByteTracker(), max_frames=10)
    assert r.frames == 10 and r.truncated and "first 10 only" in r.summary()


def test_whole_video_is_not_marked_truncated(tmp_path):
    src = write_video(tmp_path / "in.mp4", 10)
    r = track_video(src, tmp_path / "o.mp4", FakeDetector(), ByteTracker(), max_frames=10)
    assert r.frames == 10 and not r.truncated and "only" not in r.summary()


def test_resize_for_demo_caps_width_and_makes_sizes_even():
    assert resize_for_demo(np.zeros((1080, 1920, 3), np.uint8), 1280).shape == (720, 1280, 3)
    assert resize_for_demo(np.zeros((241, 321, 3), np.uint8), 1280).shape == (240, 320, 3)


def test_no_object_still_writes_video(tmp_path):
    r = track_video(write_video(tmp_path / "in.mp4", 5), tmp_path / "o.mp4", EmptyDetector(), ByteTracker())
    assert r.objects_per_class == {} and "0 objects tracked" in r.summary() and (tmp_path / "o.mp4").exists()


def test_summary_lists_classes_most_frequent_first(tmp_path):
    r = track_video(write_video(tmp_path / "in.mp4", 3), tmp_path / "o.mp4", FakeDetector(), ByteTracker())
    r.objects_per_class = {"car": 2, "pedestrian": 5}
    assert "7 objects tracked: pedestrian 5, car 2" in r.summary()


def test_unreadable_file_raises_clear_error(tmp_path):
    bad = tmp_path / "notes.mp4"
    bad.write_text("not a video")
    with pytest.raises(ValueError, match="Could not read any frame"):
        track_video(bad, tmp_path / "o.mp4", FakeDetector(), ByteTracker())
