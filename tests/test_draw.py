import numpy as np

from dronetrack.data.visdrone import CLASS_NAMES
from dronetrack.types import Detection, Track
from dronetrack.viz.draw import draw_detections, draw_tracks


def test_draw_returns_copy_and_marks_pixels():
    img = np.zeros((50, 50, 3), np.uint8)
    out = draw_detections(img, [Detection(10, 10, 30, 30, 0.9, 0)], ["pedestrian"])
    assert img.sum() == 0 and out.sum() > 0


def test_draw_with_no_detection_is_unchanged_copy():
    img = np.zeros((50, 50, 3), np.uint8)
    out = draw_detections(img, [], ["pedestrian"])
    assert out is not img and np.array_equal(out, img)


def test_same_id_same_color_across_frames():
    img = np.zeros((40, 40, 3), np.uint8)
    a = draw_tracks(img, [Track(5, 5, 15, 15, 0.9, 3, track_id=4)], CLASS_NAMES)
    b = draw_tracks(img, [Track(5, 5, 15, 15, 0.9, 0, track_id=4)], CLASS_NAMES)
    assert tuple(a[5, 10]) == tuple(b[5, 10]) != (0, 0, 0)  # top edge: color depends on the id only


def test_different_ids_get_different_colors():
    img = np.zeros((40, 40, 3), np.uint8)
    a = draw_tracks(img, [Track(5, 5, 15, 15, 0.9, 3, track_id=1)], CLASS_NAMES)
    b = draw_tracks(img, [Track(5, 5, 15, 15, 0.9, 3, track_id=2)], CLASS_NAMES)
    assert tuple(a[5, 10]) != tuple(b[5, 10])
