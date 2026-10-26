import numpy as np

from dronetrack.types import Detection
from dronetrack.viz.draw import draw_detections


def test_draw_returns_copy_and_marks_pixels():
    img = np.zeros((50, 50, 3), np.uint8)
    out = draw_detections(img, [Detection(10, 10, 30, 30, 0.9, 0)], ["pedestrian"])
    assert img.sum() == 0 and out.sum() > 0


def test_draw_with_no_detection_is_unchanged_copy():
    img = np.zeros((50, 50, 3), np.uint8)
    out = draw_detections(img, [], ["pedestrian"])
    assert out is not img and np.array_equal(out, img)
