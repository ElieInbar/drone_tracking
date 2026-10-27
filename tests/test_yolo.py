import numpy as np
import pytest

from dronetrack.detect.yolo import COCO_TO_VISDRONE, YoloDetector, to_detections
from dronetrack.types import Detection


def test_to_detections_applies_class_map_and_drops_unknown():
    dets = to_detections(
        np.array([[0, 0, 5, 5], [1, 1, 6, 6]], float),
        np.array([0.9, 0.8]),
        np.array([2, 14]),  # 2 = car, 14 = bird (no VisDrone equivalent)
        COCO_TO_VISDRONE,
    )
    assert dets == [Detection(0, 0, 5, 5, 0.9, 3)]


def test_to_detections_without_map_keeps_classes():
    dets = to_detections(np.array([[0, 0, 5, 5]], float), np.array([0.5]), np.array([7]), None)
    assert dets[0].cls == 7


def test_to_detections_empty():
    assert to_detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0), COCO_TO_VISDRONE) == []


@pytest.mark.slow
def test_yolo_detector_runs_on_blank_image():
    assert YoloDetector("yolo26n.pt").detect(np.zeros((640, 640, 3), np.uint8)) == []


@pytest.mark.slow
def test_yolo_detect_batch_matches_detect():
    det = YoloDetector("yolo26n.pt")
    img = np.zeros((480, 640, 3), np.uint8)
    img[100:300, 200:400] = 255
    assert det.detect_batch([img, img]) == [det.detect(img), det.detect(img)]
