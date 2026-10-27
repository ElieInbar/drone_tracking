"""Thin wrapper turning Ultralytics YOLO outputs into our Detection objects."""

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from ultralytics import YOLO

from dronetrack.types import Detection

# COCO class index -> VisDrone class index, for the classes both datasets share.
# person -> pedestrian, bicycle -> bicycle, car -> car, motorcycle -> motor, bus -> bus, truck -> truck
COCO_TO_VISDRONE: dict[int, int] = {0: 0, 1: 2, 2: 3, 3: 9, 5: 8, 7: 5}


def to_detections(
    xyxy: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    class_map: Mapping[int, int] | None,
) -> list[Detection]:
    """Build Detections from raw arrays; with a class_map, rename classes and drop the unmapped ones."""
    dets = []
    for (x1, y1, x2, y2), score, cls in zip(xyxy, scores, classes):
        cls = int(cls)
        if class_map is not None:
            if cls not in class_map:
                continue
            cls = class_map[cls]
        dets.append(Detection(float(x1), float(y1), float(x2), float(y2), float(score), cls))
    return dets


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class YoloDetector:
    """Run a YOLO model on one image (BGR numpy array) and return a list of Detection."""

    def __init__(
        self,
        weights: str,
        conf: float = 0.25,
        imgsz: int = 640,
        class_map: Mapping[int, int] | None = None,
        device: str | None = None,
    ):
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.class_map = class_map
        self.device = device or default_device()

    def detect(self, image: np.ndarray) -> list[Detection]:
        return self.detect_batch([image])[0]

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[Detection]]:
        """One model call for several images (faster on a GPU than one call per image)."""
        results = self.model.predict(list(images), conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)
        return [
            to_detections(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy(), self.class_map)
            for r in results
        ]
