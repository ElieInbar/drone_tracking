"""Detect small objects by running the detector on full-resolution tiles, then merging the results."""

import numpy as np

from dronetrack.detect.merge import nms
from dronetrack.detect.tiling import make_tiles
from dronetrack.types import Detection


class TiledDetector:
    """Same interface as YoloDetector (`detect(image) -> list[Detection]`), but looks at the image tile by tile.

    `base` is any detector with `detect_batch(images) -> list[list[Detection]]`.
    With include_full, the whole (downscaled) image is also searched: large objects that no tile
    sees entirely are found there. Defaults are the best setting of experiment 2 (docs/DECISIONS.md D7):
    duplicates are dropped by IoU; merging boxes (SAHI-style) fuses neighbouring objects in dense scenes.
    """

    def __init__(
        self,
        base,
        tile: int = 640,
        overlap: float = 0.2,
        include_full: bool = True,
        merge_threshold: float = 0.5,
        merge_metric: str = "iou",
        merge_boxes: bool = False,
    ):
        self.base = base
        self.tile = tile
        self.overlap = overlap
        self.include_full = include_full
        self.merge_threshold = merge_threshold
        self.merge_metric = merge_metric
        self.merge_boxes = merge_boxes

    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        tiles = make_tiles(w, h, self.tile, self.overlap)
        crops = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in tiles]
        dets = []
        for (x1, y1, _, _), tile_dets in zip(tiles, self.base.detect_batch(crops)):
            # A box found at (10, 10) in a tile starting at (512, 0) is at (522, 10) in the image.
            dets += [Detection(d.x1 + x1, d.y1 + y1, d.x2 + x1, d.y2 + y1, d.score, d.cls) for d in tile_dets]
        if self.include_full:
            dets += self.base.detect_batch([image])[0]
        return nms(dets, self.merge_threshold, self.merge_metric, self.merge_boxes)
