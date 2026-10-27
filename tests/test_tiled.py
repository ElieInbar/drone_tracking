import numpy as np

from dronetrack.detect.tiled import TiledDetector
from dronetrack.types import Detection


class FakeDetector:
    """Finds one car at (10, 10, 30, 30) in every image it is given; records the sizes it saw."""

    def __init__(self):
        self.sizes = []

    def detect_batch(self, images):
        self.sizes += [im.shape[:2] for im in images]
        return [[Detection(10, 10, 30, 30, 0.9, 3)] for _ in images]


def test_tile_boxes_are_shifted_to_image_coordinates():
    fake = FakeDetector()
    dets = TiledDetector(fake, tile=640, overlap=0.2, include_full=False).detect(np.zeros((640, 1152, 3), np.uint8))
    assert sorted((d.x1, d.y1) for d in dets) == [(10, 10), (522, 10)]
    assert fake.sizes == [(640, 640), (640, 640)]


def test_tiles_are_cut_from_the_right_place():
    image = np.zeros((640, 1152, 3), np.uint8)
    image[:, 512:] = 7  # right part of the image is grey 7
    seen = []

    class Recorder(FakeDetector):
        def detect_batch(self, images):
            seen.extend(int(im[0, 0, 0]) for im in images)
            return super().detect_batch(images)

    TiledDetector(Recorder(), include_full=False).detect(image)
    assert seen == [0, 7]  # tile 1 starts at x=0, tile 2 at x=512


def test_full_image_pass_is_added_and_duplicates_merged():
    fake = FakeDetector()
    dets = TiledDetector(fake, include_full=True).detect(np.zeros((640, 640, 3), np.uint8))
    assert len(fake.sizes) == 2  # one tile + the full image
    assert len(dets) == 1  # both see the same car at the same place
