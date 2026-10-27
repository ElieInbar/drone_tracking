import argparse

import pytest

from dronetrack.detect.factory import add_detector_args, build_detector
from dronetrack.detect.tiled import TiledDetector
from dronetrack.detect.yolo import YoloDetector


def _parse(argv):
    parser = argparse.ArgumentParser()
    add_detector_args(parser)
    return parser.parse_args(argv)


def test_defaults_are_the_plan_values():
    a = _parse(["--weights", "w.pt"])
    assert (a.tiles, a.tile, a.overlap, a.merge_metric, a.merge_threshold) == (False, 640, 0.2, "ios", 0.5)
    assert not a.no_full and not a.no_merge_boxes


@pytest.mark.slow
def test_build_tiled_detector_from_flags():
    det = build_detector(_parse(["--weights", "yolo26n.pt", "--tiles", "--no-full", "--merge-metric", "iou",
                                 "--no-merge-boxes"]), conf=0.1)
    assert isinstance(det, TiledDetector) and isinstance(det.base, YoloDetector)
    assert (det.include_full, det.merge_metric, det.merge_boxes, det.base.conf) == (False, "iou", False, 0.1)


@pytest.mark.slow
def test_build_plain_detector_without_tiles():
    assert isinstance(build_detector(_parse(["--weights", "yolo26n.pt"]), conf=0.25), YoloDetector)
