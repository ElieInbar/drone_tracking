from dronetrack.detect.merge import nms
from dronetrack.types import Detection


def test_nms_iou_removes_lower_score_duplicate():
    kept = nms([Detection(0, 0, 10, 10, 0.9, 3), Detection(1, 0, 11, 10, 0.6, 3)], 0.5, "iou", merge=False)
    assert [d.score for d in kept] == [0.9]


def test_partial_box_cut_by_a_tile_border():
    full = Detection(0, 0, 40, 20, 0.8, 3)
    partial = Detection(0, 0, 12, 20, 0.85, 3)  # IoU 0.3, IoS 1.0
    assert nms([full, partial], 0.5, "iou", merge=False) == [partial, full]  # IoU misses the duplicate
    assert nms([full, partial], 0.5, "ios", merge=False) == [partial]  # IoS finds it but keeps the cut box
    assert nms([full, partial], 0.5, "ios", merge=True) == [Detection(0, 0, 40, 20, 0.85, 3)]  # whole object


def test_two_halves_of_a_truck_and_the_full_box_become_one():
    left, right = Detection(0, 0, 110, 40, 0.7, 5), Detection(90, 0, 200, 40, 0.6, 5)
    full = Detection(2, 1, 198, 39, 0.9, 5)
    assert nms([left, right, full]) == [Detection(0, 0, 200, 40, 0.9, 5)]


def test_nms_never_suppresses_another_class():
    car, van = Detection(0, 0, 10, 10, 0.9, 3), Detection(0, 0, 10, 10, 0.8, 4)
    assert len(nms([car, van])) == 2


def test_distant_objects_are_all_kept():
    dets = [Detection(i * 20, 0, i * 20 + 10, 10, 0.5, 0) for i in range(5)]
    assert len(nms(dets)) == 5


def test_nms_empty():
    assert nms([]) == []


def test_zero_area_box_does_not_crash():
    flat = Detection(5, 5, 5, 9, 0.4, 0)  # zero width
    assert nms([flat, Detection(0, 0, 10, 10, 0.9, 0)])[0].score == 0.9
