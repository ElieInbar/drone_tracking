import pytest

from dronetrack.eval.mot_metrics import evaluate_tracking
from dronetrack.types import Detection, Track


def tr(x, tid, cls=3):
    return Track(x, 0, x + 10, 10, 1.0, cls, track_id=tid)


GT = {1: [tr(0, 1)], 2: [tr(5, 1)], 3: [tr(10, 1)], 4: [tr(15, 1)]}


def test_perfect_tracking():
    m = evaluate_tracking(GT, {f: [tr(b[0].x1, 9)] for f, b in GT.items()}, [3])
    assert (m.mota, m.idf1, m.idsw, m.fp, m.fn) == (1.0, 1.0, 0, 0, 0)


def test_one_identity_switch():
    pred = {1: [tr(0, 9)], 2: [tr(5, 9)], 3: [tr(10, 4)], 4: [tr(15, 4)]}
    m = evaluate_tracking(GT, pred, [3])
    assert m.idsw == 1 and m.mota == pytest.approx(0.75) and m.idf1 == pytest.approx(0.5)


def test_missed_frame_and_false_alarm():
    pred = {1: [tr(0, 9)], 2: [], 3: [tr(10, 9), tr(100, 5)], 4: [tr(15, 9)]}
    m = evaluate_tracking(GT, pred, [3])
    assert (m.fn, m.fp, m.idsw) == (1, 1, 0) and m.mota == pytest.approx(0.5)
    assert m.idf1 == pytest.approx(2 * 3 / (4 + 4))


def test_prediction_in_ignored_region_is_not_a_false_alarm():
    pred = {1: [tr(0, 9), tr(100, 5)]}
    m = evaluate_tracking({1: [tr(0, 1)]}, pred, [3], ignored={1: [Detection(95, 0, 120, 10, 1.0, -1)]})
    assert m.fp == 0 and m.mota == 1.0 and m.idf1 == 1.0


def test_other_classes_are_not_evaluated():
    m = evaluate_tracking(GT, {f: [tr(b[0].x1, 9, cls=0)] for f, b in GT.items()}, [3])
    assert (m.fn, m.fp) == (4, 0)


def test_a_continuing_match_is_kept_even_if_another_box_fits_better():
    # Two predictions on the object at frame 2: CLEAR MOT keeps the identity it already had (9), no switch.
    gt = {1: [tr(0, 1)], 2: [tr(0, 1)]}
    pred = {1: [tr(0, 9)], 2: [tr(2, 9), tr(0, 5)]}
    m = evaluate_tracking(gt, pred, [3])
    assert m.idsw == 0 and m.fp == 1


def test_switching_back_counts_twice():
    gt = {f: [tr(0, 1)] for f in range(1, 4)}
    pred = {1: [tr(0, 9)], 2: [tr(0, 4)], 3: [tr(0, 9)]}
    assert evaluate_tracking(gt, pred, [3]).idsw == 2


def test_classes_are_summed():
    gt = {1: [tr(0, 1, cls=3), tr(50, 2, cls=0)]}
    m = evaluate_tracking(gt, {1: [tr(0, 9, cls=3)]}, [0, 3])
    assert (m.n_gt, m.fn, m.mota) == (2, 1, 0.5)


def test_one_prediction_is_never_matched_to_two_objects():
    # Objects 1 and 2 were both followed by n°9 at some point; at frame 3 n°9 overlaps both.
    a, b = Track(0, 0, 10, 10, 1.0, 3, track_id=1), Track(1, 0, 11, 10, 1.0, 3, track_id=2)
    gt = {1: [a], 2: [b], 3: [a, b]}
    pred = {1: [tr(0, 9)], 2: [tr(1, 9)], 3: [tr(0, 9)]}
    m = evaluate_tracking(gt, pred, [3])
    assert m.fn == 1 and m.fp == 0
