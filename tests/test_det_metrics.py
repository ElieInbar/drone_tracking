import numpy as np
import pytest

from dronetrack.eval.det_metrics import average_precision, evaluate_detections
from dronetrack.types import Detection

G = {"im": [Detection(0, 0, 10, 10, 1, 0)]}


def test_perfect_predictions_give_one():
    m = evaluate_detections({"im": [Detection(0, 0, 10, 10, 0.9, 0)]}, G, [0])
    assert m.map50 == pytest.approx(1.0) and m.map50_95 == pytest.approx(1.0)


def test_no_prediction_gives_zero():
    assert evaluate_detections({"im": []}, G, [0]).map50 == 0.0


def test_false_positive_ranked_below_true_positive_is_free():
    p = {"im": [Detection(0, 0, 10, 10, 0.9, 0), Detection(50, 50, 60, 60, 0.5, 0)]}
    assert evaluate_detections(p, G, [0]).map50 == pytest.approx(1.0)


def test_false_positive_ranked_above_halves_ap():
    p = {"im": [Detection(0, 0, 10, 10, 0.5, 0), Detection(50, 50, 60, 60, 0.9, 0)]}
    assert evaluate_detections(p, G, [0]).map50 == pytest.approx(0.5, abs=0.01)


def test_duplicate_detection_counts_as_false_positive():
    p = {"im": [Detection(0, 0, 10, 10, 0.9, 0), Detection(0, 0, 10, 10, 0.8, 0)]}
    assert evaluate_detections(p, G, [0]).map50 == pytest.approx(1.0)  # the duplicate comes after the TP


def test_class_without_ground_truth_is_ignored():
    m = evaluate_detections({"im": [Detection(0, 0, 10, 10, 0.9, 0)]}, G, [0, 5])
    assert m.map50 == pytest.approx(1.0) and 5 not in m.ap50_per_class


def test_wrong_class_is_not_a_match():
    m = evaluate_detections({"im": [Detection(0, 0, 10, 10, 0.9, 3)]}, G, [0, 3])
    assert m.map50 == 0.0


def test_loose_box_counts_at_iou50_but_not_at_strict_thresholds():
    # IoU with the ground truth = 100 / 160 = 0.625: a hit at 0.50 and 0.55, 0.60; a miss from 0.65 on
    m = evaluate_detections({"im": [Detection(0, 0, 10, 16, 0.9, 0)]}, G, [0])
    assert m.map50 == pytest.approx(1.0) and m.map50_95 == pytest.approx(0.3)


def test_prediction_in_other_image_does_not_match():
    p = {"im": [], "other": [Detection(0, 0, 10, 10, 0.9, 0)]}
    assert evaluate_detections(p, G | {"other": []}, [0]).map50 == 0.0


def test_average_precision_of_a_step_curve():
    # precision 1 up to recall 0.5, then nothing: half of the area
    assert average_precision(np.array([0.5]), np.array([1.0])) == pytest.approx(51 / 101)
