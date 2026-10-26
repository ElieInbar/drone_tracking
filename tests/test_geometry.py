import numpy as np
import pytest

from dronetrack.geometry import boxes_to_array, iou_matrix
from dronetrack.types import Detection


def test_iou_identical_is_one():
    b = np.array([[0, 0, 10, 10]], float)
    assert iou_matrix(b, b)[0, 0] == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[20, 20, 30, 30]], float)
    assert iou_matrix(a, b)[0, 0] == 0.0


def test_iou_half_overlap():
    # intersection 5x10 = 50, union 100 + 100 - 50 = 150
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[5, 0, 15, 10]], float)
    assert iou_matrix(a, b)[0, 0] == pytest.approx(1 / 3)


def test_iou_matrix_shape_is_n_by_m():
    a = np.array([[0, 0, 10, 10], [0, 0, 5, 5]], float)
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [0, 0, 5, 5]], float)
    m = iou_matrix(a, b)
    assert m.shape == (2, 3)
    assert m[1, 2] == pytest.approx(1.0) and m[1, 0] == pytest.approx(0.25)


def test_iou_empty_shapes():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)


def test_boxes_to_array():
    arr = boxes_to_array([Detection(1, 2, 3, 4, 0.5, 0)])
    assert arr.shape == (1, 4) and arr.tolist() == [[1, 2, 3, 4]]


def test_boxes_to_array_empty():
    assert boxes_to_array([]).shape == (0, 4)
