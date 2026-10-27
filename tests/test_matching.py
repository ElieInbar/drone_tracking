import numpy as np

from dronetrack.track.matching import associate

B = np.array([[0, 0, 10, 10], [100, 0, 110, 10], [200, 0, 210, 10]], float)


def test_three_objects_matched_whatever_the_order():
    dets = B[[2, 0, 1]] + 2  # same objects, shuffled, moved 2 px
    m, ut, ud = associate(B, np.zeros(3), dets, np.zeros(3), 0.2)
    assert sorted(m) == [(0, 1), (1, 2), (2, 0)] and ut == [] and ud == []


def test_different_class_never_matched():
    m, ut, ud = associate(B[:1], np.array([3]), B[:1], np.array([0]), 0.2)
    assert m == [] and ut == [0] and ud == [0]


def test_low_overlap_rejected():
    m, ut, ud = associate(B[:1], np.zeros(1), B[:1] + 9, np.zeros(1), 0.2)  # IoU 1/199
    assert m == [] and ut == [0] and ud == [0]


def test_hungarian_beats_greedy():
    # Greedy: track 0 takes its best det (IoU 0.54) and track 1 is left with nothing usable.
    # Hungarian gives each track a det: track 0 <- det 1 (IoU 0.43), track 1 <- det 0 (IoU 0.54).
    tracks = np.array([[0, 0, 10, 10], [6, 0, 16, 10]], float)
    dets = np.array([[3, 0, 13, 10], [-4, 0, 6, 10]], float)
    m, _, _ = associate(tracks, np.zeros(2), dets, np.zeros(2), 0.2)
    assert sorted(m) == [(0, 1), (1, 0)]


def test_more_detections_than_tracks():
    m, ut, ud = associate(B[:1], np.zeros(1), B, np.zeros(3), 0.2)
    assert m == [(0, 0)] and ut == [] and ud == [1, 2]


def test_empty_inputs():
    assert associate(np.zeros((0, 4)), np.zeros(0), B, np.zeros(3), 0.2) == ([], [], [0, 1, 2])
    assert associate(B, np.zeros(3), np.zeros((0, 4)), np.zeros(0), 0.2) == ([], [0, 1, 2], [])
