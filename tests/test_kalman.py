import numpy as np
import pytest

from dronetrack.track.kalman import KalmanBoxFilter


def follow_moving_box(kf, frames=10, speed=5):
    """Feed the filter a 10x10 box moving `speed` px right per frame."""
    for t in range(1, frames + 1):
        kf.predict()
        kf.update(np.array([speed * t, 0, speed * t + 10, 10], float))


def test_starts_at_the_first_box():
    kf = KalmanBoxFilter(np.array([0, 0, 10, 10], float))
    assert kf.box == pytest.approx([0, 0, 10, 10])


def test_learns_constant_velocity():
    kf = KalmanBoxFilter(np.array([0, 0, 10, 10], float))
    follow_moving_box(kf)
    assert kf.predict() == pytest.approx([55, 0, 65, 10], abs=1.0)


def test_predict_without_update_keeps_moving():
    kf = KalmanBoxFilter(np.array([0, 0, 10, 10], float))
    follow_moving_box(kf)
    for _ in range(3):  # occluded for 3 frames: no measurement
        box = kf.predict()
    assert box[0] == pytest.approx(65, abs=2.0)


def test_update_pulls_toward_measurement():
    kf = KalmanBoxFilter(np.array([0, 0, 10, 10], float))
    kf.predict()
    kf.update(np.array([4, 0, 14, 10], float))
    assert 0 < kf.box[0] <= 4
