"""Kalman filter for one moving box: predict where it will be next frame, then correct with what was seen."""

import numpy as np

# Noise is proportional to the box size, as in ByteTrack: a 100 px truck can be off by more pixels
# than a 10 px pedestrian. Position noise = 1/20 of the size, velocity noise = 1/160.
STD_POSITION = 1 / 20
STD_VELOCITY = 1 / 160


def _to_state(box: np.ndarray) -> np.ndarray:
    """[x1, y1, x2, y2] -> [cx, cy, w, h] (centre and size: they move more smoothly than corners)."""
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], float)


def _to_box(cxcywh: np.ndarray) -> np.ndarray:
    cx, cy, w, h = cxcywh
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], float)


class KalmanBoxFilter:
    """Constant-velocity Kalman filter on [cx, cy, w, h, vx, vy, vw, vh] (one instance per track).

    predict(): move the box by its velocity and grow the uncertainty.
    update(box): blend the prediction with the measured box, trusting whichever is less uncertain.
    """

    def __init__(self, box: np.ndarray):
        self.x = np.concatenate([_to_state(box), np.zeros(4)])  # state: position+size, then their velocities
        size = self._size()
        std = np.concatenate([2 * STD_POSITION * size, 10 * STD_VELOCITY * size])
        self.P = np.diag(std**2)  # uncertainty of the state; velocity unknown at start, so large
        # Motion model F: next position = position + velocity (one frame later); velocity unchanged.
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        # Measurement model H: the detector measures position and size, not velocity.
        self.H = np.eye(4, 8)

    def _size(self) -> np.ndarray:
        """Scale used for the noise of [cx, cy, w, h]: height for the centre and height, width for the width."""
        w, h = self.x[2], self.x[3]
        return np.array([h, h, w, h], float)

    @property
    def box(self) -> np.ndarray:
        return _to_box(self.x[:4])

    def predict(self) -> np.ndarray:
        size = self._size()
        Q = np.diag(np.concatenate([STD_POSITION * size, STD_VELOCITY * size]) ** 2)  # motion uncertainty
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q
        return self.box

    def update(self, box: np.ndarray) -> None:
        z = _to_state(box)
        R = np.diag((STD_POSITION * self._size()) ** 2)  # detector uncertainty
        S = self.H @ self.P @ self.H.T + R  # total uncertainty of the predicted measurement
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain: how much to trust the measurement
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P
