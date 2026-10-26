"""The two objects every stage of the pipeline exchanges."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One box found in one image. Coordinates are pixels: (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float  # detector confidence in [0, 1]; 1.0 for ground truth
    cls: int  # class index, see dronetrack.data.visdrone.CLASS_NAMES

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class Track(Detection):
    """A Detection plus an identity that stays the same from frame to frame."""

    track_id: int = -1
