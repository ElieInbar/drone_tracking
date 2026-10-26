"""Read a video, or a folder of numbered images, one frame at a time."""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def iter_frames(source: Path) -> Iterator[np.ndarray]:
    """Yield BGR frames from a video file or from the images of a folder, in name order."""
    source = Path(source)
    if source.is_dir():
        # VisDrone-MOT sequences are folders of 0000001.jpg, 0000002.jpg, ...: name order = time order
        for path in sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
            yield cv2.imread(str(path))
        return
    capture = cv2.VideoCapture(str(source))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame
    finally:
        capture.release()
