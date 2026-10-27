"""Save annotated frames as an MP4 and/or a small GIF (for READMEs)."""

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


class AnnotatedVideo:
    """Collect annotated BGR frames with add(), then write them with close()."""

    def __init__(self, mp4: Path | None, gif: Path | None, fps: int = 10, gif_width: int = 480):
        self.mp4, self.gif, self.fps, self.gif_width = mp4, gif, fps, gif_width
        self.writer = None
        self.gif_frames: list[np.ndarray] = []

    def add(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if self.mp4:
            if self.writer is None:
                Path(self.mp4).parent.mkdir(parents=True, exist_ok=True)
                self.writer = cv2.VideoWriter(str(self.mp4), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            self.writer.write(frame)
        if self.gif:
            small = cv2.resize(frame, (self.gif_width, round(h * self.gif_width / w)))
            self.gif_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
        if self.gif and self.gif_frames:
            Path(self.gif).parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(self.gif, self.gif_frames, duration=1000 / self.fps, loop=0)
