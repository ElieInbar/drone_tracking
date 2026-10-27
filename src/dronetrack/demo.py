"""The online demo's pipeline: video in -> detect, track, draw -> browser-playable MP4 out.

Kept outside the Gradio app so it can be tested without a web page or a model.
"""

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from dronetrack.data.visdrone import CLASS_NAMES
from dronetrack.io import iter_frames
from dronetrack.viz.draw import draw_tracks

# Free Hugging Face hardware is 2 CPU cores, no GPU: cap the work so a visitor waits about a minute.
MAX_FRAMES = 300
MAX_WIDTH = 1280
DEFAULT_FPS = 10  # VisDrone-MOT sequences are folders of images, with no frame rate


@dataclass
class DemoResult:
    frames: int
    truncated: bool  # the video had more than max_frames frames
    objects_per_class: dict[str, int] = field(default_factory=dict)  # distinct identities, most frequent first
    seconds: float = 0.0

    def summary(self) -> str:
        frames = f"{self.frames} frames" + (f" (first {self.frames} only)" if self.truncated else "")
        total = sum(self.objects_per_class.values())
        objects = f"{total} objects tracked"
        if total:
            by_count = sorted(self.objects_per_class.items(), key=lambda kv: -kv[1])
            objects += ": " + ", ".join(f"{name} {n}" for name, n in by_count)
        return f"{frames} in {self.seconds:.0f} s · {objects}"


def resize_for_demo(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Shrink frames wider than max_width (keeping proportions), then crop to even sizes (H.264 needs them)."""
    h, w = frame.shape[:2]
    if w > max_width:
        h, w = round(h * max_width / w), max_width
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    return frame[: h - h % 2, : w - w % 2]


def _source_fps(source: Path) -> float:
    if Path(source).is_dir():
        return DEFAULT_FPS
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return fps if fps and fps > 0 else DEFAULT_FPS


def track_video(
    source: Path, out: Path, detector, tracker, max_frames: int = MAX_FRAMES, max_width: int = MAX_WIDTH
) -> DemoResult:
    """Detect and track every frame of `source` (up to max_frames) and write the annotated video to `out`."""
    start = time.perf_counter()
    class_of: dict[int, int] = {}  # track number -> class
    writer, frames, truncated = None, 0, False
    try:
        for frame in iter_frames(source):
            if frame is None:
                continue  # unreadable image in a folder
            if frames == max_frames:
                truncated = True
                break
            frame = resize_for_demo(frame, max_width)
            tracks = tracker.update(detector.detect(frame))
            class_of.update({t.track_id: t.cls for t in tracks})
            if writer is None:
                writer = imageio.get_writer(
                    out, fps=_source_fps(source), codec="libx264", pixelformat="yuv420p", macro_block_size=1
                )
            writer.append_data(cv2.cvtColor(draw_tracks(frame, tracks, CLASS_NAMES), cv2.COLOR_BGR2RGB))
            frames += 1
    finally:
        if writer is not None:
            writer.close()
    if frames == 0:
        raise ValueError("Could not read any frame from this video")
    counts = Counter(CLASS_NAMES[c] for c in class_of.values())
    return DemoResult(frames, truncated, dict(counts.most_common()), time.perf_counter() - start)
