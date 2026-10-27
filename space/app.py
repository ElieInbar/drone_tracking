"""Hugging Face Space: upload a drone video, get it back with every object boxed and numbered.

The pipeline itself lives in the dronetrack package (dronetrack.demo.track_video); this file is only the web page.
Run locally:  DRONETRACK_WEIGHTS=models/visdrone_y26n.pt uv run --group demo python space/app.py
"""

import os
import tempfile
import urllib.request
from pathlib import Path

import gradio as gr

from dronetrack.demo import MAX_FRAMES, track_video
from dronetrack.detect.yolo import YoloDetector
from dronetrack.track.byte_tracker import ByteTracker
from dronetrack.track.factory import TRACK_CONF

HERE = Path(__file__).parent
REPO_URL = "https://github.com/ElieInbar/drone_tracking"
WEIGHTS_URL = f"{REPO_URL}/releases/download/v1.0/visdrone_y26n.pt"
WEIGHTS_PATH = Path(os.environ.get("DRONETRACK_WEIGHTS", HERE / "visdrone_y26n.pt"))
EXAMPLE = HERE / "examples" / "visdrone_086.mp4"

INTRO = f"""
# Drone Tracking — detect and track small objects in drone footage
Upload a drone video (or pick the example below): every vehicle and pedestrian gets a box and a number it keeps
from frame to frame. YOLO26n fine-tuned on VisDrone + a ByteTrack-style tracker written from scratch.
Runs on a free CPU, so only the first {MAX_FRAMES} frames are processed (about a minute). Code and results: [{REPO_URL}]({REPO_URL})
"""

RESULTS = """
| Measured on VisDrone val | Result |
|---|---|
| Detection, mAP@50 (off-the-shelf → fine-tuned) | 0.14 → **0.34** |
| Small objects with tiling (not used here: 10× slower) | **+66%** |
| Tracking, MOTA (Ultralytics ByteTrack → ours, same boxes) | 0.177 → **0.193** |
| Identity switches vs Ultralytics ByteTrack | **−23%** |
"""

_detector = None


def ensure_weights(path: Path = WEIGHTS_PATH, url: str = WEIGHTS_URL) -> Path:
    """Download the model once; a half-finished download never sits under the real name."""
    path = Path(path)
    if not path.exists():
        part = path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(url, part)
            part.rename(path)
        finally:
            part.unlink(missing_ok=True)
    return path


def _get_detector() -> YoloDetector:
    global _detector
    if _detector is None:
        _detector = YoloDetector(str(ensure_weights()), conf=TRACK_CONF, device="cpu")
    return _detector


def run(video_path: str | None) -> tuple[str, str]:
    """Annotated MP4 path and a one-line summary, for the page."""
    if not video_path:
        raise gr.Error("Please upload a video or pick the example.")
    out = Path(tempfile.mkdtemp()) / "tracked.mp4"
    try:
        result = track_video(Path(video_path), out, _get_detector(), ByteTracker())
    except ValueError as e:
        raise gr.Error(str(e)) from e
    return str(out), result.summary()


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Drone Tracking") as ui:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column():
                video_in = gr.Video(label="Drone video", sources=["upload"])
                button = gr.Button("Track", variant="primary")
                gr.Examples(examples=[[str(EXAMPLE)]], inputs=[video_in], label="Example (VisDrone-MOT val)")
            with gr.Column():
                video_out = gr.Video(label="Tracked objects", autoplay=True)
                summary = gr.Textbox(label="Summary", lines=2, interactive=False)
        gr.Markdown(RESULTS)
        button.click(run, inputs=video_in, outputs=[video_out, summary])
    return ui


if __name__ == "__main__":
    build_ui().launch()
