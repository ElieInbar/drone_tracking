"""Detect and track objects in a VisDrone-MOT sequence (or any video); save a video / GIF with track numbers.

Detections are cached in outputs/dets/, so the tracker can be re-run in seconds on the same boxes.

Example:
  uv run scripts/run_track.py --source data/VisDrone2019-MOT-val/sequences/uav0000086_00000_v \
      --weights models/visdrone_y26n.pt --gif outputs/track.gif --max-frames 60
"""

import argparse
import itertools
import time
from pathlib import Path

from dronetrack.data.visdrone import CLASS_NAMES
from dronetrack.detect.factory import add_detector_args, build_detector
from dronetrack.io import iter_frames
from dronetrack.track.byte_tracker import ByteTracker
from dronetrack.track.cache import detect_sequence
from dronetrack.viz.draw import draw_tracks
from dronetrack.viz.video import AnnotatedVideo

TRACK_CONF = 0.1  # the tracker needs low-score boxes for its second pass


def cache_path(cache_dir: Path, source: Path, args: argparse.Namespace) -> Path:
    variant = Path(args.weights).stem + ("_tiles" if args.tiles else "")
    return cache_dir / f"{Path(source).stem}_{variant}.json"


def make_tracker(name: str):
    if name == "ours":
        return ByteTracker()
    raise ValueError(f"unknown tracker {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    add_detector_args(parser)
    parser.add_argument("--tracker", choices=["ours"], default="ours")
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/dets"))
    parser.add_argument("--out", type=Path, help="annotated MP4 path")
    parser.add_argument("--gif", type=Path, help="annotated GIF path")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--gif-width", type=int, default=480)
    args = parser.parse_args()

    dets = detect_sequence(build_detector(args, conf=TRACK_CONF), args.source, cache_path(args.cache_dir, args.source, args))
    tracker = make_tracker(args.tracker)
    video = AnnotatedVideo(args.out, args.gif, args.fps, args.gif_width)
    ids, n_frames, tracking_time = set(), 0, 0.0
    for frame, frame_dets in itertools.islice(zip(iter_frames(args.source), dets), args.max_frames):
        start = time.perf_counter()
        tracks = tracker.update(frame_dets)
        tracking_time += time.perf_counter() - start
        ids |= {t.track_id for t in tracks}
        n_frames += 1
        video.add(draw_tracks(frame, tracks, CLASS_NAMES))
    video.close()
    print(f"{n_frames} frames, {len(ids)} track numbers, tracker alone {n_frames / tracking_time:.0f} frames/s")


if __name__ == "__main__":
    main()
