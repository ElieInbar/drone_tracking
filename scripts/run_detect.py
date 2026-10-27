"""Run a detector on a video (or a folder of frames) and save an annotated MP4 and/or GIF.

Example (step 0, COCO-pretrained model mapped to VisDrone classes):
  uv run scripts/run_detect.py --source data/VisDrone2019-MOT-val/sequences/uav0000086_00000_v \
      --weights yolo26n.pt --coco --gif outputs/etape0_coco.gif --max-frames 150
"""

import argparse
import itertools
import time
from pathlib import Path


from dronetrack.data.visdrone import CLASS_NAMES
from dronetrack.detect.factory import add_detector_args, build_detector
from dronetrack.io import iter_frames
from dronetrack.viz.draw import draw_detections
from dronetrack.viz.video import AnnotatedVideo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    add_detector_args(parser)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", type=Path, help="annotated MP4 path")
    parser.add_argument("--gif", type=Path, help="annotated GIF path")
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--gif-width", type=int, default=480, help="GIFs are downscaled to stay small enough for a README")
    args = parser.parse_args()

    detector = build_detector(args, conf=args.conf)
    video = AnnotatedVideo(args.out, args.gif, args.fps, args.gif_width)
    n_frames = n_dets = 0
    start = time.perf_counter()
    for frame in itertools.islice(iter_frames(args.source), args.max_frames):
        dets = detector.detect(frame)
        n_frames, n_dets = n_frames + 1, n_dets + len(dets)
        video.add(draw_detections(frame, dets, CLASS_NAMES))
    elapsed = time.perf_counter() - start
    video.close()
    print(f"{n_frames} frames, {n_dets / n_frames:.1f} detections/frame, {n_frames / elapsed:.1f} frames/s")


if __name__ == "__main__":
    main()
