"""Run a detector on a video (or a folder of frames) and save an annotated MP4 and/or GIF.

Example (step 0, COCO-pretrained model mapped to VisDrone classes):
  uv run scripts/run_detect.py --source data/VisDrone2019-MOT-val/sequences/uav0000086_00000_v \
      --weights yolo26n.pt --coco --gif outputs/etape0_coco.gif --max-frames 150
"""

import argparse
import itertools
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio

from dronetrack.data.visdrone import CLASS_NAMES
from dronetrack.detect.yolo import COCO_TO_VISDRONE, YoloDetector
from dronetrack.io import iter_frames
from dronetrack.viz.draw import draw_detections

GIF_WIDTH = 640  # GIFs are downscaled to stay small enough for a README


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--coco", action="store_true", help="model is COCO-pretrained: map its classes to VisDrone")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", type=Path, help="annotated MP4 path")
    parser.add_argument("--gif", type=Path, help="annotated GIF path")
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    detector = YoloDetector(args.weights, conf=args.conf, class_map=COCO_TO_VISDRONE if args.coco else None)
    writer, gif_frames, n_dets, start = None, [], 0, time.perf_counter()
    frames = itertools.islice(iter_frames(args.source), args.max_frames)
    for i, frame in enumerate(frames):
        dets = detector.detect(frame)
        n_dets += len(dets)
        drawn = draw_detections(frame, dets, CLASS_NAMES)
        if args.out:
            if writer is None:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                h, w = drawn.shape[:2]
                writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            writer.write(drawn)
        if args.gif:
            h, w = drawn.shape[:2]
            small = cv2.resize(drawn, (GIF_WIDTH, round(h * GIF_WIDTH / w)))
            gif_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    n_frames = i + 1
    elapsed = time.perf_counter() - start
    if writer is not None:
        writer.release()
    if args.gif:
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.gif, gif_frames, duration=1000 / args.fps, loop=0)
    print(f"{n_frames} frames, {n_dets / n_frames:.1f} detections/frame, {n_frames / elapsed:.1f} frames/s")


if __name__ == "__main__":
    main()
