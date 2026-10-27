"""Turn the first frames of a VisDrone-MOT sequence (a folder of images) into a short MP4 for the online demo.

Example:
  uv run scripts/make_example_clip.py --sequence data/VisDrone2019-MOT-val/sequences/uav0000086_00000_v \
      --out space/examples/visdrone_086.mp4 --frames 100 --fps 10
"""

import argparse
import itertools
from pathlib import Path

import cv2
import imageio.v2 as imageio

from dronetrack.demo import MAX_WIDTH, resize_for_demo
from dronetrack.io import iter_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1)
    for frame in itertools.islice(iter_frames(args.sequence), args.frames):
        writer.append_data(cv2.cvtColor(resize_for_demo(frame, MAX_WIDTH), cv2.COLOR_BGR2RGB))
    writer.close()
    print(f"{args.out}: {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
