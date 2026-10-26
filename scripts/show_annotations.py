"""Draw the ground-truth boxes of one VisDrone-DET image, to check the annotations by eye.

Usage: uv run scripts/show_annotations.py --raw data/VisDrone2019-DET-train --image 0000002_00005_d_0000014 \
           --out outputs/verif.jpg
"""

import argparse
from pathlib import Path

import cv2

from dronetrack.data.visdrone import CLASS_NAMES, parse_det_annotation
from dronetrack.viz.draw import draw_detections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--image", required=True, help="image name without extension")
    parser.add_argument("--out", type=Path, default=Path("outputs/verif.jpg"))
    args = parser.parse_args()
    image = cv2.imread(str(args.raw / "images" / f"{args.image}.jpg"))
    h, w = image.shape[:2]
    dets = parse_det_annotation(args.raw / "annotations" / f"{args.image}.txt", w, h)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), draw_detections(image, dets, CLASS_NAMES))
    print(f"{len(dets)} objects drawn on {args.out}")


if __name__ == "__main__":
    main()
