"""Command-line options shared by the scripts that run a detector (plain or tiled)."""

import argparse

from dronetrack.detect.tiled import TiledDetector
from dronetrack.detect.yolo import COCO_TO_VISDRONE, YoloDetector


def add_detector_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weights", required=True)
    parser.add_argument("--coco", action="store_true", help="model is COCO-pretrained: map its classes to VisDrone")
    parser.add_argument("--imgsz", type=int, default=640)
    tiling = parser.add_argument_group("tiling (small objects)")
    tiling.add_argument("--tiles", action="store_true", help="detect on full-resolution tiles")
    tiling.add_argument("--tile", type=int, default=640)
    tiling.add_argument("--overlap", type=float, default=0.2)
    tiling.add_argument("--no-full", action="store_true", help="tiles only, skip the whole-image pass")
    tiling.add_argument("--merge-metric", choices=["iou", "ios"], default="ios")
    tiling.add_argument("--merge-threshold", type=float, default=0.5)
    tiling.add_argument("--no-merge-boxes", action="store_true", help="drop duplicates instead of merging them")


def build_detector(args: argparse.Namespace, conf: float) -> YoloDetector | TiledDetector:
    base = YoloDetector(args.weights, conf=conf, imgsz=args.imgsz, class_map=COCO_TO_VISDRONE if args.coco else None)
    if not args.tiles:
        return base
    return TiledDetector(
        base,
        tile=args.tile,
        overlap=args.overlap,
        include_full=not args.no_full,
        merge_threshold=args.merge_threshold,
        merge_metric=args.merge_metric,
        merge_boxes=not args.no_merge_boxes,
    )
