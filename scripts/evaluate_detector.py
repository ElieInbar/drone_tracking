"""Score a detector on VisDrone-DET val with our own mAP (and optionally Ultralytics' for comparison).

Examples:
  # experiment 1, "before": COCO-pretrained model on the 6 shared classes, pedestrian + people merged
  uv run scripts/evaluate_detector.py --weights yolo26n.pt --coco --classes 0 2 3 5 8 9 --merge-person
  # experiment 2: tiles + whole image, duplicates dropped by IoU (default tiling settings)
  uv run scripts/evaluate_detector.py --weights models/visdrone_y26n.pt --tiles
  # check our mAP against Ultralytics on the fine-tuned model (10 classes)
  uv run scripts/evaluate_detector.py --weights models/visdrone_y26n.pt \
      --compare-ultralytics data/visdrone_yolo/data.yaml
"""

import argparse
import time
from pathlib import Path

import cv2
from PIL import Image

from dronetrack.data.visdrone import CLASS_NAMES, parse_det_annotation
from dronetrack.detect.factory import add_detector_args, build_detector
from dronetrack.detect.yolo import default_device
from dronetrack.eval.det_metrics import PERSON_MERGE, SIZE_RANGES, evaluate_detections, remap_classes

EVAL_CONF = 0.001  # keep almost every box: mAP scores all confidence thresholds at once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_detector_args(parser)
    parser.add_argument("--raw", type=Path, default=Path("data/VisDrone2019-DET-val"))
    parser.add_argument("--classes", type=int, nargs="+", default=list(range(len(CLASS_NAMES))))
    parser.add_argument("--merge-person", action="store_true", help="count 'people' as 'pedestrian' everywhere")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--compare-ultralytics", metavar="DATA_YAML")
    args = parser.parse_args()

    detector = build_detector(args, conf=EVAL_CONF)
    image_paths = sorted((args.raw / "images").glob("*.jpg"))[: args.max_images]
    detector.detect(cv2.imread(str(image_paths[0])))  # warm-up: the first call is slow on any device
    preds, gts, elapsed = {}, {}, 0.0
    for path in image_paths:
        with Image.open(path) as im:
            w, h = im.size
        gts[path.stem] = parse_det_annotation(args.raw / "annotations" / f"{path.stem}.txt", w, h)
        image = cv2.imread(str(path))
        start = time.perf_counter()  # time the detection only, not disk reads
        preds[path.stem] = detector.detect(image)
        elapsed += time.perf_counter() - start

    classes = args.classes
    if args.merge_person:
        preds = {k: remap_classes(v, PERSON_MERGE) for k, v in preds.items()}
        gts = {k: remap_classes(v, PERSON_MERGE) for k, v in gts.items()}
        classes = sorted({PERSON_MERGE.get(c, c) for c in classes})

    m = evaluate_detections(preds, gts, classes)
    print(f"{len(image_paths)} images, {len(image_paths) / elapsed:.1f} images/s (detection only)")
    print(f"ours        mAP@50 {m.map50:.4f}   mAP@50-95 {m.map50_95:.4f}")
    for size, area_range in SIZE_RANGES.items():
        ms = evaluate_detections(preds, gts, classes, area_range=area_range)
        print(f"  {size:<7} objects   mAP@50 {ms.map50:.4f}   mAP@50-95 {ms.map50_95:.4f}")
    for c, ap in sorted(m.ap50_per_class.items()):
        name = "person (pedestrian+people)" if args.merge_person and c == 0 else CLASS_NAMES[c]
        print(f"  {name:<28} AP@50 {ap:.4f}")

    if args.compare_ultralytics:
        from ultralytics import YOLO

        r = YOLO(args.weights).val(
            data=args.compare_ultralytics, imgsz=args.imgsz, conf=EVAL_CONF, device=default_device(), verbose=False
        )
        print(f"ultralytics mAP@50 {r.box.map50:.4f}   mAP@50-95 {r.box.map:.4f}")


if __name__ == "__main__":
    main()
