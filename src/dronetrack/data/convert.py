"""Convert VisDrone-DET annotations to the YOLO label format Ultralytics trains on."""

from pathlib import Path

import cv2
import yaml

from dronetrack.data.visdrone import CLASS_NAMES, parse_det_annotation
from dronetrack.types import Detection


def to_yolo_line(det: Detection, img_w: int, img_h: int) -> str:
    """YOLO label line: 'class center_x center_y width height', all divided by the image size."""
    cx = (det.x1 + det.x2) / 2 / img_w
    cy = (det.y1 + det.y2) / 2 / img_h
    return f"{det.cls} {cx:.6f} {cy:.6f} {det.width / img_w:.6f} {det.height / img_h:.6f}"


def convert_split(src: Path, dst: Path) -> int:
    """src/{images,annotations} (VisDrone) -> dst/images (symlinks) + dst/labels/*.txt. Returns the image count."""
    src, dst = Path(src).resolve(), Path(dst)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path in sorted((src / "images").glob("*.jpg")):
        h, w = cv2.imread(str(image_path)).shape[:2]
        dets = parse_det_annotation(src / "annotations" / f"{image_path.stem}.txt", w, h)
        # An image without objects still gets a (empty) label file: YOLO learns it as background.
        lines = "".join(to_yolo_line(d, w, h) + "\n" for d in dets)
        (dst / "labels" / f"{image_path.stem}.txt").write_text(lines)
        link = dst / "images" / image_path.name
        if not link.exists():
            link.symlink_to(image_path)  # avoid copying 1.6 GB of images
        count += 1
    return count


def write_data_yaml(root: Path, out: Path) -> None:
    """Write the dataset description file Ultralytics reads (paths and class names)."""
    cfg = {
        "path": str(Path(root).resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": dict(enumerate(CLASS_NAMES)),
    }
    Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False))
