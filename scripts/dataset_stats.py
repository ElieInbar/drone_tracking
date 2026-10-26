"""Print object counts per class and per size for one VisDrone-DET split.

Usage: uv run scripts/dataset_stats.py --raw data/VisDrone2019-DET-train
"""

import argparse
from pathlib import Path

from PIL import Image

from dronetrack.data.stats import SIZE_BUCKETS, dataset_stats
from dronetrack.data.visdrone import CLASS_NAMES, parse_det_annotation


def load_annotations(raw: Path):
    for image_path in sorted((raw / "images").glob("*.jpg")):
        with Image.open(image_path) as im:  # reads only the header, not the pixels
            w, h = im.size
        yield parse_det_annotation(raw / "annotations" / f"{image_path.stem}.txt", w, h)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    s = dataset_stats(load_annotations(args.raw))
    total = sum(s["per_class"].values())
    print(f"{total} objects, {s['objects_per_image_mean']:.1f} per image on average\n")
    print(f"{'class':<16}{'objects':>9}{'share':>8}")
    for cls, name in enumerate(CLASS_NAMES):
        n = s["per_class"].get(cls, 0)
        print(f"{name:<16}{n:>9}{n / total:>8.1%}")
    print(f"\n{'size':<16}{'objects':>9}{'share':>8}")
    for bucket in SIZE_BUCKETS:
        n = s["per_size"][bucket]
        print(f"{bucket:<16}{n:>9}{n / total:>8.1%}")


if __name__ == "__main__":
    main()
