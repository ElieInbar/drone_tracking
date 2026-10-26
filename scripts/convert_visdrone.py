"""Convert VisDrone-DET train and val to YOLO format and write data.yaml.

Usage: uv run scripts/convert_visdrone.py --raw data --out data/visdrone_yolo
"""

import argparse
from pathlib import Path

from dronetrack.data.convert import convert_split, write_data_yaml

SPLITS = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("data/visdrone_yolo"))
    args = parser.parse_args()
    for split, folder in SPLITS.items():
        n = convert_split(args.raw / folder, args.out / split)
        print(f"{split}: {n} images")
    write_data_yaml(args.out, args.out / "data.yaml")
    print(f"wrote {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
