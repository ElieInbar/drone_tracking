"""Fine-tune a YOLO model on VisDrone (YOLO format).

Usage:
  uv run scripts/train.py --data data/visdrone_yolo/data.yaml --model yolo26n.pt --epochs 50
  uv run scripts/train.py --resume runs/visdrone_y26n/weights/last.pt     # continue an interrupted run
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from dronetrack.detect.yolo import default_device


def train(
    data: str,
    model: str = "yolo26n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs",
    name: str = "visdrone_y26n",
    fraction: float = 1.0,
    device: str | None = None,
    resume: str | None = None,
) -> Path:
    """Train (or resume) and return the path of the best weights."""
    if resume:
        trainer_model = YOLO(resume)
        trainer_model.train(resume=True)
    else:
        trainer_model = YOLO(model)
        trainer_model.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=str(Path(project).resolve()),  # absolute, else Ultralytics nests it under runs/detect/
            name=name,
            fraction=fraction,  # share of the training images used, for quick trials
            device=device or default_device(),
            save_period=5,  # also keep a checkpoint every 5 epochs, in case the machine stops
            exist_ok=True,
        )
    return Path(trainer_model.trainer.best)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/visdrone_yolo/data.yaml")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="visdrone_y26n")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--device")
    parser.add_argument("--resume")
    args = parser.parse_args()
    best = train(**vars(args))
    print(f"best weights: {best}")


if __name__ == "__main__":
    main()
