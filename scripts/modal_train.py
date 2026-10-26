"""Fine-tune YOLO on VisDrone on a Modal cloud GPU.

The raw VisDrone images and every training run live in the Modal volume "visdrone", so an
interrupted run can be resumed and the weights downloaded afterwards.

Usage (from the repo root):
  modal run --detach scripts/modal_train.py                      # full run: 50 epochs, YOLO26n, 640 px
  modal run scripts/modal_train.py --epochs 1 --fraction 0.02 --name smoke   # quick check (~cents)
  modal run --detach scripts/modal_train.py --resume             # continue an interrupted run
  modal volume get visdrone runs/visdrone_y26n/weights/best.pt models/visdrone_y26n.pt
"""

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
VOLUME_DIR = "/vol"
GPU = "L4"  # 24 GB, about 0.80 USD/hour on Modal

volume = modal.Volume.from_name("visdrone", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")  # needed by OpenCV
    .pip_install("ultralytics", "numpy", "opencv-python-headless", "imageio", "gdown", "pyyaml")
    .env({"PYTHONPATH": "/root/src:/root/scripts"})
    .add_local_dir(REPO / "src", "/root/src")
    .add_local_dir(REPO / "scripts", "/root/scripts")
)
app = modal.App("drone-tracking-train", image=image)


@app.function(volumes={VOLUME_DIR: volume}, timeout=3600)
def prepare_data() -> None:
    """Download VisDrone-DET train and val into the volume, once."""
    from download_visdrone import download

    for split in ("det-train", "det-val"):
        download(split, Path(VOLUME_DIR) / "raw")
    volume.commit()


@app.function(volumes={VOLUME_DIR: volume}, gpu=GPU, timeout=6 * 3600)
def train_remote(model: str, epochs: int, imgsz: int, batch: int, name: str, fraction: float, resume: bool) -> str:
    """Convert the dataset onto the container's local disk (fast reads), then train; runs are saved in the volume."""
    from dronetrack.data.convert import convert_split, write_data_yaml
    from train import train

    local = Path("/tmp/visdrone_yolo")
    raw = Path(VOLUME_DIR) / "raw"
    convert_split(raw / "VisDrone2019-DET-train", local / "train")
    convert_split(raw / "VisDrone2019-DET-val", local / "val")
    write_data_yaml(local, local / "data.yaml")

    runs = Path(VOLUME_DIR) / "runs"
    last = runs / name / "weights" / "last.pt"
    best = train(
        data=str(local / "data.yaml"),
        model=model,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(runs),
        name=name,
        fraction=fraction,
        device="0",
        resume=str(last) if resume else None,
    )
    volume.commit()
    # Path inside the volume (Modal resolves /vol to an internal path, so rebuild it rather than relativize)
    return f"runs/{name}/weights/{best.name}"


@app.local_entrypoint()
def main(
    model: str = "yolo26n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    name: str = "visdrone_y26n",
    fraction: float = 1.0,
    resume: bool = False,
) -> None:
    prepare_data.remote()
    best = train_remote.remote(model, epochs, imgsz, batch, name, fraction, resume)
    print(f"done. download with: modal volume get visdrone {best} models/{name}.pt")
