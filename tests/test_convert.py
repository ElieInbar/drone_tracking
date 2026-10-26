import cv2
import numpy as np
import yaml

from dronetrack.data.convert import convert_split, to_yolo_line, write_data_yaml
from dronetrack.types import Detection


def test_to_yolo_line():
    assert to_yolo_line(Detection(0, 0, 50, 100, 1.0, 3), 100, 200) == "3 0.250000 0.250000 0.500000 0.500000"


def _make_split(root, annotation_text):
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir()
    cv2.imwrite(str(root / "images" / "x.jpg"), np.zeros((100, 200, 3), np.uint8))
    (root / "annotations" / "x.txt").write_text(annotation_text)


def test_convert_split_writes_yolo_labels(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_split(src, "0,0,50,100,1,4,0,0\n")  # image is 200 wide, 100 high
    assert convert_split(src, dst) == 1
    assert (dst / "labels" / "x.txt").read_text() == "3 0.125000 0.500000 0.250000 1.000000\n"


def test_convert_split_writes_empty_label_for_empty_image(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_split(src, "")
    assert convert_split(src, dst) == 1
    assert (dst / "labels" / "x.txt").read_text() == ""
    assert (dst / "images" / "x.jpg").exists()


def test_write_data_yaml(tmp_path):
    write_data_yaml(tmp_path, tmp_path / "data.yaml")
    cfg = yaml.safe_load((tmp_path / "data.yaml").read_text())
    assert cfg["train"] == "train/images" and cfg["val"] == "val/images"
    assert cfg["names"][0] == "pedestrian" and len(cfg["names"]) == 10
