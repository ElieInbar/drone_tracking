import cv2
import numpy as np

from dronetrack.io import iter_frames


def test_iter_frames_folder_sorted_and_filtered(tmp_path):
    for name, value in [("0000002.jpg", 2), ("0000001.jpg", 1)]:
        cv2.imwrite(str(tmp_path / name), np.full((4, 4, 3), value, np.uint8))
    (tmp_path / ".DS_Store").write_text("x")
    frames = list(iter_frames(tmp_path))
    assert [int(f[0, 0, 0]) for f in frames] == [1, 2]


def test_iter_frames_video_file(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
    for _ in range(3):
        writer.write(np.zeros((32, 32, 3), np.uint8))
    writer.release()
    frames = list(iter_frames(path))
    assert len(frames) == 3 and frames[0].shape == (32, 32, 3)
