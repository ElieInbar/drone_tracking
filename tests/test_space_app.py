import importlib.util
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

_spec = importlib.util.spec_from_file_location("space_app", Path(__file__).parents[1] / "space" / "app.py")
app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app)


def test_ensure_weights_downloads_once(tmp_path):
    src = tmp_path / "src.pt"
    src.write_bytes(b"w")
    dst = tmp_path / "w.pt"
    assert app.ensure_weights(dst, src.as_uri()).read_bytes() == b"w"
    src.write_bytes(b"changed")
    assert app.ensure_weights(dst, src.as_uri()).read_bytes() == b"w"  # not downloaded again


def test_failed_download_leaves_no_weights_file(tmp_path):
    with pytest.raises(Exception):
        app.ensure_weights(tmp_path / "w.pt", (tmp_path / "missing.pt").as_uri())
    assert not (tmp_path / "w.pt").exists()


def test_run_without_video_shows_message():
    with pytest.raises(gr.Error, match="upload a video"):
        app.run(None)


def test_ui_builds():
    assert isinstance(app.build_ui(), gr.Blocks)
