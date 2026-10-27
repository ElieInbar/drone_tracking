import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.slow
def test_wheel_contains_every_module(tmp_path):
    """The online demo installs the package from GitHub: a module left out of the wheel crashes it at startup."""
    subprocess.run(["uv", "build", "--wheel", "-o", str(tmp_path), "-q"], cwd=ROOT, check=True)
    (wheel,) = tmp_path.glob("*.whl")
    packaged = {n for n in zipfile.ZipFile(wheel).namelist() if n.endswith(".py")}
    source = {p.relative_to(ROOT / "src").as_posix() for p in (ROOT / "src" / "dronetrack").rglob("*.py")}
    assert source - packaged == set()
