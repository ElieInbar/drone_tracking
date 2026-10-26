"""Download and unzip VisDrone splits into data/.

Usage: uv run scripts/download_visdrone.py det-train det-val mot-val [--out data]
"""

import argparse
import urllib.request
import zipfile
from pathlib import Path

import gdown

ULTRALYTICS = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
SPLITS = {
    # name: (folder created by the zip, how to download)
    "det-train": ("VisDrone2019-DET-train", f"{ULTRALYTICS}/VisDrone2019-DET-train.zip"),  # 1.55 GB
    "det-val": ("VisDrone2019-DET-val", f"{ULTRALYTICS}/VisDrone2019-DET-val.zip"),  # 82 MB
    "mot-val": ("VisDrone2019-MOT-val", "gdrive:1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu"),  # 1.48 GB
}


def download(split: str, out: Path) -> Path:
    folder, source = SPLITS[split]
    target = out / folder
    if target.exists():
        print(f"{split}: already in {target}, skipped")
        return target
    out.mkdir(parents=True, exist_ok=True)
    archive = out / f"{folder}.zip"
    if archive.exists():
        print(f"{split}: using {archive} already on disk (e.g. downloaded by hand in a browser)")
    elif source.startswith("gdrive:"):
        print(f"{split}: downloading {source}")
        gdown.download(id=source.removeprefix("gdrive:"), output=str(archive), quiet=False)
    else:
        print(f"{split}: downloading {source}")
        urllib.request.urlretrieve(source, archive)
    print(f"{split}: unzipping")
    with zipfile.ZipFile(archive) as z:
        z.extractall(out)
    archive.unlink()  # free disk space once extracted
    if not target.exists():
        raise RuntimeError(f"{split}: expected folder {target} after unzip, check the archive layout")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("splits", nargs="+", choices=sorted(SPLITS))
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()
    for split in args.splits:
        print(f"{split}: ready in {download(split, args.out)}")


if __name__ == "__main__":
    main()
