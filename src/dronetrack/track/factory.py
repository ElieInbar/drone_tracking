"""Choose a tracker by name, and name the detection cache of a video, for the scripts."""

import argparse
from pathlib import Path

from dronetrack.track.byte_tracker import ByteTracker

TRACK_CONF = 0.1  # trackers need low-score boxes for ByteTrack's second pass
TRACKERS = ("ours", "bytetrack")


def make_tracker(name: str):
    if name == "ours":
        return ByteTracker()
    if name == "bytetrack":
        from dronetrack.track.reference import UltralyticsByteTrack

        return UltralyticsByteTrack()
    raise ValueError(f"unknown tracker {name!r}")


def cache_path(cache_dir: Path, source: Path, args: argparse.Namespace) -> Path:
    """One cache file per video and detector variant, e.g. outputs/dets/uav0000086_00000_v_visdrone_y26n.json."""
    variant = Path(args.weights).stem + ("_tiles" if args.tiles else "")
    return Path(cache_dir) / f"{Path(source).stem}_{variant}.json"
