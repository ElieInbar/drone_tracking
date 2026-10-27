"""Dataset statistics: how many objects, of which class, and how small."""

from collections import Counter
from collections.abc import Iterable

from dronetrack.types import Detection

SIZE_BUCKETS = ("small", "medium", "large")


def size_bucket(det: Detection) -> str:
    """COCO size convention: small < 32x32 px of area, medium < 96x96, large otherwise."""
    if det.area < 32**2:
        return "small"
    if det.area < 96**2:
        return "medium"
    return "large"


def dataset_stats(annotations: Iterable[list[Detection]]) -> dict:
    """Count objects per class and per size bucket over a list of per-image annotations."""
    per_class: Counter[int] = Counter()
    per_size = dict.fromkeys(SIZE_BUCKETS, 0)
    n_images = n_objects = 0
    for dets in annotations:
        n_images += 1
        n_objects += len(dets)
        for d in dets:
            per_class[d.cls] += 1
            per_size[size_bucket(d)] += 1
    return {
        "per_class": dict(per_class),
        "per_size": per_size,
        "objects_per_image_mean": n_objects / n_images if n_images else 0.0,
    }
