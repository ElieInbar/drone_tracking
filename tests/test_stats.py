from dronetrack.data.stats import dataset_stats, size_bucket
from dronetrack.types import Detection


def test_size_buckets_coco_thresholds():
    assert size_bucket(Detection(0, 0, 31, 31, 1, 0)) == "small"
    assert size_bucket(Detection(0, 0, 32, 32, 1, 0)) == "medium"
    assert size_bucket(Detection(0, 0, 96, 96, 1, 0)) == "large"


def test_dataset_stats_counts():
    s = dataset_stats([[Detection(0, 0, 10, 10, 1, 3), Detection(0, 0, 100, 100, 1, 3)], []])
    assert s["per_class"] == {3: 2}
    assert s["per_size"] == {"small": 1, "medium": 0, "large": 1}
    assert s["objects_per_image_mean"] == 1.0


def test_dataset_stats_empty():
    s = dataset_stats([])
    assert s["per_class"] == {} and s["objects_per_image_mean"] == 0.0
