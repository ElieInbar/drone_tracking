from dronetrack.types import Detection, Track


def test_detection_size():
    d = Detection(10, 20, 40, 80, 0.9, 3)
    assert (d.width, d.height, d.area) == (30, 60, 1800)


def test_track_is_a_detection_with_an_id():
    t = Track(0, 0, 10, 10, 0.8, 1, track_id=7)
    assert isinstance(t, Detection) and t.track_id == 7
