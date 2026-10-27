import pytest

from dronetrack.types import Detection, Track


def car(x, score=0.9):
    return Detection(x, 0, x + 20, 20, score, 3)


@pytest.mark.slow
def test_reference_tracker_follows_a_moving_car():
    from dronetrack.track.reference import UltralyticsByteTrack

    t = UltralyticsByteTrack()
    out = [t.update([car(5 * i)]) for i in range(10)]
    assert all(isinstance(tr, Track) and tr.cls == 3 for f in out for tr in f)
    assert {tr.track_id for f in out for tr in f} == {1}
    assert out[-1][0].x1 == pytest.approx(45, abs=2)


@pytest.mark.slow
def test_reference_tracker_handles_empty_frames():
    from dronetrack.track.reference import UltralyticsByteTrack

    t = UltralyticsByteTrack()
    assert [t.update([]) for _ in range(3)] == [[], [], []]
