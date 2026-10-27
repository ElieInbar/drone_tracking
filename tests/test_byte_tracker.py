from dronetrack.track.byte_tracker import ByteTracker
from dronetrack.types import Detection, Track


def car(x, score=0.9):
    return Detection(x, 0, x + 20, 20, score, 3)


def run(frames, **kw):
    t = ByteTracker(**kw)
    return [t.update(f) for f in frames]


def ids(out):
    return {tr.track_id for f in out for tr in f}


def test_outputs_tracks():
    out = run([[car(0)]])
    assert isinstance(out[0][0], Track) and out[0][0].cls == 3 and out[0][0].score == 0.9


def test_moving_object_keeps_its_id():
    assert ids(run([[car(5 * i)] for i in range(10)])) == {1}


def test_object_reappearing_after_occlusion_keeps_its_id():
    frames = [[car(5 * i)] for i in range(10)] + [[] for _ in range(5)] + [[car(5 * i)] for i in range(15, 20)]
    out = run(frames)
    assert out[10] == [] and ids(out[15:]) == {1}


def test_object_lost_too_long_gets_a_new_id():
    frames = [[car(0)], [car(0)]] + [[] for _ in range(31)] + [[car(0)], [car(0)]]
    assert run(frames)[-1][0].track_id == 2


def test_low_score_detection_keeps_track_alive():
    frames = [[car(5 * i)] for i in range(5)] + [[car(25, score=0.15)]] + [[car(30)]]
    out = run(frames)
    assert len(out[5]) == 1 and out[6][0].track_id == 1


def test_low_score_detection_alone_never_starts_a_track():
    assert run([[car(0, score=0.15)]] * 3) == [[], [], []]


def test_crossing_objects_keep_their_ids():
    frames = [[car(10 * i), Detection(200 - 10 * i, 5, 220 - 10 * i, 25, 0.9, 3)] for i in range(21)]
    out = run(frames)
    first = {tr.track_id: tr.x1 for tr in out[0]}
    last = {tr.track_id: tr.x1 for tr in out[-1]}
    left_id = min(first, key=first.get)
    assert last[left_id] > 150  # the object that started on the left ended on the right


def test_new_track_needs_a_second_sighting_after_first_frame():
    out = run([[car(0)], [car(0), car(300)], [car(0), car(300)]])
    assert len(out[1]) == 1 and len(out[2]) == 2


def test_one_frame_false_alarm_never_gets_an_id():
    out = run([[car(0)], [car(0), car(300)], [car(0)], [car(0), car(600)], [car(0), car(600)]])
    assert ids(out) == {1, 2}  # the flash at x=300 was never confirmed, so the next object is n°2


def test_empty_frames_do_not_crash():
    assert run([[], [], []]) == [[], [], []]
