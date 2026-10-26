from dronetrack.data.visdrone import CLASS_NAMES, parse_det_annotation
from dronetrack.types import Detection


def test_ten_classes():
    assert len(CLASS_NAMES) == 10 and CLASS_NAMES[0] == "pedestrian" and CLASS_NAMES[9] == "motor"


def test_parse_keeps_valid_and_shifts_class(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("10,20,30,40,1,4,0,0\n")  # left, top, w, h, score, category 4 = car
    assert parse_det_annotation(p, 100, 100) == [Detection(10, 20, 40, 60, 1.0, 3)]


def test_parse_drops_ignored_others_and_score0(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0,0,5,5,0,0,0,0\n0,0,5,5,1,11,0,0\n0,0,5,5,0,4,0,0\n")
    assert parse_det_annotation(p, 100, 100) == []


def test_parse_clips_and_drops_degenerate(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("90,90,20,20,1,1,0,0\n50,50,0,10,1,1,0,0\n")
    assert parse_det_annotation(p, 100, 100) == [Detection(90, 90, 100, 100, 1.0, 0)]


def test_parse_tolerates_trailing_comma_and_blank_lines(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("10,20,30,40,1,4,0,0,\n\n")
    assert len(parse_det_annotation(p, 100, 100)) == 1
