from dronetrack.data.visdrone import CLASS_NAMES, EVAL_CLASSES, parse_det_annotation, parse_mot_annotation
from dronetrack.types import Detection, Track


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


def test_eval_classes_are_the_five_visdrone_mot_classes():
    assert [CLASS_NAMES[c] for c in EVAL_CLASSES] == ["pedestrian", "car", "van", "truck", "bus"]


def test_parse_mot_groups_by_frame_and_keeps_ids(tmp_path):
    p = tmp_path / "seq.txt"
    p.write_text("1,7,10,20,30,40,1,4,0,0\n2,7,12,20,30,40,1,4,0,0\n1,8,0,0,5,5,1,1,0,0\n")
    gt, ignored = parse_mot_annotation(p)
    assert gt[1] == [Track(10, 20, 40, 60, 1.0, 3, track_id=7), Track(0, 0, 5, 5, 1.0, 0, track_id=8)]
    assert gt[2][0].x1 == 12 and ignored == {}


def test_parse_mot_separates_ignored_regions_and_drops_others(tmp_path):
    p = tmp_path / "seq.txt"
    p.write_text("1,0,0,0,50,50,1,0,0,0\n1,3,0,0,5,5,1,11,0,0\n1,4,0,0,5,5,0,4,0,0\n")
    gt, ignored = parse_mot_annotation(p)
    assert gt == {} and ignored[1] == [Detection(0, 0, 50, 50, 1.0, -1)]


def test_parse_mot_drops_degenerate_boxes(tmp_path):
    p = tmp_path / "seq.txt"
    p.write_text("1,7,10,20,0,40,1,4,0,0\n")
    assert parse_mot_annotation(p) == ({}, {})
