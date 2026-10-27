import numpy as np

from dronetrack.detect.tiling import make_tiles


def test_visdrone_det_size_gives_4_by_3_tiles():
    tiles = make_tiles(2000, 1500)
    assert len(tiles) == 12
    assert sorted({t[0] for t in tiles}) == [0, 512, 1024, 1360]
    assert sorted({t[1] for t in tiles}) == [0, 512, 860]


def test_tiles_are_640_square_when_the_image_is_big_enough():
    assert all(x2 - x1 == 640 and y2 - y1 == 640 for x1, y1, x2, y2 in make_tiles(2000, 1500))


def test_tiles_cover_the_image_and_stay_inside():
    tiles = make_tiles(1344, 756)
    assert all(0 <= x1 < x2 <= 1344 and 0 <= y1 < y2 <= 756 for x1, y1, x2, y2 in tiles)
    covered = np.zeros((756, 1344), bool)
    for x1, y1, x2, y2 in tiles:
        covered[y1:y2, x1:x2] = True
    assert covered.all()


def test_image_smaller_than_tile_is_one_tile():
    assert make_tiles(480, 360) == [(0, 0, 480, 360)]


def test_no_duplicate_tile_when_size_is_a_multiple_of_the_stride():
    tiles = make_tiles(1152, 640)  # 1152 = 512 + 640: the second tile already touches the border
    assert len(tiles) == len(set(tiles)) == 2
