"""Cut a large image into overlapping square tiles, so small objects keep their full resolution."""


def _starts(length: int, tile: int, stride: int) -> list[int]:
    """Start positions along one axis: every `stride` pixels, plus one last tile flush with the border."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)  # last tile ends exactly on the border, so nothing is left out
    return starts


def make_tiles(img_w: int, img_h: int, tile: int = 640, overlap: float = 0.2) -> list[tuple[int, int, int, int]]:
    """Tiles (x1, y1, x2, y2) in pixels covering the whole image, neighbours overlapping by `overlap`."""
    stride = int(tile * (1 - overlap))  # 640 * 0.8 = 512: each tile shares 128 px with the next one
    return [
        (x, y, min(x + tile, img_w), min(y + tile, img_h))
        for y in _starts(img_h, tile, stride)
        for x in _starts(img_w, tile, stride)
    ]
