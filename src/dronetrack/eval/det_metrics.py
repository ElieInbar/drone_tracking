"""Detection quality: AP per class and mAP, computed the COCO way."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from dronetrack.geometry import boxes_to_array, iou_matrix
from dronetrack.types import Detection

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)  # 0.50, 0.55, ..., 0.95
RECALL_POINTS = np.linspace(0, 1, 101)  # COCO reads precision at recall 0.00, 0.01, ..., 1.00
# COCO object sizes, by box area in pixels: small < 32x32 <= medium < 96x96 <= large
SIZE_RANGES: dict[str, tuple[float, float]] = {
    "small": (0, 32**2),
    "medium": (32**2, 96**2),
    "large": (96**2, float("inf")),
}
# "people" (sitting, crouching) counted as "pedestrian": COCO-trained models only know "person"
PERSON_MERGE: dict[int, int] = {1: 0}


def remap_classes(dets: Sequence[Detection], mapping: Mapping[int, int]) -> list[Detection]:
    """Return copies of `dets` whose class is renamed by `mapping` (classes not in it are kept)."""
    return [replace(d, cls=mapping.get(d.cls, d.cls)) for d in dets]


@dataclass
class DetMetrics:
    map50: float  # mean over classes of AP at IoU 0.5
    map50_95: float  # mean over classes and over the 10 IoU thresholds
    ap50_per_class: dict[int, float]


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the precision-recall curve, sampled at 101 recall points (COCO method).

    `recall` and `precision` are the running values after each detection, in decreasing score order.
    """
    if len(recall) == 0:  # no prediction at all for this class
        return 0.0
    # Envelope: at each point, the best precision reachable at this recall or any higher one.
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    # For each recall point, first detection index where that recall is reached (len = never reached).
    idx = np.searchsorted(recall, RECALL_POINTS, side="left")
    sampled = np.where(idx < len(envelope), envelope[np.minimum(idx, len(envelope) - 1)], 0.0)
    return float(sampled.mean())


def _in_range(dets: Sequence[Detection], area_range: tuple[float, float] | None) -> np.ndarray:
    """True for each detection whose area lies in [low, high); all True when there is no range."""
    if area_range is None:
        return np.ones(len(dets), dtype=bool)
    low, high = area_range
    return np.array([low <= d.area < high for d in dets], dtype=bool)


def _match_image(
    preds: list[Detection],
    gts: list[Detection],
    thresholds: np.ndarray,
    area_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy matching in one image and one class. Returns (hits, ignored), each (n_preds, n_thresholds).

    Predictions are visited by decreasing score; each takes the free ground truth it overlaps most,
    if that overlap reaches the threshold. A ground truth can be taken only once (a duplicate is a miss).
    With an area range (COCO "small/medium/large"), ground truths outside it are "ignored": a prediction
    may still take one, but is then ignored (neither hit nor false positive), as is an unmatched
    prediction outside the range. A ground truth inside the range is always preferred.
    """
    hits = np.zeros((len(preds), len(thresholds)), dtype=bool)
    ignored = np.zeros_like(hits)
    pred_in_range = _in_range(preds, area_range)
    if not preds:
        return hits, ignored
    if not gts:
        ignored[~pred_in_range] = True
        return hits, ignored
    gt_ignored = ~_in_range(gts, area_range)
    ious = iou_matrix(boxes_to_array(preds), boxes_to_array(gts))
    taken = np.zeros((len(thresholds), len(gts)), dtype=bool)
    for p in np.argsort([-d.score for d in preds], kind="stable"):
        for t, threshold in enumerate(thresholds):
            matched = False
            for allow_ignored in (False, True):  # first try ground truths that count
                usable = ~taken[t] & (gt_ignored == allow_ignored)
                candidates = np.where(usable, ious[p], -1.0)
                g = int(np.argmax(candidates))
                if candidates[g] >= threshold:
                    taken[t, g] = True
                    hits[p, t] = not allow_ignored
                    ignored[p, t] = allow_ignored
                    matched = True
                    break
            if not matched and not pred_in_range[p]:
                ignored[p, t] = True
    return hits, ignored


def evaluate_detections(
    preds: Mapping[str, list[Detection]],
    gts: Mapping[str, list[Detection]],
    classes: Sequence[int],
    iou_thresholds: np.ndarray = IOU_THRESHOLDS,
    area_range: tuple[float, float] | None = None,
) -> DetMetrics:
    """mAP@50 and mAP@50-95 over `classes`; dict keys are image names.

    With `area_range` (see SIZE_RANGES), only objects of that size count. Classes with no ground truth
    (in range) are skipped.
    """
    n_thr = len(iou_thresholds)
    ap = {}  # class -> AP at each IoU threshold
    for c in classes:
        scores, hits, ignored, n_gt = [], [], [], 0
        for image in gts.keys() | preds.keys():
            image_gts = [d for d in gts.get(image, []) if d.cls == c]
            image_preds = [d for d in preds.get(image, []) if d.cls == c]
            n_gt += int(_in_range(image_gts, area_range).sum())
            scores += [d.score for d in image_preds]
            h, i = _match_image(image_preds, image_gts, iou_thresholds, area_range)
            hits.append(h)
            ignored.append(i)
        if n_gt == 0:
            continue
        order = np.argsort(-np.array(scores), kind="stable")
        hits_sorted = np.concatenate(hits)[order] if scores else np.zeros((0, n_thr), bool)
        ignored_sorted = np.concatenate(ignored)[order] if scores else np.zeros((0, n_thr), bool)
        ap_t = []
        for t in range(n_thr):
            counted = hits_sorted[~ignored_sorted[:, t], t]  # ignored predictions leave the ranking
            tp = np.cumsum(counted)  # true positives so far
            n_seen = np.arange(1, len(counted) + 1)
            ap_t.append(average_precision(tp / n_gt, tp / n_seen))
        ap[c] = np.array(ap_t)
    if not ap:
        return DetMetrics(0.0, 0.0, {})
    per_class = np.stack(list(ap.values()))
    return DetMetrics(
        map50=float(per_class[:, 0].mean()),
        map50_95=float(per_class.mean()),
        ap50_per_class={c: float(v[0]) for c, v in ap.items()},
    )
