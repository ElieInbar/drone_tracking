"""Detection quality: AP per class and mAP, computed the COCO way."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from dronetrack.geometry import boxes_to_array, iou_matrix
from dronetrack.types import Detection

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)  # 0.50, 0.55, ..., 0.95
RECALL_POINTS = np.linspace(0, 1, 101)  # COCO reads precision at recall 0.00, 0.01, ..., 1.00


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


def _match_image(preds: list[Detection], gts: list[Detection], thresholds: np.ndarray) -> np.ndarray:
    """Greedy matching in one image and one class: (n_preds, n_thresholds) True where the prediction is a hit.

    Predictions are visited by decreasing score; each takes the free ground truth it overlaps most,
    if that overlap reaches the threshold. A ground truth can be taken only once (a duplicate is a miss).
    """
    hits = np.zeros((len(preds), len(thresholds)), dtype=bool)
    if not preds or not gts:
        return hits
    ious = iou_matrix(boxes_to_array(preds), boxes_to_array(gts))
    taken = np.zeros((len(thresholds), len(gts)), dtype=bool)
    for p in np.argsort([-d.score for d in preds], kind="stable"):
        for t, threshold in enumerate(thresholds):
            candidates = np.where(taken[t], -1.0, ious[p])
            g = int(np.argmax(candidates))
            if candidates[g] >= threshold:
                taken[t, g] = True
                hits[p, t] = True
    return hits


def evaluate_detections(
    preds: Mapping[str, list[Detection]],
    gts: Mapping[str, list[Detection]],
    classes: Sequence[int],
    iou_thresholds: np.ndarray = IOU_THRESHOLDS,
) -> DetMetrics:
    """mAP@50 and mAP@50-95 over `classes`; dict keys are image names. Classes with no ground truth are skipped."""
    ap = {}  # class -> AP at each IoU threshold
    for c in classes:
        scores, hits, n_gt = [], [], 0
        for image in gts.keys() | preds.keys():
            image_gts = [d for d in gts.get(image, []) if d.cls == c]
            image_preds = [d for d in preds.get(image, []) if d.cls == c]
            n_gt += len(image_gts)
            scores += [d.score for d in image_preds]
            hits.append(_match_image(image_preds, image_gts, iou_thresholds))
        if n_gt == 0:
            continue
        order = np.argsort(-np.array(scores), kind="stable")
        hits_sorted = np.concatenate(hits)[order] if scores else np.zeros((0, len(iou_thresholds)), bool)
        tp = np.cumsum(hits_sorted, axis=0)  # true positives so far, per threshold
        n_seen = np.arange(1, len(order) + 1)[:, None]
        recall, precision = tp / n_gt, tp / n_seen
        ap[c] = np.array([average_precision(recall[:, t], precision[:, t]) for t in range(len(iou_thresholds))])
    if not ap:
        return DetMetrics(0.0, 0.0, {})
    per_class = np.stack(list(ap.values()))
    return DetMetrics(
        map50=float(per_class[:, 0].mean()),
        map50_95=float(per_class.mean()),
        ap50_per_class={c: float(v[0]) for c, v in ap.items()},
    )
