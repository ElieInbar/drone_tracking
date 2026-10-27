"""Tracking quality: MOTA (CLEAR MOT) and IDF1, class by class then summed."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from dronetrack.geometry import boxes_to_array, ios_matrix, iou_matrix
from dronetrack.types import Detection, Track

IGNORED_MIN_IOS = 0.5  # a prediction lying at least half inside an ignored region is not scored


@dataclass
class MotMetrics:
    mota: float  # 1 - (misses + false alarms + identity switches) / ground-truth objects
    idf1: float  # share of objects that carry the right identity, over the whole video
    idsw: int  # identity switches
    fp: int  # false alarms
    fn: int  # misses
    n_gt: int  # ground-truth boxes (sum over frames)
    n_pred: int  # predicted boxes that were scored (not dropped in ignored regions)
    idtp: int  # boxes with the right identity (IDF1 numerator / 2)


def _hungarian(score: np.ndarray, allowed: np.ndarray) -> list[tuple[int, int]]:
    """Pairs maximising the total score, among allowed pairs only."""
    if score.size == 0:
        return []
    rows, cols = linear_sum_assignment(np.where(allowed, -score, 1e6))
    return [(int(r), int(c)) for r, c in zip(rows, cols) if allowed[r, c]]


def _evaluate_class(gt, pred, ignored, cls: int, min_iou: float) -> tuple[int, int, int, int, int, int]:
    """(fn, fp, idsw, n_gt, n_pred_scored, idtp) for one class."""
    fn = fp = idsw = n_gt = n_pred = 0
    last_pred_of: dict[int, int] = {}  # ground-truth id -> predicted id it was last matched with
    overlap_frames: dict[tuple[int, int], int] = defaultdict(int)  # (gt id, pred id) -> frames with IoU >= min_iou
    for frame in sorted(gt.keys() | pred.keys()):
        gts = [t for t in gt.get(frame, []) if t.cls == cls]
        preds = [t for t in pred.get(frame, []) if t.cls == cls]
        iou = iou_matrix(boxes_to_array(gts), boxes_to_array(preds))
        ok = iou >= min_iou

        # CLEAR MOT, step 1: keep last frame's pairs that still overlap (a track keeps its object).
        matches = []
        for g, gt_box in enumerate(gts):
            p_id = last_pred_of.get(gt_box.track_id)
            for p, pred_box in enumerate(preds):
                if pred_box.track_id == p_id and ok[g, p] and p not in {m[1] for m in matches}:
                    matches.append((g, p))  # a prediction can follow only one object
                    break
        free_g = [g for g in range(len(gts)) if g not in {m[0] for m in matches}]
        free_p = [p for p in range(len(preds)) if p not in {m[1] for m in matches}]
        # Step 2: Hungarian matching of everything else on IoU.
        sub = iou[np.ix_(free_g, free_p)]
        matches += [(free_g[i], free_p[j]) for i, j in _hungarian(sub, sub >= min_iou)]

        for g, p in matches:
            gt_id, p_id = gts[g].track_id, preds[p].track_id
            if gt_id in last_pred_of and last_pred_of[gt_id] != p_id:
                idsw += 1  # this object was followed under another number before
            last_pred_of[gt_id] = p_id

        # Unmatched predictions inside an ignored region (unannotated crowd) are dropped, not false alarms.
        matched_p = {p for _, p in matches}
        dropped = set()
        if ignored and ignored.get(frame):
            ios = ios_matrix(boxes_to_array(preds), boxes_to_array(ignored[frame]))
            dropped = {p for p in range(len(preds)) if p not in matched_p and (ios[p] >= IGNORED_MIN_IOS).any()}
        fp += len(preds) - len(matched_p) - len(dropped)
        fn += len(gts) - len(matches)
        n_gt += len(gts)
        n_pred += len(preds) - len(dropped)
        for g in range(len(gts)):
            for p in range(len(preds)):
                if ok[g, p]:
                    overlap_frames[gts[g].track_id, preds[p].track_id] += 1

    # IDF1: pair each real identity with at most one predicted identity, maximising frames in common.
    gt_ids = sorted({g for g, _ in overlap_frames})
    pred_ids = sorted({p for _, p in overlap_frames})
    counts = np.zeros((len(gt_ids), len(pred_ids)))
    for (g, p), n in overlap_frames.items():
        counts[gt_ids.index(g), pred_ids.index(p)] = n
    idtp = int(sum(counts[i, j] for i, j in _hungarian(counts, counts > 0)))
    return fn, fp, idsw, n_gt, n_pred, idtp


def evaluate_tracking(
    gt: Mapping[int, list[Track]],
    pred: Mapping[int, list[Track]],
    classes: Sequence[int],
    ignored: Mapping[int, list[Detection]] | None = None,
    min_iou: float = 0.5,
) -> MotMetrics:
    """MOTA and IDF1 summed over `classes`; dict keys are frame numbers."""
    fn = fp = idsw = n_gt = n_pred = idtp = 0
    for cls in classes:
        c_fn, c_fp, c_idsw, c_gt, c_pred, c_idtp = _evaluate_class(gt, pred, ignored, cls, min_iou)
        fn, fp, idsw, n_gt, n_pred, idtp = fn + c_fn, fp + c_fp, idsw + c_idsw, n_gt + c_gt, n_pred + c_pred, idtp + c_idtp
    mota = 1 - (fn + fp + idsw) / n_gt if n_gt else 0.0
    idf1 = 2 * idtp / (n_gt + n_pred) if n_gt + n_pred else 0.0
    return MotMetrics(mota=mota, idf1=idf1, idsw=idsw, fp=fp, fn=fn, n_gt=n_gt, n_pred=n_pred, idtp=idtp)
