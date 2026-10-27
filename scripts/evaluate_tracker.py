"""Experiment 3: compare trackers on the same detections, over the VisDrone-MOT val sequences.

Detections are computed once per video (cached in outputs/dets/) and fed to every tracker, so only the
tracking algorithm differs. Scores: our MOTA / IDF1 on the 5 VisDrone-MOT classes, ignored regions applied.

Examples:
  uv run scripts/evaluate_tracker.py --weights models/visdrone_y26n.pt
  uv run scripts/evaluate_tracker.py --weights models/visdrone_y26n.pt --check-reference \
      --sequences uav0000086_00000_v uav0000268_05773_v
"""

import argparse
import time
from pathlib import Path

import numpy as np

from dronetrack.data.visdrone import EVAL_CLASSES, parse_mot_annotation
from dronetrack.detect.factory import add_detector_args, build_detector
from dronetrack.eval.mot_metrics import evaluate_tracking
from dronetrack.geometry import boxes_to_array, iou_matrix
from dronetrack.track.cache import detect_sequence
from dronetrack.track.factory import TRACK_CONF, TRACKERS, cache_path, make_tracker

CHECK_CLASS = 3  # car: the reference check uses one class, without ignored regions


def trackeval_scores(gt: dict, pred: dict, n_frames: int, cls: int) -> tuple[float, float]:
    """MOTA and IDF1 of one class computed by TrackEval (the reference implementation used in papers)."""
    # TrackEval still uses numpy aliases removed in numpy 1.24 (np.int is exactly the builtin int).
    for alias, builtin in (("int", int), ("float", float), ("bool", bool)):
        if not hasattr(np, alias):
            setattr(np, alias, builtin)
    from trackeval.metrics import CLEAR, Identity

    frames = range(1, n_frames + 1)
    gts = [[t for t in gt.get(f, []) if t.cls == cls] for f in frames]
    preds = [[t for t in pred.get(f, []) if t.cls == cls] for f in frames]
    gt_index = {i: k for k, i in enumerate(sorted({t.track_id for f in gts for t in f}))}  # TrackEval wants 0..n-1
    pred_index = {i: k for k, i in enumerate(sorted({t.track_id for f in preds for t in f}))}
    data = {
        "num_timesteps": n_frames,
        "num_gt_ids": len(gt_index),
        "num_tracker_ids": len(pred_index),
        "num_gt_dets": sum(map(len, gts)),
        "num_tracker_dets": sum(map(len, preds)),
        "gt_ids": [np.array([gt_index[t.track_id] for t in f], int) for f in gts],
        "tracker_ids": [np.array([pred_index[t.track_id] for t in f], int) for f in preds],
        "similarity_scores": [iou_matrix(boxes_to_array(g), boxes_to_array(p)) for g, p in zip(gts, preds)],
    }
    clear = CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}).eval_sequence(data)
    identity = Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False}).eval_sequence(data)
    return float(clear["MOTA"]), float(identity["IDF1"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_detector_args(parser)
    parser.add_argument("--root", type=Path, default=Path("data/VisDrone2019-MOT-val"))
    parser.add_argument("--sequences", nargs="+", help="default: all sequences of --root")
    parser.add_argument("--trackers", nargs="+", choices=TRACKERS, default=list(TRACKERS))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/dets"))
    parser.add_argument("--check-reference", action="store_true", help="also score with TrackEval (class car)")
    args = parser.parse_args()

    sequences = args.sequences or sorted(p.name for p in (args.root / "sequences").iterdir() if p.is_dir())
    detector = None
    totals = {name: dict(fn=0, fp=0, idsw=0, n_gt=0, n_pred=0, idtp=0, frames=0, seconds=0.0) for name in args.trackers}
    print(f"{'sequence':<20}{'tracker':<11}{'MOTA':>7}{'IDF1':>7}{'IDSW':>6}{'FP':>7}{'FN':>7}{'fps':>7}")
    for seq in sequences:
        cache = cache_path(args.cache_dir, Path(seq), args)
        if detector is None and not cache.exists():
            detector = build_detector(args, conf=TRACK_CONF)
        dets = detect_sequence(detector, args.root / "sequences" / seq, cache)
        gt, ignored = parse_mot_annotation(args.root / "annotations" / f"{seq}.txt")
        for name in args.trackers:
            tracker = make_tracker(name)
            start = time.perf_counter()
            pred = {i + 1: tracker.update(frame_dets) for i, frame_dets in enumerate(dets)}
            seconds = time.perf_counter() - start
            m = evaluate_tracking(gt, pred, EVAL_CLASSES, ignored=ignored)
            print(f"{seq:<20}{name:<11}{m.mota:>7.3f}{m.idf1:>7.3f}{m.idsw:>6}{m.fp:>7}{m.fn:>7}{len(dets) / seconds:>7.0f}")
            t = totals[name]
            for key in ("fn", "fp", "idsw", "n_gt", "n_pred", "idtp"):
                t[key] += getattr(m, key)
            t["frames"] += len(dets)
            t["seconds"] += seconds
            if args.check_reference:
                ours = evaluate_tracking(gt, pred, [CHECK_CLASS])
                ref_mota, ref_idf1 = trackeval_scores(gt, pred, len(dets), CHECK_CLASS)
                print(f"{'':<20}{'  car only':<11} ours MOTA {ours.mota:.4f} IDF1 {ours.idf1:.4f} | "
                      f"TrackEval MOTA {ref_mota:.4f} IDF1 {ref_idf1:.4f}")
    print("-" * 72)
    for name, t in totals.items():
        mota = 1 - (t["fn"] + t["fp"] + t["idsw"]) / t["n_gt"]
        idf1 = 2 * t["idtp"] / (t["n_gt"] + t["n_pred"])
        fps = t["frames"] / t["seconds"]
        print(f"{'ALL':<20}{name:<11}{mota:>7.3f}{idf1:>7.3f}{t['idsw']:>6}{t['fp']:>7}{t['fn']:>7}{fps:>7.0f}")


if __name__ == "__main__":
    main()
