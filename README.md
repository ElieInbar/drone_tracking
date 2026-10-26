# Drone Tracking

Detect and track vehicles and pedestrians in drone footage: every object gets an identity that it
keeps from frame to frame. Built on the [VisDrone](https://github.com/VisDrone/VisDrone-Dataset)
dataset, where 60% of the objects are smaller than 32×32 pixels.

| Off-the-shelf YOLO26n (COCO) | YOLO26n fine-tuned on VisDrone |
|---|---|
| ![before](docs/media/before_coco.gif) | ![after](docs/media/after_finetuned.gif) |
| 15 detections per frame: distant people are missed | 30 detections per frame: distant and seated people found |

## Results so far

**Experiment 1: fine-tuning on drone imagery.** VisDrone-DET val (548 images, 38,759 objects),
6 classes shared with COCO (pedestrian and people merged into "person" so both models are judged fairly).

| | Off-the-shelf (COCO) | Fine-tuned | Gain |
|---|---|---|---|
| mAP@50 | 0.144 | **0.343** | ×2.4 |
| mAP@50-95 | 0.083 | **0.192** | ×2.3 |

Largest gains on objects COCO rarely sees from above: motorcycles 0.05 → 0.34, trucks 0.08 → 0.25.
Fine-tuning: 50 epochs in 43 minutes on one L4 GPU ([Modal](https://modal.com)).
Full tables and analysis: [docs/RESULTATS.md](docs/RESULTATS.md) (French).

**The mAP is computed by our own code** (COCO method, [det_metrics.py](src/dronetrack/eval/det_metrics.py)),
unit-tested on hand-checkable cases and cross-checked against Ultralytics: on the same predictions the two
agree within 0.002.

## Roadmap

- [x] Step 0: pretrained YOLO on VisDrone videos (baseline)
- [x] Step 1: VisDrone → YOLO conversion, dataset statistics
- [x] Step 2: fine-tuning on a cloud GPU, homemade mAP, **experiment 1**
- [ ] Step 3: homemade tiling for small objects: accuracy vs speed (**experiment 2**)
- [ ] Step 4: homemade multi-object tracker (Kalman filter + Hungarian matching, ByteTrack-style),
      MOTA / IDF1, compared with official ByteTrack (**experiment 3**)
- [ ] Step 5: online demo
- [ ] Next: counter-drone variant (detect and track drones in the sky), ONNX / C++ speed benchmark

## Design

Each stage does one thing and they all exchange the same two objects: `Detection` (a box, a score,
a class) and `Track` (a detection plus a persistent identity). The tracker never knows which detector
produced its boxes, so the detector or the dataset can be swapped without touching the rest.

```
video → frames → detect → list[Detection] → track → list[Track] → draw (annotated video)
                                                              └→ evaluate (mAP, MOTA, IDF1)
```

| Part | Choice | Why |
|---|---|---|
| Detector | Ultralytics YOLO26n, fine-tuned | Rewriting YOLO adds nothing |
| Small-object tiling | Homemade (SAHI-like) | To understand and measure the trade-off |
| Tracker | Homemade, ByteTrack-style | Core algorithmic part |
| Metrics | Homemade mAP, MOTA, IDF1 | Each checked against the official tool |

Every technical choice and its reason: [docs/DECISIONS.md](docs/DECISIONS.md).
Glossary of the terms used: [docs/GLOSSAIRE.md](docs/GLOSSAIRE.md) (French).

## Run it

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest                                              # unit tests
uv run scripts/download_visdrone.py det-val                # 82 MB
uv run scripts/evaluate_detector.py --weights yolo26n.pt --coco --classes 0 2 3 5 8 9 --merge-person
uv run scripts/run_detect.py --source path/to/video.mp4 --weights yolo26n.pt --coco --gif out.gif
```

Fine-tuning on Modal: `modal run --detach scripts/modal_train.py` (see the script's header).

## Credits and licenses

- Data: VisDrone (AISKYEYE team, Tianjin University), for non-commercial academic use.
- Detector: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0), which requires code built on it to stay open source.
