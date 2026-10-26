# Drone Tracking

Detect and track vehicles and pedestrians in drone footage: every object gets an identity
that it keeps from frame to frame. Built on the VisDrone dataset.

**Status: work in progress.**

- Detector: YOLO26 (Ultralytics), fine-tuned on VisDrone.
- Homemade parts: small-object tiling, multi-object tracker (Kalman + Hungarian, ByteTrack-style),
  evaluation metrics (mAP, MOTA, IDF1), each checked against the official tools.
