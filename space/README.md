---
title: Drone Tracking
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
python_version: "3.11"
license: agpl-3.0
short_description: Detect and track vehicles and pedestrians in drone video
---

# Drone Tracking

Upload a drone video and get it back with every vehicle and pedestrian boxed and numbered: each object keeps its
number from frame to frame. Detector: YOLO26n fine-tuned on VisDrone. Tracker: a ByteTrack-style tracker (Kalman
filter + Hungarian matching) written from scratch, which beats Ultralytics' ByteTrack on the same detections.

Code, experiments and results: https://github.com/ElieInbar/drone_tracking

The example clip is the first 10 seconds of sequence `uav0000086_00000_v` of the
[VisDrone2019-MOT](https://github.com/VisDrone/VisDrone-Dataset) validation set (AISKYEYE team, Tianjin University),
shown here for non-commercial, research demonstration purposes.
