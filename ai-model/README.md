# Architecture Overview
```commandline
CCTV Video Frame → YOLOv12 → Weapon Detection Features
                   ↓
           CNN (ResNet) → Frame-level features
                   ↓
           LSTM → Temporal aggregation
                   ↓
        Concatenate with YOLO features
                   ↓
          Fully Connected → Class (Normal/Violence/Weaponized)

```

## ✅ Model 1 — Violence Action Detection

- Input: video clips

- Labels: normal / violence

- Architecture: CNN + LSTM

- Output: Violence probability per video

## ✅ Model 2 — Weapon Detection

- Input: video frames

- Labels: weapon bounding boxes

- Architecture: YOLOv12

- Output: Weapon presence + location