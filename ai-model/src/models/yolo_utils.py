# YOLOv12 inference helper (ready for weapon detection)

import cv2

def run_yolo(model, frame):
    """
    frame: numpy image (BGR)
    model: YOLOv12 model
    """
    results = model(frame)
    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append((cls, conf))

    return detections

#For weapon detection, a YOLO-based one-stage object detector pretrained on the
# COCO dataset was fine-tuned on annotated weapon frames. YOLO was selected due to its real-time performance
# and high localization accuracy.
