"""
Vehicle detection using a real pretrained YOLOv8 model (COCO classes).
Detects: person, bicycle, car, motorcycle, bus, truck.

Note on scope: number-plate OCR and helmet-classification require
specially trained models (local plate fonts / helmet datasets) that
aren't available in this environment. This module exposes clean
interfaces (recognize_plate, has_helmet) with clearly marked stub
implementations so a real model can be dropped in later without
changing the rest of the pipeline.
"""
import os
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8n.pt")

# Only classes relevant to traffic enforcement
VEHICLE_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

_model = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def detect_objects(image_path, conf_threshold=0.35):
    """
    Runs YOLO detection on an image.
    Returns a list of dicts: {class, confidence, box: [x1,y1,x2,y2], box_norm: [x1,y1,x2,y2] (0-1)}
    """
    model = get_model()
    results = model(image_path, conf=conf_threshold, verbose=False)[0]

    img_h, img_w = results.orig_shape
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        if cls_name not in VEHICLE_CLASSES:
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        detections.append({
            "class": cls_name,
            "confidence": round(conf, 3),
            "box": [x1, y1, x2, y2],
            "box_norm": [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h],
        })
    return detections, (img_w, img_h)


def recognize_plate(image_path, box):
    """
    STUB: Number-plate OCR.
    A production system needs a plate-detector + OCR fine-tuned on local
    plate formats (e.g. Tesseract/EasyOCR trained/fine-tuned on regional
    plates, run on the cropped `box` region of `image_path`).
    Returns None here to signal "not available" rather than a fake value.
    """
    return None


def has_helmet(image_path, box):
    """
    STUB: Helmet presence classifier for motorcycle riders.
    Needs a model trained on a labeled helmet/no-helmet dataset.
    Returns None to signal "unknown" rather than guessing.
    """
    return None
