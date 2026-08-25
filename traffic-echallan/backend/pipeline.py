"""
Core capture-processing pipeline, factored out so it can be called from:
  - main.py's /process-frame HTTP endpoint (browser uploads / webcam snapshots)
  - camera_worker.py (live RTSP camera polling)

Keeping this in one place means a live camera and a manually uploaded
photo go through the exact same detection + rules + logging code.
"""
import os
import json
import uuid
from datetime import datetime

import cv2

from database import get_conn
from detector import detect_objects
from rules_engine import evaluate_rules

BASE_DIR = os.path.dirname(__file__)
EVIDENCE_DIR = os.path.join(BASE_DIR, "..", "data", "evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)


def process_frame_file(raw_path, camera_id=None, speed_kmh=None, wrong_direction=False,
                        plate_number_manual=None):
    """
    Takes a path to an already-saved image file, runs it through the full
    pipeline (detect -> evaluate rules -> annotate -> log violations),
    and returns the same result shape used by the API.
    """
    frame_id = str(uuid.uuid4())[:8]

    detections, (w, h) = detect_objects(raw_path)

    conn = get_conn()
    rules = conn.execute("SELECT * FROM rules WHERE active=1").fetchall()

    extra_context = {"speed_kmh": speed_kmh, "wrong_direction": wrong_direction}
    violations = evaluate_rules(detections, rules, extra_context)

    # Draw annotated evidence image
    img = cv2.imread(raw_path)
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(img, f"{det['class']} {det['confidence']:.2f}", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
    for v in violations:
        x1, y1, x2, y2 = [int(c) for c in v["box"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img, "VIOLATION", (x1, max(y1 - 25, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    annotated_path = os.path.join(EVIDENCE_DIR, f"{frame_id}_annotated.jpg")
    cv2.imwrite(annotated_path, img)

    saved_violations = []
    for v in violations:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO violations (camera_id, rule_id, vehicle_class, plate_number, confidence,
                                     evidence_path, detected_boxes, status, created_at)
            VALUES (?,?,?,?,?,?,?, 'pending_review', ?)
        """, (camera_id, v["rule_id"], v["vehicle_class"], plate_number_manual, v["confidence"],
              f"evidence/{frame_id}_annotated.jpg", json.dumps(detections), datetime.utcnow().isoformat()))
        conn.commit()
        v_id = cur.lastrowid
        saved_violations.append({**v, "violation_id": v_id})

    conn.close()

    return {
        "frame_id": frame_id,
        "detections": detections,
        "violations": saved_violations,
        "evidence_image": f"/evidence/{frame_id}_annotated.jpg",
        "image_size": {"w": w, "h": h},
    }
