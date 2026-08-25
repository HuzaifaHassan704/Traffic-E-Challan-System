"""
Rules engine: evaluates YOLO detections against the rules stored in the
database (defined at setup / via the dashboard) and returns violations.

Rules are DATA, not code -- adding a new rule of an existing rule_type
requires no code change, only a new row in the `rules` table.
"""
import json


def point_in_polygon(x, y, polygon):
    """Ray-casting point-in-polygon test. polygon: list of [x,y] normalized points."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def box_center_norm(box_norm):
    x1, y1, x2, y2 = box_norm
    return (x1 + x2) / 2, (y1 + y2) / 2


def evaluate_rules(detections, rules, extra_context=None):
    """
    detections: list from detector.detect_objects()
    rules: list of sqlite3.Row from the `rules` table (active ones)
    extra_context: optional dict, e.g. {"speed_kmh": 78} supplied by the
                    caller (a real system would derive this from two
                    timestamped frames + camera calibration).

    Returns a list of violation dicts:
        {rule_id, rule_name, vehicle_class, confidence, box, fine_amount, reason}
    """
    extra_context = extra_context or {}
    violations = []

    for rule in rules:
        applies_to = json.loads(rule["applies_to"]) if rule["applies_to"] else []
        rule_type = rule["rule_type"]

        for det in detections:
            if det["class"] not in applies_to:
                continue

            if rule_type == "restricted_vehicle_type":
                # Zone entry AND wrong vehicle type
                polygon = json.loads(rule["zone_polygon"]) if rule["zone_polygon"] else None
                if polygon:
                    cx, cy = box_center_norm(det["box_norm"])
                    if point_in_polygon(cx, cy, polygon):
                        violations.append(_make_violation(rule, det,
                            f"{det['class']} detected inside restricted zone (not permitted here)"))
                else:
                    violations.append(_make_violation(rule, det,
                        f"{det['class']} detected — restricted vehicle type"))

            elif rule_type == "restricted_zone_entry":
                polygon = json.loads(rule["zone_polygon"]) if rule["zone_polygon"] else None
                if polygon:
                    cx, cy = box_center_norm(det["box_norm"])
                    if point_in_polygon(cx, cy, polygon):
                        violations.append(_make_violation(rule, det,
                            "Vehicle entered a no-entry zone"))

            elif rule_type == "overspeeding":
                speed = extra_context.get("speed_kmh")
                limit = rule["speed_limit_kmh"]
                if speed is not None and limit is not None and speed > limit:
                    violations.append(_make_violation(rule, det,
                        f"Recorded speed {speed} km/h exceeds limit of {limit} km/h"))

            elif rule_type == "wrong_direction":
                if extra_context.get("wrong_direction"):
                    violations.append(_make_violation(rule, det,
                        "Vehicle detected travelling against permitted direction"))

    return violations


def _make_violation(rule, det, reason):
    return {
        "rule_id": rule["id"],
        "rule_name": rule["name"],
        "vehicle_class": det["class"],
        "confidence": det["confidence"],
        "box": det["box"],
        "fine_amount": rule["fine_amount"],
        "reason": reason,
    }
