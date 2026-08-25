"""
SQLite database layer for the e-Challan system.
Stores: rules (defined at setup time), cameras, violations, challans.
"""
import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "echallan.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            rule_type TEXT NOT NULL,        -- restricted_vehicle_type, restricted_zone_entry, overspeeding, wrong_direction
            description TEXT,
            applies_to TEXT,                -- JSON list of vehicle classes e.g. ["truck","bus"]
            zone_polygon TEXT,               -- JSON list of [x,y] points, normalized 0-1 (image coords)
            speed_limit_kmh REAL,            -- used for overspeeding rules
            fine_amount REAL NOT NULL DEFAULT 1000,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            location TEXT NOT NULL,
            rtsp_url TEXT,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER,
            rule_id INTEGER NOT NULL,
            vehicle_class TEXT,
            plate_number TEXT,
            confidence REAL,
            evidence_path TEXT,
            detected_boxes TEXT,             -- JSON of detected bounding boxes for debugging
            status TEXT NOT NULL DEFAULT 'pending_review',  -- pending_review, confirmed, rejected
            created_at TEXT NOT NULL,
            FOREIGN KEY(rule_id) REFERENCES rules(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS challans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            challan_no TEXT UNIQUE NOT NULL,
            violation_id INTEGER NOT NULL,
            plate_number TEXT,
            fine_amount REAL NOT NULL,
            payment_status TEXT NOT NULL DEFAULT 'unpaid',  -- unpaid, paid, disputed
            issued_at TEXT NOT NULL,
            FOREIGN KEY(violation_id) REFERENCES violations(id)
        )
    """)

    conn.commit()

    # Seed default rules only if table is empty
    cur.execute("SELECT COUNT(*) as c FROM rules")
    if cur.fetchone()["c"] == 0:
        default_rules = [
            {
                "name": "No Heavy Vehicles in Restricted Zone",
                "rule_type": "restricted_vehicle_type",
                "description": "Trucks and buses are not allowed to enter the marked zone during restricted hours.",
                "applies_to": json.dumps(["truck", "bus"]),
                "zone_polygon": json.dumps([[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]]),
                "speed_limit_kmh": None,
                "fine_amount": 2000,
            },
            {
                "name": "No Entry Zone Violation",
                "rule_type": "restricted_zone_entry",
                "description": "No vehicle of any kind may enter the marked no-entry zone.",
                "applies_to": json.dumps(["car", "motorcycle", "truck", "bus"]),
                "zone_polygon": json.dumps([[0.3, 0.0], [0.7, 0.0], [0.7, 0.3], [0.3, 0.3]]),
                "speed_limit_kmh": None,
                "fine_amount": 1500,
            },
            {
                "name": "Overspeeding",
                "rule_type": "overspeeding",
                "description": "Vehicle speed exceeds the posted speed limit, calculated from two timestamped frames.",
                "applies_to": json.dumps(["car", "motorcycle", "truck", "bus"]),
                "zone_polygon": None,
                "speed_limit_kmh": 60,
                "fine_amount": 3000,
            },
        ]
        for r in default_rules:
            cur.execute("""
                INSERT INTO rules (name, rule_type, description, applies_to, zone_polygon,
                                    speed_limit_kmh, fine_amount, active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
            """, (r["name"], r["rule_type"], r["description"], r["applies_to"], r["zone_polygon"],
                  r["speed_limit_kmh"], r["fine_amount"], datetime.utcnow().isoformat()))
        conn.commit()

    conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized at", DB_PATH)
