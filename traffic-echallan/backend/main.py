"""
Traffic e-Challan System — FastAPI backend.

Pipeline: camera frame upload -> YOLO vehicle detection -> rules engine
(rules defined by the user at setup time, editable via /rules) ->
violation logged for human review -> on confirmation, an e-challan
(fine notice) is auto-generated.

Run with:  uvicorn main:app --reload --port 8000
"""
import os
import json
import shutil
import uuid
import random
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from database import init_db, get_conn
from pipeline import process_frame_file, EVIDENCE_DIR

app = FastAPI(title="Traffic e-Challan System")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/evidence", StaticFiles(directory=EVIDENCE_DIR), name="evidence")

init_db()


# ---------- Pydantic schemas ----------

class RuleIn(BaseModel):
    name: str
    rule_type: str  # restricted_vehicle_type | restricted_zone_entry | overspeeding | wrong_direction
    description: Optional[str] = ""
    applies_to: List[str] = []
    zone_polygon: Optional[List[List[float]]] = None  # normalized [x,y] points
    speed_limit_kmh: Optional[float] = None
    fine_amount: float = 1000
    active: bool = True


class CameraIn(BaseModel):
    name: str
    location: str
    rtsp_url: Optional[str] = None
    active: bool = True


class ViolationDecision(BaseModel):
    approve: bool


# ---------- Rules CRUD ----------

@app.get("/rules")
def list_rules():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM rules ORDER BY id DESC").fetchall()
    conn.close()
    return [_rule_to_dict(r) for r in rows]


@app.post("/rules")
def create_rule(rule: RuleIn):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO rules (name, rule_type, description, applies_to, zone_polygon,
                            speed_limit_kmh, fine_amount, active, created_at)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (rule.name, rule.rule_type, rule.description, json.dumps(rule.applies_to),
          json.dumps(rule.zone_polygon) if rule.zone_polygon else None,
          rule.speed_limit_kmh, rule.fine_amount, int(rule.active),
          datetime.utcnow().isoformat()))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return {"id": new_id, "status": "created"}


@app.put("/rules/{rule_id}")
def update_rule(rule_id: int, rule: RuleIn):
    conn = get_conn()
    existing = conn.execute("SELECT id FROM rules WHERE id=?", (rule_id,)).fetchone()
    if not existing:
        conn.close()
        raise HTTPException(404, "Rule not found")
    conn.execute("""
        UPDATE rules SET name=?, rule_type=?, description=?, applies_to=?, zone_polygon=?,
                          speed_limit_kmh=?, fine_amount=?, active=? WHERE id=?
    """, (rule.name, rule.rule_type, rule.description, json.dumps(rule.applies_to),
          json.dumps(rule.zone_polygon) if rule.zone_polygon else None,
          rule.speed_limit_kmh, rule.fine_amount, int(rule.active), rule_id))
    conn.commit()
    conn.close()
    return {"status": "updated"}


@app.delete("/rules/{rule_id}")
def delete_rule(rule_id: int):
    conn = get_conn()
    conn.execute("DELETE FROM rules WHERE id=?", (rule_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted"}


# ---------- Cameras CRUD ----------

@app.get("/cameras")
def list_cameras():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM cameras ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/cameras")
def create_camera(cam: CameraIn):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO cameras (name, location, rtsp_url, active, created_at) VALUES (?,?,?,?,?)",
                (cam.name, cam.location, cam.rtsp_url, int(cam.active), datetime.utcnow().isoformat()))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return {"id": new_id, "status": "created"}


# ---------- Core pipeline: process a frame ----------

@app.post("/process-frame")
async def process_frame(
    file: UploadFile = File(...),
    camera_id: Optional[int] = Form(None),
    speed_kmh: Optional[float] = Form(None),
    wrong_direction: Optional[bool] = Form(False),
    plate_number_manual: Optional[str] = Form(None),
):
    """
    Simulates a camera capture being processed:
    1. Save the uploaded frame
    2. Run YOLO detection
    3. Evaluate active rules against detections
    4. Draw evidence image with boxes
    5. Log each violation as 'pending_review' (human-in-the-loop)
    """
    frame_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    raw_path = os.path.join(EVIDENCE_DIR, f"{frame_id}_raw{ext}")
    with open(raw_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return process_frame_file(raw_path, camera_id=camera_id, speed_kmh=speed_kmh,
                               wrong_direction=wrong_direction, plate_number_manual=plate_number_manual)


# ---------- Violations review + challan issuance ----------

@app.get("/violations")
def list_violations(status: Optional[str] = None):
    conn = get_conn()
    q = """SELECT v.*, r.name as rule_name, r.fine_amount as fine_amount
           FROM violations v JOIN rules r ON v.rule_id = r.id"""
    params = ()
    if status:
        q += " WHERE v.status=?"
        params = (status,)
    q += " ORDER BY v.id DESC"
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/violations/{violation_id}/decide")
def decide_violation(violation_id: int, decision: ViolationDecision):
    conn = get_conn()
    v = conn.execute("""SELECT v.*, r.fine_amount as fine_amount FROM violations v
                         JOIN rules r ON v.rule_id = r.id WHERE v.id=?""", (violation_id,)).fetchone()
    if not v:
        conn.close()
        raise HTTPException(404, "Violation not found")

    if decision.approve:
        conn.execute("UPDATE violations SET status='confirmed' WHERE id=?", (violation_id,))
        challan_no = f"CH-{datetime.utcnow().strftime('%Y%m%d')}-{random.randint(10000,99999)}"
        conn.execute("""
            INSERT INTO challans (challan_no, violation_id, plate_number, fine_amount, payment_status, issued_at)
            VALUES (?,?,?,?, 'unpaid', ?)
        """, (challan_no, violation_id, v["plate_number"], v["fine_amount"], datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
        return {"status": "confirmed", "challan_no": challan_no}
    else:
        conn.execute("UPDATE violations SET status='rejected' WHERE id=?", (violation_id,))
        conn.commit()
        conn.close()
        return {"status": "rejected"}


# ---------- Challans ----------

@app.get("/challans")
def list_challans():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM challans ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/challans/{challan_id}/mark-paid")
def mark_paid(challan_id: int):
    conn = get_conn()
    conn.execute("UPDATE challans SET payment_status='paid' WHERE id=?", (challan_id,))
    conn.commit()
    conn.close()
    return {"status": "paid"}


# ---------- helpers ----------

def _rule_to_dict(row):
    d = dict(row)
    d["applies_to"] = json.loads(d["applies_to"]) if d["applies_to"] else []
    d["zone_polygon"] = json.loads(d["zone_polygon"]) if d["zone_polygon"] else None
    return d


@app.get("/")
def root():
    return {"status": "ok", "message": "Traffic e-Challan API running. See /docs for API, or open frontend/index.html"}
