# Traffic e-Challan System (Prototype)

A working prototype of an AI-based traffic-camera violation detector and
e-challan (fine) generator, built and tested end-to-end.

**This is a real, running pipeline** — it uses an actual pretrained
YOLOv8 object-detection model (not a mock) to detect vehicles/people in
uploaded camera frames, checks them against rules you define, and issues
challans after human confirmation. It was tested with a real street photo
and correctly detected a bus, flagged it for a simulated overspeeding
rule, and generated a challan (`CH-20260809-23974`) — see the demo image
in this delivery.

## How it works

```
Camera frame  ->  YOLOv8 detection  ->  Rules engine (your rules)
                                              |
                                   violation found? --> queued for
                                              |          human review
                                              v
                                     reviewer confirms  -->  e-Challan
                                     or rejects              issued
```

- **Detection**: `backend/detector.py` runs a real pretrained YOLOv8
  model (COCO-trained) that recognizes `car`, `motorcycle`, `bus`,
  `truck`, `bicycle`, `person`.
- **Rules engine**: `backend/rules_engine.py` evaluates detections
  against rules stored in the database — **rules are data, not code**,
  so you add/edit them from the dashboard without touching Python.
  Supported rule types out of the box:
  - `restricted_vehicle_type` — a vehicle class isn't allowed in a zone
  - `restricted_zone_entry` — no vehicle allowed in a marked zone
  - `overspeeding` — speed (supplied per-frame) exceeds a limit
  - `wrong_direction` — flagged when marked as travelling the wrong way
- **Human-in-the-loop review**: every AI-flagged violation sits as
  `pending_review` until a person confirms or rejects it. Only confirmed
  violations become challans. This is intentional — auto-issuing legal
  fines straight from an ML model's output is a real liability and
  accuracy risk (see "Important limitations" below).
- **Database**: SQLite (`data/echallan.db`), auto-created on first run.
- **Evidence**: annotated frame (with bounding boxes) saved to
  `data/evidence/` and linked to each violation/challan.

## Project structure

```
traffic-echallan/
├── backend/
│   ├── main.py            FastAPI app / API endpoints
│   ├── detector.py        YOLOv8 detection wrapper
│   ├── rules_engine.py    Rule evaluation logic
│   ├── database.py        SQLite schema + default rules
│   ├── yolov8n.pt          Pretrained model weights (already downloaded)
│   └── requirements.txt
├── frontend/
│   └── index.html         Dashboard (capture sim, violations, challans, rules)
└── data/                  Created at runtime (db + evidence images)
```

## Running it

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open `frontend/index.html` in a browser (it talks to
`http://localhost:8000`). API docs are auto-generated at
`http://localhost:8000/docs`.

### Try it
1. Go to the **Rules Configuration** tab — three example rules are
   pre-seeded (heavy-vehicle zone, no-entry zone, overspeeding). Add
   your own.
2. Go to **Simulate Camera Capture**, upload a photo containing a
   vehicle, optionally enter a speed, and click **Run Detection & Rule
   Check**. You'll see real bounding boxes and any triggered violations.
3. Go to **Violations Review** and confirm or reject the flagged item.
4. Confirmed items appear in **Challans** with a generated challan
   number and unpaid status.

## Connecting real cameras

There are two different paths depending on the camera type:

### Laptop webcam or phone camera (browser-based)
Go to the **Live Camera** tab in the dashboard, click **Start Camera**,
allow the browser's camera permission prompt, then click **Capture &
Check Frame** (or **Start Auto-Capture** to poll every 5 seconds
automatically). This works for:
- A laptop's built-in or USB webcam.
- A phone's camera, if you open `frontend/index.html`'s URL on the
  phone's browser. To do that, your phone needs to reach the backend —
  either run the backend on a machine on the same Wi-Fi network and use
  its local IP (e.g. `http://192.168.1.20:8000`) instead of `localhost`
  in `frontend/index.html`'s `API` constant, or host the frontend
  somewhere reachable from the phone.
- **Note**: browsers only allow camera access on `https://` or
  `localhost` — a plain `http://<ip>` page from another device may be
  blocked. For local testing across devices you may need a tunnel
  (e.g. ngrok) or a self-signed cert.

### Real installed IP / CCTV camera (RTSP)
Browsers can't connect to RTSP streams directly, so this runs
server-side via `backend/camera_worker.py`, which was built and tested
as part of this delivery (verified against a synthetic test video —
see demo notes below):

```bash
cd backend
python camera_worker.py --source "rtsp://user:pass@192.168.1.50:554/stream1" --camera-id 1 --interval 5
```

- Register the camera first via the `/cameras` endpoint (or add a
  small form to the dashboard) so `--camera-id` refers to something
  meaningful.
- `--interval` controls how often a frame is pulled and analyzed —
  you don't need to run detection on every video frame.
- The exact RTSP URL format depends on your camera/DVR manufacturer
  (Hikvision, Dahua, generic ONVIF, etc.) — check its manual. It's
  usually `rtsp://<username>:<password>@<camera-ip>:554/<path>`.
- You can also point `--source` at a plain video file path to test the
  pipeline without a live camera, or at a webcam device index (e.g.
  `--source 0`) to run the same script against a USB webcam headlessly
  (no browser needed) — useful if the backend runs on a dedicated
  machine physically connected to the camera.
- This worker calls the exact same detection + rules + logging code as
  the dashboard upload path (`backend/pipeline.py`), so violations from
  a live camera show up in the **Violations Review** tab exactly like
  everything else.

## Important limitations (read before treating this as production-ready)

- **No number-plate OCR (ANPR)** — this needs a model fine-tuned on
  local plate fonts/formats. `detector.recognize_plate()` is a clearly
  marked stub; the dashboard lets you enter a plate manually for now.
- **No helmet-detection model** — same reason; `detector.has_helmet()`
  is a stub for a future specialized classifier.
- **Speed is supplied, not measured** — real speed estimation needs
  two timestamped frames plus camera calibration (known distance in the
  frame). The `overspeeding` rule accepts a `speed_kmh` value so you can
  plug in a real speed-estimation module later.
- **No owner lookup / SMS / payment gateway integration** — plate numbers
  aren't yet resolved to a vehicle owner, and no notification or payment
  provider is wired in.
- **Legal/regulatory**: issuing real traffic fines generally requires
  authorization from the local traffic police / transport authority.
  Treat this as a technical foundation to build with them, not a
  standalone product to deploy unilaterally.
- **Zone drawing**: rules use a full-frame zone by default in this
  prototype's UI. A production build would let you draw a precise
  polygon on a camera snapshot.

## Suggested next steps, in order

1. Point `detector.py` at an RTSP stream and sample frames on an interval.
2. Add a real ANPR model (e.g. a plate detector + OCR fine-tuned on
   local plates) in `recognize_plate()`.
3. Add owner lookup against a vehicle-registration database/API.
4. Add SMS/email notification on challan issuance.
5. Add a polygon-drawing tool in the dashboard for zone rules.
6. Add a payment gateway integration for challan settlement.
