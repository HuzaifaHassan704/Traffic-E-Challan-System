"""
Live camera worker — connects to a real IP/CCTV camera (RTSP stream) or a
locally attached USB webcam, grabs a frame every N seconds, and runs it
through the same detect -> rules -> log pipeline used by the dashboard.

This runs as its own long-lived process, separate from the API server,
so you can run one worker per physical camera.

USAGE

  IP / CCTV camera (RTSP):
    python camera_worker.py --source "rtsp://user:pass@192.168.1.50:554/stream1" --camera-id 1 --interval 5

  Laptop / USB webcam (device index, usually 0 for the built-in camera):
    python camera_worker.py --source 0 --camera-id 1 --interval 5

  Video file (for testing without a real camera):
    python camera_worker.py --source path/to/video.mp4 --camera-id 1 --interval 2

Notes
-----
- Register the camera first via POST /cameras (or the dashboard) so you
  have a camera_id to pass in — this links captured violations back to
  a named camera + location.
- --interval controls how often a frame is pulled and analyzed; you don't
  need to run every single video frame through detection.
- For an RTSP URL, check your camera/DVR's manual for the exact URL
  format — it varies by manufacturer (Hikvision, Dahua, generic ONVIF,
  etc). It usually looks like:
    rtsp://<username>:<password>@<camera-ip>:554/<stream-path>
- If OpenCV can't open the stream, try installing ffmpeg, or confirm the
  camera's RTSP port (554 is standard) is reachable from this machine
  (same network / port forwarded).
"""
import argparse
import os
import sys
import time
import uuid

import cv2

from pipeline import process_frame_file, EVIDENCE_DIR


def main():
    parser = argparse.ArgumentParser(description="Live camera -> e-Challan pipeline worker")
    parser.add_argument("--source", required=True,
                         help="RTSP URL, video file path, or webcam device index (e.g. 0)")
    parser.add_argument("--camera-id", type=int, default=None,
                         help="ID of the camera as registered via /cameras (optional but recommended)")
    parser.add_argument("--interval", type=float, default=5.0,
                         help="Seconds between analyzed frames (default: 5)")
    parser.add_argument("--speed-kmh", type=float, default=None,
                         help="Fixed simulated speed to attach to every frame, for testing overspeeding rules")
    args = parser.parse_args()

    # Webcam device indices come in as strings from argparse; convert if numeric
    source = args.source
    if source.isdigit():
        source = int(source)

    print(f"[camera_worker] Connecting to source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[camera_worker] ERROR: could not open source '{source}'. "
              f"Check the RTSP URL/credentials, or that the webcam index is correct.")
        sys.exit(1)

    print(f"[camera_worker] Connected. Capturing a frame every {args.interval}s. Press Ctrl+C to stop.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[camera_worker] Warning: failed to read frame, retrying...")
                time.sleep(2)
                # Try to reconnect in case the stream dropped
                cap.release()
                cap = cv2.VideoCapture(source)
                continue

            frame_id = str(uuid.uuid4())[:8]
            raw_path = os.path.join(EVIDENCE_DIR, f"{frame_id}_raw.jpg")
            cv2.imwrite(raw_path, frame)

            result = process_frame_file(
                raw_path,
                camera_id=args.camera_id,
                speed_kmh=args.speed_kmh,
                wrong_direction=False,
                plate_number_manual=None,
            )

            n_det = len(result["detections"])
            n_viol = len(result["violations"])
            stamp = time.strftime("%H:%M:%S")
            if n_viol:
                print(f"[{stamp}] {n_det} object(s) detected — {n_viol} VIOLATION(S) flagged for review")
            else:
                print(f"[{stamp}] {n_det} object(s) detected — no violations")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[camera_worker] Stopped by user.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
