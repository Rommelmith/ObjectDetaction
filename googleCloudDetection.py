"""
HYBRID REALTIME DETECTION
-------------------------
Fast local YOLO (if available) + periodic Google Cloud Vision object localization.
Falls back to Cloud-only if torch/ultralytics not installed or fail.

Press 'q' to quit.

Prereqs (full hybrid):
    pip install ultralytics
    pip install torch torchvision torchaudio (see instructions)
    pip install opencv-python google-cloud-vision

Set GOOGLE_APPLICATION_CREDENTIALS environment variable to your JSON key.

Tested on Python 3.10/3.11 Windows 64-bit. For 3.12 you may hit torch wheel issues.
"""


import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\romme\OneDrive\Documents\Generative Language Client (1).json"

import time
import threading
import queue
import cv2
import numpy as np

# ========= CONFIG =========
FRAME_W, FRAME_H       = 640, 480
YOLO_MODEL_PATH        = "yolov8n.pt"   # or yolo11n.pt if you have it
YOLO_CONF              = 0.30
YOLO_IMG_SIZE          = 640
CLOUD_INTERVAL_SEC     = 2.5            # attempt Cloud Vision this often
JPEG_QUALITY           = 55
CLOUD_RESCALE          = (640, 480)     # w,h frame sent to cloud
MAX_CLOUD_QUEUE        = 1              # keep only newest request
DRAW_THICKNESS         = 2
FONT_SCALE             = 0.5
SHOW_PERF_OVERLAY      = True
WINDOW_NAME            = "Hybrid (YOLO + Cloud Vision Fallback)"
# ==========================

# ----- Try YOLO Import -----
have_yolo = False
yolo_model = None
yolo_names = {}
yolo_device = "cpu"
try:
    from ultralytics import YOLO
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        yolo_names = yolo_model.names
        have_yolo = True
        # Detect GPU
        try:
            import torch
            if torch.cuda.is_available():
                yolo_device = "cuda"
        except Exception:
            pass
        print(f"[INFO] YOLO loaded ({YOLO_MODEL_PATH}) on {yolo_device}.")
    except Exception as e:
        print("[WARN] YOLO model load failed, continuing Cloud-only:", e)
except ImportError:
    print("[WARN] ultralytics not installed; running Cloud-only mode.")

# ----- Cloud Vision Client -----
try:
    from google.cloud import vision
    cloud_client = vision.ImageAnnotatorClient()
except Exception as e:
    raise RuntimeError(f"Failed to init Google Cloud Vision client: {e}")

# ----- Camera -----
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, 30)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# ----- Shared State -----
cloud_q = queue.Queue(MAX_CLOUD_QUEUE)
cloud_lock = threading.Lock()
cloud_objects = []
cloud_timestamp = 0.0

perf_lock = threading.Lock()
perf = {
    "frames": 0,
    "start": time.time(),
    "yolo_ms": 0.0,
    "cloud_calls": 0,
    "last_cloud_latency_ms": 0.0
}

stop_flag = False
last_cloud_attempt = 0.0

def cloud_worker():
    global cloud_objects, cloud_timestamp
    while not stop_flag:
        try:
            frame = cloud_q.get(timeout=0.2)
        except queue.Empty:
            continue
        if frame is None:
            break
        t0 = time.perf_counter()
        ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            cloud_q.task_done()
            continue
        image = vision.Image(content=jpg.tobytes())
        try:
            resp = cloud_client.object_localization(image=image)
            if resp.error.message:
                print("Cloud Vision error:", resp.error.message)
            else:
                with cloud_lock:
                    cloud_objects = resp.localized_object_annotations
                    cloud_timestamp = time.time()
                with perf_lock:
                    perf["cloud_calls"] += 1
        except Exception as e:
            print("Cloud Vision exception:", e)
        latency_ms = (time.perf_counter() - t0) * 1000
        with perf_lock:
            perf["last_cloud_latency_ms"] = latency_ms
        cloud_q.task_done()

worker_thread = threading.Thread(target=cloud_worker, daemon=True)
worker_thread.start()

def draw_cloud(frame):
    with cloud_lock:
        objs = list(cloud_objects)
        ts = cloud_timestamp
    h, w = frame.shape[:2]
    for obj in objs:
        verts = [(int(v.x * w), int(v.y * h)) for v in obj.bounding_poly.normalized_vertices]
        if len(verts) >= 2:
            cv2.polylines(frame, [np.array(verts, np.int32)], True, (0, 200, 255), DRAW_THICKNESS)
            cv2.putText(frame, f"[C]{obj.name}", verts[0],
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE + 0.1, (0, 200, 255), 2, cv2.LINE_AA)
    return ts

print("[INFO] Starting loop. YOLO:", have_yolo, "| Cloud Vision enabled.")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t_yolo0 = time.perf_counter()
        if have_yolo:
            # YOLO predict
            try:
                res = yolo_model.predict(frame, imgsz=YOLO_IMG_SIZE, conf=YOLO_CONF,
                                         verbose=False, device=yolo_device)[0]
                for box, cls in zip(res.boxes.xyxy, res.boxes.cls):
                    x1, y1, x2, y2 = map(int, box)
                    label = yolo_names.get(int(cls), str(int(cls)))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, label, (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0,255,0), 2, cv2.LINE_AA)
            except Exception as e:
                print("[WARN] YOLO inference error; disabling YOLO:", e)
                have_yolo = False
        t_yolo1 = time.perf_counter()

        # Periodically enqueue Cloud frame
        now = time.time()
        if now - last_cloud_attempt >= CLOUD_INTERVAL_SEC:
            if cloud_q.empty():
                send_frame = cv2.resize(frame, CLOUD_RESCALE)
                # copy to avoid race if frame reused
                cloud_q.put(send_frame.copy())
            last_cloud_attempt = now

        ts = draw_cloud(frame)

        # Perf accounting
        with perf_lock:
            perf["frames"] += 1
            perf["yolo_ms"] = (t_yolo1 - t_yolo0) * 1000

        if SHOW_PERF_OVERLAY:
            with perf_lock:
                elapsed = now - perf["start"]
                fps = perf["frames"] / elapsed if elapsed > 0 else 0.0
                yolo_ms = perf["yolo_ms"]
                cloud_calls = perf["cloud_calls"]
                cloud_lat = perf["last_cloud_latency_ms"]
            overlay_lines = [
                f"FPS: {fps:.1f}",
                f"YOLO: {yolo_ms:.1f} ms" if have_yolo else "YOLO: OFF",
                f"CloudCalls: {cloud_calls}",
                f"LastCloud: {cloud_lat:.0f} ms" if cloud_calls else "LastCloud: -",
                f"CloudAge: {now - ts:.1f}s" if ts else "CloudAge: -"
            ]
            y0 = 18
            for line in overlay_lines:
                cv2.putText(frame, line, (8, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,50,50), 3, cv2.LINE_AA)
                cv2.putText(frame, line, (8, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                y0 += 17

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Signal worker to stop
    stop_flag = True
    cloud_q.put(None)
    worker_thread.join(timeout=1.5)
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Clean shutdown.")
