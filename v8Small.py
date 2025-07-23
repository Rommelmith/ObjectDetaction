import threading
import queue
import time
import cv2
import torch
from ultralytics import YOLO

# —————————————————————
# Configuration
# —————————————————————
CAM_INDEX = 0
IMG_SIZE = 640
QUEUE_MAX = 2

# Determine device
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

# —————————————————————
# Shared Structures
# —————————————————————
grab_queue   = queue.Queue(maxsize=QUEUE_MAX)
result_queue = queue.Queue(maxsize=QUEUE_MAX)
stop_event   = threading.Event()

# —————————————————————
# Load & Prep Model
# —————————————————————
model = YOLO('yolov8n.pt')      # load
model.fuse()                    # in‑place fuse conv+bn
model.model.half()              # in‑place to FP16
model.to(DEVICE)                # in‑place to chosen device

# —————————————————————
# Thread Functions
# —————————————————————
def grab_frames():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            grab_queue.put(frame, timeout=0.1)
        except queue.Full:
            pass
    cap.release()

def run_inference():
    while not stop_event.is_set():
        try:
            frame = grab_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        results = model(frame, imgsz=IMG_SIZE)[0]
        annotated = results.plot()
        try:
            result_queue.put(annotated, timeout=0.1)
        except queue.Full:
            pass

def display_and_control():
    prev_time = time.time()
    while not stop_event.is_set():
        try:
            annotated = result_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        cv2.putText(
            annotated, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow("Threaded YOLOv8 Live", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()

# —————————————————————
# Launch Threads
# —————————————————————
threads = [
    threading.Thread(target=grab_frames, daemon=True),
    threading.Thread(target=run_inference, daemon=True),
    threading.Thread(target=display_and_control)
]

for t in threads:
    t.start()
for t in threads:
    t.join()
