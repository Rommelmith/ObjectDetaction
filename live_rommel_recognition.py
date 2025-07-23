import cv2, torch, numpy as np, time, threading
from collections import deque
from pathlib import Path
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import re

# ----------------- CONFIG -----------------
ENTER_THRESH = 0.70      # distance must fall BELOW to become ROMMEL
EXIT_THRESH  = 0.75      # distance must rise ABOVE to become UNKNOWN
SMOOTH_K = 5             # number of recent embeddings to smooth
DETECT_EVERY = 2         # run full detection every N frames
SPIKE_IGNORE_DIST = 0.90 # ignore isolated spikes above this
CAM_INDEX = 0            # webcam index
MIN_DET_PROB = 0.90
WINDOW_NAME = "Rommel Threaded"
# ------------------------------------------

device = torch.device("cpu")

# ---- Load centroid & threshold (we ignore threshold.txt now; use ENTER/EXIT) ----
centroid = np.load("rommel_centroid.npy")

# ---- Models ----
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ---- Shared State ----
latest_frame = None
frame_lock = threading.Lock()
running = True

# Detection / classification state
recent_embs = deque(maxlen=SMOOTH_K)
state = "UNKNOWN"
face_box = None  # (x1,y1,x2,y2)
frame_count = 0

def preprocess_crop(bgr, box):
    x1,y1,x2,y2 = map(int, box)
    h,w,_ = bgr.shape
    x1=max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((160,160))
    arr = np.float32(pil)/255.0
    mean, std = arr.mean(), arr.std()+1e-6
    arr = (arr-mean)/std
    arr = np.transpose(arr,(2,0,1))
    tensor = torch.from_numpy(arr).unsqueeze(0)
    return tensor

def classify(tensor):
    with torch.no_grad():
        emb = resnet(tensor).squeeze(0).cpu().numpy()
    recent_embs.append(emb)
    avg_emb = np.mean(recent_embs, axis=0)
    dist = np.linalg.norm(avg_emb - centroid)
    return dist

def update_state(dist):
    global state
    if state == "UNKNOWN":
        if dist < ENTER_THRESH:
            state = "ROMMEL"
            print(f"[EVENT] ENTER ROMMEL (smoothed dist={dist:.3f})")
    else:  # ROMMEL
        if dist > EXIT_THRESH:
            state = "UNKNOWN"
            print(f"[EVENT] EXIT ROMMEL (smoothed dist={dist:.3f})")

def capture_loop():
    global latest_frame, running
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Failed to open camera.")
        running = False
        return
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        with frame_lock:
            latest_frame = frame
        # small sleep helps release CPU a bit
        time.sleep(0.001)
    cap.release()

# Start capture thread
t = threading.Thread(target=capture_loop, daemon=True)
t.start()

prev_time = time.time()
fps = 0.0

try:
    while running:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Decide whether to detect
        do_detect = (frame_count % DETECT_EVERY == 1) or (face_box is None)

        if do_detect:
            # Full detection
            boxes, probs = mtcnn.detect(Image.fromarray(rgb))
            # Pick highest-prob single face for Rommel classification (multi-face extension later)
            best_box = None; best_prob = 0
            if boxes is not None:
                for b, p in zip(boxes, probs):
                    if p and p > MIN_DET_PROB and p > best_prob:
                        best_box, best_prob = b, p
            face_box = best_box
        # Else reuse previous face_box

        label = "Unknown"
        dist_display = None

        if face_box is not None:
            tensor = preprocess_crop(frame, face_box)
            if tensor is not None:
                dist = classify(tensor)
                dist_display = dist

                # Spike ignore logic
                if dist > SPIKE_IGNORE_DIST and len(recent_embs) >= 2:
                    # Check previous smoothed dist
                    temp = np.mean(list(recent_embs)[:-1], axis=0)
                    prev_dist = np.linalg.norm(temp - centroid)
                    if prev_dist < 0.80:
                        # Drop newest embedding as spike
                        recent_embs.pop()
                        dist = prev_dist
                    else:
                        # Keep it if previous already large
                        pass

                update_state(dist)
                label = "Rommel" if state == "ROMMEL" else "Unknown"

                # Draw box
                x1,y1,x2,y2 = map(int, face_box)
                color = (0,255,0) if label == "Rommel" else (0,0,255)
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                if dist_display is not None:
                    cv2.putText(frame, f"{label} {dist_display:.3f}",
                                (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            else:
                # Lost crop this frame; optionally clear after repeated fails
                pass

        # FPS
        now = time.time()
        dt = now - prev_time
        if dt > 0:
            fps = 1.0 / dt
        prev_time = now
        cv2.putText(frame, f"FPS:{fps:.1f}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255),2)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
        # Optional: reset history if no detection for a while
        # (not critical here)

    cv2.destroyAllWindows()

except KeyboardInterrupt:
    running = False
    cv2.destroyAllWindows()

