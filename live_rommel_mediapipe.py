import cv2, time, threading, torch, numpy as np
from collections import deque
from facenet_pytorch import InceptionResnetV1
import mediapipe as mp
from pathlib import Path

# ================== CONFIG (TUNE HERE) ==================
CAM_INDEX        = 0
DETECT_INTERVAL  = 3        # Run detection every N frames (increase for speed, decrease for stability)
DETECT_WIDTH     = 640      # Downscale width for detection; keep >=480 for quality
SMOOTH_K         = 7        # Number of recent embeddings averaged (temporal smoothing)
ENTER_THRESH     = 0.74     # Distance to enter "ROMMEL"
EXIT_THRESH      = 0.82     # Distance to exit "ROMMEL" (must be > ENTER_THRESH)
MIN_HOLD_SECONDS = 1.0      # Minimum time to stay in ROMMEL before an EXIT is allowed
HARD_EXIT_DIST   = 0.95     # Immediate exit if smoothed distance exceeds this
SPIKE_IGNORE_DIST= 0.90     # Ignore a single spike above this if previous was stable
MIN_SCORE        = 0.60     # MediaPipe detection score threshold
EXPAND_FACTOR    = 0.20     # Expand face box by this fraction (improves alignment stability)
SKIP_EMBED_ODD   = False    # If True: skip embedding on odd frames (higher FPS, tiny delay)
WINDOW_NAME      = "Rommel MediaPipe"
# ========================================================

device = torch.device("cpu")

# ---- Load centroid (after pruning) ----
centroid = np.load("rommel_centroid.npy")

# ---- Embedding model ----
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ---- MediaPipe detector ----
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=MIN_SCORE)

# ---- Capture thread shared state ----
latest_frame = None
frame_lock = threading.Lock()
running = True

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
        time.sleep(0.001)
    cap.release()

cap_thread = threading.Thread(target=capture_loop, daemon=True)
cap_thread.start()

recent_embs = deque(maxlen=SMOOTH_K)
state = "UNKNOWN"
enter_time = 0.0
frame_count = 0
face_box = None  # last (x1,y1,x2,y2)
prev_time = time.time()

def expand_box(box, expand_factor, frame_shape):
    x1,y1,x2,y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w/2
    cy = y1 + h/2
    new_w = w * (1 + expand_factor)
    new_h = h * (1 + expand_factor)
    nx1 = int(cx - new_w/2)
    ny1 = int(cy - new_h/2)
    nx2 = int(cx + new_w/2)
    ny2 = int(cy + new_h/2)
    H, W = frame_shape[:2]
    nx1 = max(0, nx1); ny1 = max(0, ny1)
    nx2 = min(W, nx2); ny2 = min(H, ny2)
    return (nx1, ny1, nx2, ny2)

def preprocess_crop(bgr, box):
    x1,y1,x2,y2 = map(int, box)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (160,160), interpolation=cv2.INTER_AREA)
    arr = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    m, s = arr.mean(), arr.std() + 1e-6
    arr = (arr - m)/s
    arr = np.transpose(arr, (2,0,1))
    return torch.from_numpy(arr).unsqueeze(0)

def classify(tensor):
    with torch.no_grad():
        emb = resnet(tensor).squeeze(0).cpu().numpy()
    recent_embs.append(emb)
    avg_emb = np.mean(recent_embs, axis=0)
    dist = np.linalg.norm(avg_emb - centroid)
    return dist

def update_state(dist):
    global state, enter_time
    now = time.time()
    if state == "UNKNOWN":
        if dist < ENTER_THRESH:
            state = "ROMMEL"
            enter_time = now
            print(f"[EVENT] ENTER ROMMEL (dist={dist:.3f})")
    else:
        if dist > HARD_EXIT_DIST:
            state = "UNKNOWN"
            print(f"[EVENT] HARD EXIT (dist={dist:.3f})")
        elif dist > EXIT_THRESH and (now - enter_time) > MIN_HOLD_SECONDS:
            state = "UNKNOWN"
            print(f"[EVENT] EXIT ROMMEL (dist={dist:.3f})")

try:
    while running:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        H, W, _ = frame.shape

        # Decide detection
        do_detect = (frame_count % DETECT_INTERVAL == 1) or (face_box is None)

        if do_detect:
            scale = DETECT_WIDTH / W if W > DETECT_WIDTH else 1.0
            det_frame = cv2.resize(frame, (int(W*scale), int(H*scale))) if scale != 1.0 else frame
            rgb_small = cv2.cvtColor(det_frame, cv2.COLOR_BGR2RGB)
            result = face_detector.process(rgb_small)
            best_box = None
            if result.detections:
                best_conf = 0.0
                df_h, df_w = rgb_small.shape[:2]
                for det in result.detections:
                    score = det.score[0]
                    if score < MIN_SCORE:
                        continue
                    rel = det.location_data.relative_bounding_box
                    x1 = int(rel.xmin * df_w)
                    y1 = int(rel.ymin * df_h)
                    bw = int(rel.width * df_w)
                    bh = int(rel.height * df_h)
                    if bw <= 0 or bh <= 0:
                        continue
                    x2 = x1 + bw
                    y2 = y1 + bh
                    if scale != 1.0:
                        inv = 1.0/scale
                        x1 = int(x1 * inv); y1 = int(y1 * inv)
                        x2 = int(x2 * inv); y2 = int(y2 * inv)
                    if score > best_conf:
                        best_conf = score
                        best_box = (x1,y1,x2,y2)
            face_box = best_box

        label = "Unknown"
        dist_display = None

        skip_embed = SKIP_EMBED_ODD and (frame_count % 2 == 1)

        if face_box and not skip_embed:
            adj_box = expand_box(face_box, EXPAND_FACTOR, frame.shape)
            tensor = preprocess_crop(frame, adj_box)
            if tensor is not None:
                dist = classify(tensor)
                dist_display = dist

                # Spike ignore
                if dist > SPIKE_IGNORE_DIST and len(recent_embs) >= 2:
                    prev_avg = np.mean(list(recent_embs)[:-1], axis=0)
                    prev_dist = np.linalg.norm(prev_avg - centroid)
                    if prev_dist < 0.80:
                        recent_embs.pop()
                        dist = prev_dist
                        dist_display = dist

                update_state(dist)
                label = "Rommel" if state == "ROMMEL" else "Unknown"

        # Draw
        if face_box:
            x1,y1,x2,y2 = map(int, face_box)
            color = (0,255,0) if state == "ROMMEL" else (0,0,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            if dist_display is not None:
                cv2.putText(frame, f"{label} {dist_display:.3f}",
                            (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            else:
                cv2.putText(frame, label,
                            (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now
        cv2.putText(frame, f"FPS:{fps:.1f}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255),2)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

    cv2.destroyAllWindows()

except KeyboardInterrupt:
    running = False
    cv2.destroyAllWindows()
finally:
    running = False
