import cv2, time, threading, torch, numpy as np
from facenet_pytorch import InceptionResnetV1
import mediapipe as mp
from pathlib import Path

# ================== CONFIG (DEBUG STABLE BASE) ==================
CAM_INDEX        = 0
DETECT_INTERVAL  = 1        # Detect every frame for now (simplest & most stable)
DETECT_WIDTH     = 640
ENTER_THRESH     = 0.78     # Raised because alignment is OFF
EXIT_THRESH      = 0.86
MIN_HOLD_SECONDS = 0.8
HARD_EXIT_DIST   = 1.05
SPIKE_IGNORE_DIST= 1.00
MIN_SCORE        = 0.60
EXPAND_FACTOR    = 0.10
SKIP_EMBED_ODD   = False

# Optional accuracy features (currently OFF for debugging)
USE_ALIGNMENT    = False    # Re-enable later
USE_CLAHE        = False
BLUR_VAR_THRESH  = 20.0     # Lenient (was 45). Reject only very blurry frames.

# EMA smoothing / outlier rejection
EMA_ALPHA          = 0.30
EMB_OUTLIER_FACTOR = 3.0     # Lenient so early embeddings not rejected
RECENT_DIFF_HISTORY= 50

# Session adaptive thresholds (off for now)
USE_SESSION_ADAPT = False

WINDOW_NAME      = "Rommel MediaPipe Debug"
# ================================================================

device = torch.device("cpu")

centroid_path = Path("rommel_centroid.npy")
assert centroid_path.exists(), "rommel_centroid.npy not found."
centroid = np.load(str(centroid_path))

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=MIN_SCORE)

# We keep face mesh import optional (alignment disabled now)
if USE_ALIGNMENT:
    mp_mesh = mp.solutions.face_mesh
    face_mesh = mp_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                 refine_landmarks=False, min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)
else:
    face_mesh = None

if USE_CLAHE:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
else:
    clahe = None

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

threading.Thread(target=capture_loop, daemon=True).start()

# State
state = "UNKNOWN"
enter_time = 0.0
frame_count = 0
face_box = None

# Smoothing
ema_emb = None
recent_diff = []
session_genuine = []
SESSION_LOCKED = False

prev_time = time.time()

def expand_box(box, expand_factor, shape):
    x1,y1,x2,y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    cx = x1 + w/2
    cy = y1 + h/2
    nw = w * (1 + expand_factor)
    nh = h * (1 + expand_factor)
    nx1 = int(cx - nw/2); ny1 = int(cy - nh/2)
    nx2 = int(cx + nw/2); ny2 = int(cy + nh/2)
    H,W = shape[:2]
    nx1 = max(0, nx1); ny1 = max(0, ny1)
    nx2 = min(W, nx2); ny2 = min(H, ny2)
    return (nx1, ny1, nx2, ny2)

def align_face(crop_bgr):
    # Currently unused (disabled) – placeholder
    return crop_bgr

def preprocess_crop(frame, box):
    x1,y1,x2,y2 = map(int, box)
    if x2 <= x1 or y2 <= y1:
        print("DEBUG invalid box dims")
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        print("DEBUG empty crop")
        return None

    if USE_CLAHE:
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        y,cr,cb = cv2.split(ycrcb)
        y = clahe.apply(y)
        ycrcb = cv2.merge([y,cr,cb])
        crop = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    if USE_ALIGNMENT:
        if crop.shape[0] < 180 or crop.shape[1] < 180:
            crop = cv2.resize(crop, (180,180), interpolation=cv2.INTER_AREA)
        crop = align_face(crop)

    # Ensure size
    if crop.shape[0] != 160 or crop.shape[1] != 160:
        crop = cv2.resize(crop, (160,160), interpolation=cv2.INTER_AREA)

    # Blur gate
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_var < BLUR_VAR_THRESH:
        # Just skip embedding this frame (will reuse last state)
        print(f"DEBUG blur_reject var={blur_var:.1f}")
        return None

    arr = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    m, s = arr.mean(), arr.std() + 1e-6
    arr = (arr - m)/s
    arr = np.transpose(arr, (2,0,1))
    return torch.from_numpy(arr).unsqueeze(0)

def classify(tensor):
    global ema_emb, recent_diff
    with torch.no_grad():
        emb = resnet(tensor).squeeze(0).cpu().numpy()

    if ema_emb is None:
        ema_emb = emb
        recent_diff.append(0.0)
        print("[DEBUG] EMA initialized")
    else:
        diff = np.linalg.norm(emb - ema_emb)
        median_diff = np.median(recent_diff) if recent_diff else diff
        # lenient outlier rejection
        if diff <= EMB_OUTLIER_FACTOR * (median_diff + 1e-6):
            ema_emb = EMA_ALPHA * emb + (1 - EMA_ALPHA) * ema_emb
            recent_diff.append(diff)
            if len(recent_diff) > RECENT_DIFF_HISTORY:
                recent_diff = recent_diff[-RECENT_DIFF_HISTORY:]
        else:
            print(f"DEBUG embedding_outlier diff={diff:.3f} med={median_diff:.3f}")
    dist = np.linalg.norm(ema_emb - centroid)
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

        # Force detection every frame (debug)
        scale = DETECT_WIDTH / W if W > DETECT_WIDTH else 1.0
        det_frame = cv2.resize(frame, (int(W*scale), int(H*scale))) if scale != 1.0 else frame
        rgb_small = cv2.cvtColor(det_frame, cv2.COLOR_BGR2RGB)
        result = face_detector.process(rgb_small)

        best_box = None
        if result.detections:
            best_conf = 0.0
            df_h, df_w = rgb_small.shape[:2]
            for det in result.detections:
                s = det.score[0]
                if s < MIN_SCORE:
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
                    x1 = int(x1*inv); y1 = int(y1*inv)
                    x2 = int(x2*inv); y2 = int(y2*inv)
                if s > best_conf:
                    best_conf = s
                    best_box = (x1,y1,x2,y2)
        face_box = best_box

        dist_display = None
        if face_box:
            expanded = expand_box(face_box, EXPAND_FACTOR, frame.shape)
            tensor = preprocess_crop(frame, expanded)
            if tensor is not None:
                if not (SKIP_EMBED_ODD and frame_count % 2 == 1):
                    dist = classify(tensor)
                    dist_display = dist
                    print(f"DEBUG dist={dist:.3f}")
                    update_state(dist)

        # Draw
        if face_box:
            x1,y1,x2,y2 = map(int, face_box)
            color = (0,255,0) if state == "ROMMEL" else (0,0,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            if dist_display is not None:
                cv2.putText(frame, f"{state} {dist_display:.3f}",
                            (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            else:
                cv2.putText(frame, state, (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now
        cv2.putText(frame, f"FPS:{fps:.1f}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255),2)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

    cv2.destroyAllWindows()

except KeyboardInterrupt:
    running = False
    cv2.destroyAllWindows()
finally:
    running = False
