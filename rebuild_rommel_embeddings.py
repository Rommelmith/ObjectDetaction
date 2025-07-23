import cv2, torch, numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from pathlib import Path
from PIL import Image

device = torch.device("cpu")

ROM_DIR = Path("faces/rommel")
imgs = sorted([p for p in ROM_DIR.glob("*") if p.suffix.lower() in (".jpg",".jpeg",".png")])
assert imgs, "No images found."

mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def preprocess_crop(bgr, box):
    x1,y1,x2,y2 = map(int, box)
    h,w,_ = bgr.shape
    x1=max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)
    crop = bgr[y1:y2, x1:x2]
    if crop.size==0: return None
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((160,160))
    arr = np.float32(pil)/255.0
    mean, std = arr.mean(), arr.std()+1e-6
    arr = (arr-mean)/std
    arr = np.transpose(arr,(2,0,1))
    return torch.from_numpy(arr).unsqueeze(0)

embs = []
names = []
bad = []

for p in imgs:
    img = cv2.imread(str(p))
    if img is None:
        bad.append((p.name,"read_fail")); continue
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(Image.fromarray(rgb))
    if boxes is None:
        bad.append((p.name,"no_face")); continue
    # highest prob
    best_i = np.argmax(probs)
    if probs[best_i] < 0.80:
        bad.append((p.name,"low_prob")); continue
    tensor = preprocess_crop(img, boxes[best_i])
    if tensor is None:
        bad.append((p.name,"crop_fail")); continue
    with torch.no_grad():
        emb = resnet(tensor).squeeze(0).cpu().numpy()
    embs.append(emb); names.append(p.name)

print(f"Good embeddings: {len(embs)} / {len(imgs)}")
if bad:
    print("Issues:")
    for b in bad:
        print("  -", b[0], "=>", b[1])

if len(embs) < 5:
    raise SystemExit("Too few usable images.")

E = np.vstack(embs)
centroid = E.mean(axis=0)
np.save("rommel_centroid.npy", centroid)
np.save("rommel_all.npy", E)

dists = np.linalg.norm(E - centroid, axis=1)
print("\nPer-image distances:")
for n,d in sorted(zip(names,dists), key=lambda x:x[1]):
    print(f"  {n}: {d:.3f}")
print(f"\nStats: mean={dists.mean():.3f} max={dists.max():.3f} min={dists.min():.3f}")
