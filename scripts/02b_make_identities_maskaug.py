import os
import random
import cv2
import numpy as np
import shutil

IN_DIR = r"dataset\processed_faces"
OUT_DIR = r"dataset\identities_30_maskaug"

NUM_IDENTITIES = 30
IMAGES_PER_ID = 10
MASK_COUNT = 5          # out of 10
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# Clear output folder if re-running
for name in os.listdir(OUT_DIR):
    p = os.path.join(OUT_DIR, name)
    if os.path.isdir(p):
        shutil.rmtree(p)

imgs = [f for f in os.listdir(IN_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
imgs.sort()

bases = random.sample(imgs, NUM_IDENTITIES)

def augment(img):
    out = img.copy()

    # flip
    if random.random() < 0.5:
        out = cv2.flip(out, 1)

    # rotate
    angle = random.uniform(-10, 10)
    h, w = out.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # brightness/contrast
    alpha = random.uniform(0.85, 1.15)
    beta = random.randint(-20, 20)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # blur
    if random.random() < 0.3:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    # noise
    if random.random() < 0.3:
        noise = np.random.normal(0, 5, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out

def apply_synthetic_mask(bgr):
    h, w = bgr.shape[:2]
    out = bgr.copy()

    y1 = int(h * 0.55)
    y2 = int(h * 0.95)
    x1 = int(w * 0.15)
    x2 = int(w * 0.85)

    mask_color = (200, 200, 200)
    cv2.rectangle(out, (x1, y1), (x2, y2), mask_color, thickness=-1)
    cv2.line(out, (x1, int(h * 0.72)), (x2, int(h * 0.72)), (170, 170, 170), 2)

    blur = cv2.GaussianBlur(out, (9, 9), 0)
    alpha = 0.35
    out = cv2.addWeighted(out, 1 - alpha, blur, alpha, 0)

    return out

for pid, base_name in enumerate(bases, start=1):
    person_dir = os.path.join(OUT_DIR, f"person_{pid:03d}")
    os.makedirs(person_dir, exist_ok=True)

    base_path = os.path.join(IN_DIR, base_name)
    base_img = cv2.imread(base_path)
    if base_img is None:
        raise RuntimeError(f"Failed to read: {base_path}")

    # create 10 variants
    for j in range(IMAGES_PER_ID):
        img = augment(base_img)

        # first MASK_COUNT are masked
        if j < MASK_COUNT:
            img = apply_synthetic_mask(img)

        cv2.imwrite(os.path.join(person_dir, f"img_{j:03d}.png"), img)

print("✅ Done.")
print("Created identities:", NUM_IDENTITIES)
print("Images per identity:", IMAGES_PER_ID, f"(masked={MASK_COUNT}, unmasked={IMAGES_PER_ID-MASK_COUNT})")
print("Output:", OUT_DIR)