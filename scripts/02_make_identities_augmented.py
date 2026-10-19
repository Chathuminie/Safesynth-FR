import os
import random
import cv2
import numpy as np
import shutil

IN_DIR = r"dataset\processed_faces"
OUT_DIR = r"dataset\identities_30"

NUM_IDENTITIES = 30
IMAGES_PER_ID = 10
SEED = 42

random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# Clear output folder
for name in os.listdir(OUT_DIR):
    p = os.path.join(OUT_DIR, name)
    if os.path.isdir(p):
        shutil.rmtree(p)

imgs = [f for f in os.listdir(IN_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
imgs.sort()

# pick 30 base faces
bases = random.sample(imgs, NUM_IDENTITIES)

def augment(img, idx):
    out = img.copy()

    # random flip
    if random.random() < 0.5:
        out = cv2.flip(out, 1)

    # small rotation
    angle = random.uniform(-10, 10)
    h, w = out.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # brightness/contrast
    alpha = random.uniform(0.85, 1.15)  # contrast
    beta = random.randint(-20, 20)      # brightness
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # slight blur sometimes
    if random.random() < 0.3:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    # slight noise sometimes
    if random.random() < 0.3:
        noise = np.random.normal(0, 5, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out

for pid, base_name in enumerate(bases, start=1):
    person_dir = os.path.join(OUT_DIR, f"person_{pid:03d}")
    os.makedirs(person_dir, exist_ok=True)

    base_path = os.path.join(IN_DIR, base_name)
    base_img = cv2.imread(base_path)
    if base_img is None:
        raise RuntimeError(f"Failed to read: {base_path}")

    # save original as one sample
    cv2.imwrite(os.path.join(person_dir, "img_000.png"), base_img)

    # generate remaining variations
    for j in range(1, IMAGES_PER_ID):
        aug = augment(base_img, j)
        cv2.imwrite(os.path.join(person_dir, f"img_{j:03d}.png"), aug)

print("✅ Done.")
print("Created identities:", NUM_IDENTITIES)
print("Images per identity:", IMAGES_PER_ID)
print("Output:", OUT_DIR)