import os
import numpy as np
import tensorflow as tf
import cv2

from sklearn.metrics import accuracy_score

# ---------------- CONFIG ----------------
TEST_DIR = r"dataset\test"
MODEL_PATH = r"models\best_model_maskaug.keras"
OUT_REPORT = r"results\masked_vs_unmasked_maskaug.txt"
CLASSES_PATH = r"models\classes.txt"
IMG_SIZE = 224
CONF_THRESH = 0.0  # keep 0 for pure accuracy (no Unknown)
SAVE_SAMPLES = 12
OUT_SAMPLES_DIR = r"results\masked_samples"
SEED = 42
# ----------------------------------------

os.makedirs("results", exist_ok=True)
os.makedirs(OUT_SAMPLES_DIR, exist_ok=True)
np.random.seed(SEED)

# Load classes
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]
name_to_idx = {n: i for i, n in enumerate(class_names)}

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

def list_test_images(root_dir):
    exts = (".png", ".jpg", ".jpeg")
    paths = []
    labels = []
    for person in sorted(os.listdir(root_dir)):
        pdir = os.path.join(root_dir, person)
        if not os.path.isdir(pdir):
            continue
        for fn in os.listdir(pdir):
            if fn.lower().endswith(exts):
                paths.append(os.path.join(pdir, fn))
                labels.append(name_to_idx[person])
    return paths, np.array(labels, dtype=np.int64)

def preprocess_bgr_for_mobilenet(bgr):
    bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = bgr.astype(np.float32)
    # MobileNetV2 expects [-1, 1]
    x = (x / 127.5) - 1.0
    return np.expand_dims(x, axis=0)

def apply_synthetic_mask(bgr):
    """
    Simple occlusion mask: covers lower half of face.
    Works well for masked-face robustness testing.
    """
    h, w = bgr.shape[:2]
    out = bgr.copy()

    # Mask region (lower face)
    y1 = int(h * 0.55)
    y2 = int(h * 0.95)
    x1 = int(w * 0.15)
    x2 = int(w * 0.85)

    # Solid color mask (light gray) + slight shading
    mask_color = (200, 200, 200)
    cv2.rectangle(out, (x1, y1), (x2, y2), mask_color, thickness=-1)

    # Add a simple "crease" line
    cv2.line(out, (x1, int(h * 0.72)), (x2, int(h * 0.72)), (170, 170, 170), 2)

    # Optional: soft blend edges for realism
    blur = out.copy()
    blur = cv2.GaussianBlur(blur, (9, 9), 0)
    alpha = 0.35
    out = cv2.addWeighted(out, 1 - alpha, blur, alpha, 0)

    return out

def predict_label(bgr):
    inp = preprocess_bgr_for_mobilenet(bgr)
    probs = model.predict(inp, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    conf = float(np.max(probs))

    if conf < CONF_THRESH:
        return -1, conf  # unknown
    return pred_idx, conf

# ---- Run evaluation ----
paths, y_true = list_test_images(TEST_DIR)

y_pred_unmasked = []
y_pred_masked = []

# pick random samples for saving
sample_indices = np.random.choice(len(paths), size=min(SAVE_SAMPLES, len(paths)), replace=False)
sample_set = set(sample_indices.tolist())

saved = 0

for i, p in enumerate(paths):
    bgr = cv2.imread(p)
    if bgr is None:
        continue

    # unmasked pred
    pred_u, conf_u = predict_label(bgr)
    y_pred_unmasked.append(pred_u)

    # masked pred
    bgr_m = apply_synthetic_mask(bgr)
    pred_m, conf_m = predict_label(bgr_m)
    y_pred_masked.append(pred_m)

    # Save sample masked images
    if i in sample_set:
        # write masked image as PNG
        out_path = os.path.join(OUT_SAMPLES_DIR, f"sample_{saved:02d}.png")
        cv2.imwrite(out_path, bgr_m)
        saved += 1

# Compute accuracies (ignore unknown = -1 if any; here CONF_THRESH=0 so none)
acc_unmasked = accuracy_score(y_true, y_pred_unmasked)
acc_masked = accuracy_score(y_true, y_pred_masked)

drop = acc_unmasked - acc_masked

report = (
    "SafeSynth-FR Masked vs Unmasked Evaluation\n"
    "----------------------------------------\n"
    f"Total test images: {len(y_true)}\n"
    f"Classes: {len(class_names)}\n\n"
    f"Unmasked accuracy: {acc_unmasked:.4f}\n"
    f"Masked accuracy:   {acc_masked:.4f}\n"
    f"Accuracy drop:     {drop:.4f}\n\n"
    f"Masked samples saved to: {OUT_SAMPLES_DIR}\n"
)

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print(f"✅ Saved report: {OUT_REPORT}")