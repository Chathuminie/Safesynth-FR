import os
import numpy as np
import tensorflow as tf
import cv2
from sklearn.metrics import accuracy_score

# ---------------- CONFIG ----------------
TEST_DIR = r"dataset_maskaug\test"
MODEL_PATH = r"models\best_model_maskaug.keras"
CLASSES_PATH = r"models\classes_maskaug.txt"
OUT_REPORT = r"results\lighting_robustness_report.txt"

IMG_SIZE = 224
SEED = 42
# ----------------------------------------

os.makedirs("results", exist_ok=True)
np.random.seed(SEED)

# Load classes
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

name_to_idx = {name: i for i, name in enumerate(class_names)}

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
    x = (x / 127.5) - 1.0
    return np.expand_dims(x, axis=0)

def predict_label(bgr):
    inp = preprocess_bgr_for_mobilenet(bgr)
    probs = model.predict(inp, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    conf = float(np.max(probs))
    return pred_idx, conf

# ---------- Lighting transformations ----------

def make_dark(bgr):
    return cv2.convertScaleAbs(bgr, alpha=0.6, beta=-20)

def make_bright(bgr):
    return cv2.convertScaleAbs(bgr, alpha=1.2, beta=25)

def make_low_contrast(bgr):
    gray_mean = np.mean(bgr)
    out = (bgr.astype(np.float32) - gray_mean) * 0.65 + gray_mean
    return np.clip(out, 0, 255).astype(np.uint8)

def make_outdoor_like(bgr):
    # Simulates stronger highlights and slight color shift
    out = cv2.convertScaleAbs(bgr, alpha=1.15, beta=20)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 1.10   # slightly more saturation
    hsv[..., 2] *= 1.08   # slightly brighter
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out

def evaluate_condition(paths, y_true, transform_fn):
    y_pred = []

    for p in paths:
        bgr = cv2.imread(p)
        if bgr is None:
            continue

        bgr_mod = transform_fn(bgr)
        pred, _ = predict_label(bgr_mod)
        y_pred.append(pred)

    return accuracy_score(y_true, y_pred)

# ---------- Run evaluation ----------

paths, y_true = list_test_images(TEST_DIR)

baseline_acc = evaluate_condition(paths, y_true, lambda x: x)
dark_acc = evaluate_condition(paths, y_true, make_dark)
bright_acc = evaluate_condition(paths, y_true, make_bright)
low_contrast_acc = evaluate_condition(paths, y_true, make_low_contrast)
outdoor_acc = evaluate_condition(paths, y_true, make_outdoor_like)

report = (
    "SafeSynth-FR Lighting Robustness Evaluation\n"
    "------------------------------------------\n"
    f"Total test images: {len(y_true)}\n"
    f"Classes: {len(class_names)}\n\n"
    f"Baseline accuracy:      {baseline_acc:.4f}\n"
    f"Dark lighting accuracy: {dark_acc:.4f}\n"
    f"Bright lighting accuracy: {bright_acc:.4f}\n"
    f"Low contrast accuracy:  {low_contrast_acc:.4f}\n"
    f"Outdoor-like accuracy:  {outdoor_acc:.4f}\n\n"
    f"Drop under dark lighting:   {baseline_acc - dark_acc:.4f}\n"
    f"Drop under bright lighting: {baseline_acc - bright_acc:.4f}\n"
    f"Drop under low contrast:    {baseline_acc - low_contrast_acc:.4f}\n"
    f"Drop under outdoor-like:    {baseline_acc - outdoor_acc:.4f}\n"
)

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print(f"✅ Saved report: {OUT_REPORT}")