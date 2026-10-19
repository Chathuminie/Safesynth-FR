import os
import numpy as np
import tensorflow as tf
import cv2
from sklearn.metrics import accuracy_score

# ---------------- CONFIG ----------------
TEST_DIR = r"dataset_maskaug\test"
MODEL_PATH = r"models\best_model_maskaug.keras"
CLASSES_PATH = r"models\classes_maskaug.txt"
OUT_REPORT = r"results\angle_robustness_report.txt"

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

def rotate_image(bgr, angle):
    h, w = bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        bgr,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return rotated

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
rot_10_acc = evaluate_condition(paths, y_true, lambda x: rotate_image(x, 10))
rot_minus_10_acc = evaluate_condition(paths, y_true, lambda x: rotate_image(x, -10))
rot_20_acc = evaluate_condition(paths, y_true, lambda x: rotate_image(x, 20))
rot_minus_20_acc = evaluate_condition(paths, y_true, lambda x: rotate_image(x, -20))

report = (
    "SafeSynth-FR Angle Robustness Evaluation\n"
    "---------------------------------------\n"
    f"Total test images: {len(y_true)}\n"
    f"Classes: {len(class_names)}\n\n"
    f"Baseline accuracy:   {baseline_acc:.4f}\n"
    f"Rotation +10°:       {rot_10_acc:.4f}\n"
    f"Rotation -10°:       {rot_minus_10_acc:.4f}\n"
    f"Rotation +20°:       {rot_20_acc:.4f}\n"
    f"Rotation -20°:       {rot_minus_20_acc:.4f}\n\n"
    f"Drop at +10°:        {baseline_acc - rot_10_acc:.4f}\n"
    f"Drop at -10°:        {baseline_acc - rot_minus_10_acc:.4f}\n"
    f"Drop at +20°:        {baseline_acc - rot_20_acc:.4f}\n"
    f"Drop at -20°:        {baseline_acc - rot_minus_20_acc:.4f}\n"
)

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print(f"✅ Saved report: {OUT_REPORT}")