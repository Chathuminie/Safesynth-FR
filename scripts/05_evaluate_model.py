import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels

DATA_DIR = "dataset"
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_PATH = os.path.join("models", "best_model.keras")
CLASSES_PATH = os.path.join("models", "classes.txt")

IMG_SIZE = 224
BATCH = 16

os.makedirs("results", exist_ok=True)

# Load class names (the correct order!)
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

num_classes = len(class_names)
print("Classes:", num_classes)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Build test dataset (same class folder names)
def list_image_files(root_dir):
    exts = (".png", ".jpg", ".jpeg")
    paths = []
    labels = []
    name_to_idx = {name: i for i, name in enumerate(class_names)}

    for person in sorted(os.listdir(root_dir)):
        pdir = os.path.join(root_dir, person)
        if not os.path.isdir(pdir):
            continue
        for fn in os.listdir(pdir):
            if fn.lower().endswith(exts):
                paths.append(os.path.join(pdir, fn))
                labels.append(name_to_idx[person])
    return paths, np.array(labels, dtype=np.int64)

paths, y_true = list_image_files(TEST_DIR)

def load_img(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1.0   # MobileNetV2 range
    return img

ds = tf.data.Dataset.from_tensor_slices(paths)
ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

# Predict
probs = model.predict(ds, verbose=1)
y_pred = np.argmax(probs, axis=1)

# Report
rep = classification_report(y_true, y_pred, target_names=class_names, digits=4)
with open("results/classification_report.txt", "w", encoding="utf-8") as f:
    f.write(rep)
print("✅ Saved: results/classification_report.txt")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
plt.imshow(cm)
plt.title("Confusion Matrix (All Classes)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=200)
plt.close()
print("✅ Saved: results/confusion_matrix.png")

# Top-20 classes confusion matrix (by support)
# Since each class has same test count, just take first 20
top_k = min(20, num_classes)
idx = list(range(top_k))
cm20 = cm[np.ix_(idx, idx)]

plt.figure(figsize=(10, 8))
plt.imshow(cm20)
plt.title("Confusion Matrix (Top 20 Classes)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(range(top_k), [class_names[i] for i in idx], rotation=90, fontsize=7)
plt.yticks(range(top_k), [class_names[i] for i in idx], fontsize=7)
plt.colorbar()
plt.tight_layout()
plt.savefig("results/confusion_matrix_top20.png", dpi=200)
plt.close()
print("✅ Saved: results/confusion_matrix_top20.png")