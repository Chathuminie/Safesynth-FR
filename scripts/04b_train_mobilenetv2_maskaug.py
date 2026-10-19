import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
os.makedirs("results", exist_ok=True)

DATA_DIR = "dataset_maskaug"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
TEST_DIR  = os.path.join(DATA_DIR, "test")

IMG_SIZE = 224
BATCH = 16
EPOCHS_1 = 5
EPOCHS_2 = 8
SEED = 42

os.makedirs("models", exist_ok=True)
tf.random.set_seed(SEED)

def list_image_files(root_dir):
    exts = (".png", ".jpg", ".jpeg")
    paths = []
    for person in sorted(os.listdir(root_dir)):
        pdir = os.path.join(root_dir, person)
        if not os.path.isdir(pdir):
            continue
        for fn in os.listdir(pdir):
            if fn.lower().endswith(exts):
                paths.append(os.path.join(pdir, fn))
    return paths

class_names = [d for d in sorted(os.listdir(TRAIN_DIR)) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
num_classes = len(class_names)
print("Classes:", num_classes)

# Save classes
with open("models/classes_maskaug.txt", "w", encoding="utf-8") as f:
    for c in class_names:
        f.write(c + "\n")

keys = tf.constant(class_names)
vals = tf.constant(list(range(num_classes)), dtype=tf.int64)
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys, vals),
    default_value=-1
)

def path_to_label(path):
    parts = tf.strings.split(path, os.sep)
    person = parts[-2]
    return table.lookup(person)

def load_and_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1.0
    return img, path_to_label(path)

def make_ds(paths, shuffle):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_ds(list_image_files(TRAIN_DIR), shuffle=True)
val_ds   = make_ds(list_image_files(VAL_DIR), shuffle=False)
test_ds  = make_ds(list_image_files(TEST_DIR), shuffle=False)

base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n=== Stage 1 ===")
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_1,
    verbose=1
)

print("\n=== Stage 2 (fine-tune) ===")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_2,
    verbose=1
)

# Combine training history from both stages
acc = history1.history["accuracy"] + history2.history["accuracy"]
val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
losses = history1.history["loss"] + history2.history["loss"]
val_losses = history1.history["val_loss"] + history2.history["val_loss"]

print("Training Accuracy:", acc)
print("Validation Accuracy:", val_acc)
print("Training Loss:", losses)
print("Validation Loss:", val_losses)

# Plot accuracy graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(acc) + 1), acc, marker="o", label="Training Accuracy")
plt.plot(range(1, len(val_acc) + 1), val_acc, marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy over Training Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/model_accuracy_over_epochs.png", dpi=300)
plt.show()

# Plot loss graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(losses) + 1), losses, marker="o", label="Training Loss")
plt.plot(range(1, len(val_losses) + 1), val_losses, marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss over Training Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/model_loss_over_epochs.png", dpi=300)
plt.show()

print("\nEvaluating on test set...")
loss, acc_test = model.evaluate(test_ds)
print(f"✅ MaskAug Test accuracy: {acc_test:.4f}")

model.save("models/best_model_maskaug.keras")
model.save("models/model_maskaug.h5")
print("✅ Saved: models/best_model_maskaug.keras and models/model_maskaug.h5")

# Save web-friendly model
model.save("models/web_model.h5", include_optimizer=False)
print("✅ Saved: models/web_model.h5")

