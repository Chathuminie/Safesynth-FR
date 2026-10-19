import os
import tensorflow as tf

# ---------------- CONFIG ----------------
DATA_DIR = "dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
TEST_DIR  = os.path.join(DATA_DIR, "test")

IMG_SIZE = 224
BATCH = 16
EPOCHS_1 = 5
EPOCHS_2 = 8
SEED = 42
# ----------------------------------------

os.makedirs("models", exist_ok=True)
tf.random.set_seed(SEED)

def list_image_files(root_dir):
    exts = (".png", ".jpg", ".jpeg")
    paths = []
    for person in sorted(os.listdir(root_dir)):
        person_dir = os.path.join(root_dir, person)
        if not os.path.isdir(person_dir):
            continue
        for f in os.listdir(person_dir):
            if f.lower().endswith(exts):
                paths.append(os.path.join(person_dir, f))
    return paths

# ✅ Single source of truth: class order from TRAIN folder names
class_names = [d for d in sorted(os.listdir(TRAIN_DIR)) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
num_classes = len(class_names)
print("Classes:", class_names)
print("Number of classes:", num_classes)

# Save classes for deployment
with open("models/classes.txt", "w", encoding="utf-8") as f:
    for c in class_names:
        f.write(c + "\n")

# Create lookup table: "person_001" -> index
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
    # MobileNetV2 expects [-1, 1]
    img = (img / 127.5) - 1.0
    label = path_to_label(path)
    return img, label

def make_ds(paths, shuffle):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds

train_paths = list_image_files(TRAIN_DIR)
val_paths   = list_image_files(VAL_DIR)
test_paths  = list_image_files(TEST_DIR)

print("Train images:", len(train_paths))
print("Val images:", len(val_paths))
print("Test images:", len(test_paths))

train_ds = make_ds(train_paths, shuffle=True)
val_ds   = make_ds(val_paths, shuffle=False)
test_ds  = make_ds(test_paths, shuffle=False)

# ---------------- MODEL ----------------
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

print("\n=== Stage 1: Feature Extraction ===")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_1)

print("\n=== Stage 2: Fine Tuning (last 30 layers) ===")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_2)

print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"✅ Test accuracy: {test_acc:.4f}")

model.save("models/best_model.keras")
model.save("models/model.h5")
print("✅ Saved: models/best_model.keras and models/model.h5")
print("✅ Saved: models/classes.txt")