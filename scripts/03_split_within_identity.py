import os
import shutil
import random

SRC_DIR = r"dataset\identities_30"
BASE_OUT = r"dataset"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
# remaining → test

random.seed(42)

train_dir = os.path.join(BASE_OUT, "train")
val_dir = os.path.join(BASE_OUT, "val")
test_dir = os.path.join(BASE_OUT, "test")

# Remove old splits if exist
for d in [train_dir, val_dir, test_dir]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

persons = sorted(os.listdir(SRC_DIR))

for person in persons:
    person_path = os.path.join(SRC_DIR, person)
    imgs = sorted(os.listdir(person_path))
    random.shuffle(imgs)

    n = len(imgs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_imgs = imgs[:n_train]
    val_imgs = imgs[n_train:n_train + n_val]
    test_imgs = imgs[n_train + n_val:]

    for split_name, split_imgs in [
        ("train", train_imgs),
        ("val", val_imgs),
        ("test", test_imgs),
    ]:
        split_person_dir = os.path.join(BASE_OUT, split_name, person)
        os.makedirs(split_person_dir, exist_ok=True)

        for img in split_imgs:
            src = os.path.join(person_path, img)
            dst = os.path.join(split_person_dir, img)
            shutil.copy(src, dst)

print("✅ Done splitting dataset.")
print("Train / Val / Test created.")