import tensorflow as tf

SOURCE = r"models\model_maskaug.h5"   # use this first
OUT_H5 = r"models\web_model.h5"

print("Loading:", SOURCE)
model = tf.keras.models.load_model(SOURCE, compile=False)
print("Loaded successfully.")

model.save(OUT_H5)
print("✅ Saved:", OUT_H5)