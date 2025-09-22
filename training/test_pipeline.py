import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Input

# ---- CONFIG ----
IMAGE_DIR = "test_images"  # folder with images
MODEL_PATH = "kolam_classifier.keras"  # your trained model
CLASS_NAMES = {0: "Kambi Kolam", 1: "Sikku Kolam", 2: "Pulli Kolam"}
IMG_SIZE = (224, 224)

# ---- LOAD MODEL ----
norm_layer = tf.keras.layers.Normalization()
base_model = EfficientNetB0(weights=None, include_top=False, input_tensor=Input(shape=(224,224,3)))

model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

# Load weights instead of the full model
model.load_weights("models/kolam_classifier.keras")

# ---- FUNCTION TO PREPROCESS IMAGE ----
def preprocess_image(img_path, norm_layer):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE).astype("float32")
    # ensure 3 channels
    if img_resized.shape[-1] == 1:
        img_resized = np.repeat(img_resized, 3, axis=-1)
    img_batch = np.expand_dims(img_resized, axis=0)
    img_norm = norm_layer(img_batch)  # normalize exactly like during training
    return img_norm

# ---- LOOP OVER IMAGES AND PREDICT ----
results = []
for fname in os.listdir(IMAGE_DIR):
    path = os.path.join(IMAGE_DIR, fname)
    inp = preprocess_image(path, norm_layer)
    if inp is None:
        print(f"Skipping {fname} (cannot read image)")
        continue
    preds = model.predict(inp)
    idx = int(np.argmax(preds, axis=1)[0])
    prob = float(np.max(preds))
    label = CLASS_NAMES.get(idx, str(idx))
    results.append((fname, label, prob))

# ---- PRINT RESULTS ----
for fname, label, prob in results:
    print(f"{fname} -> {label} ({prob:.2f})")
