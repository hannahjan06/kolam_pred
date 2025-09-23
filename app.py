# app.py (excerpt)
from flask import Flask, request, render_template, url_for
import os
import cv2
from image_utils import save_bytes_as_png, load_image_bgr_from_path, preprocess_for_keras, allowed_file
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Input

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# somewhere near the top of app.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress INFO and WARNING logs

app = Flask(__name__)
# ensure static/uploads path exists
UPLOAD_DIR = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# load model once
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

CLASS_NAMES = {0: "Kambi Kolam", 1: "Sikku Kolam", 2: "Pulli Kolam"}

# Choose mode depending on how your model was trained:
# - "image" => model expects (1,224,224,3) (recommended for EfficientNet)
# - "flatten" => model expects (1, 224*224*3) or (1, 224*224) if grayscale
MODEL_INPUT_MODE = "image"  # <- change to "flatten" if your .keras model expects flattened input

emotion_folders = {
    "happy": os.path.join(app.static_folder, "images", "happy"),
    "sad": os.path.join(app.static_folder, "images", "sad"),
    "angry": os.path.join(app.static_folder, "images", "angry"),
    "surprise": os.path.join(app.static_folder, "images", "surprise"),
    "fear": os.path.join(app.static_folder, "images", "fear"),
    "neutral": os.path.join(app.static_folder, "images", "neutral"),
}

@app.route("/")
def index():
    return render_template("index.html")

def preprocess_image(img_path, norm_layer, img_size=(224,224)):
    # Load BGR image
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize
    img_resized = cv2.resize(img_rgb, img_size).astype("float32") / 255.0

    # Ensure 3 channels
    if img_resized.shape[-1] == 1:
        img_resized = np.repeat(img_resized, 3, axis=-1)

    # Add batch dimension
    img_batch = np.expand_dims(img_resized, axis=0)

    # Normalize using the same layer as training
    img_norm = norm_layer(img_batch)
    return img_norm

@app.route("/upload", methods=["POST"])
def upload_file():
    files = request.files.getlist("userFile")
    if not files:
        return render_template("index.html", prediction=None, image_url=None)

    f = files[0]
    if not f or not allowed_file(f.filename):
        return render_template("index.html", prediction="Invalid file", image_url=None)

    # Save uploaded file
    bytes_data = f.read()
    saved_basename, saved_path = save_bytes_as_png(bytes_data, UPLOAD_DIR, f.filename)
    image_url = url_for('static', filename=f"uploads/{saved_basename}")

    # Preprocessing
    norm_layer = tf.keras.layers.Normalization()
    norm_layer.adapt(np.zeros((1,224,224,3)))  # dummy adapt to avoid errors; replace with real train stats if available
    inp = preprocess_image(saved_path, norm_layer)

    if inp is None:
        return render_template("index.html", prediction="Error processing image", image_url=image_url)

    # Predict
    preds = model.predict(inp)
    idx = int(np.argmax(preds, axis=1)[0])
    prob = float(np.max(preds))
    label = CLASS_NAMES.get(idx, str(idx))
    prediction_text = f"{label} ({prob:.2f})"

    return render_template("index.html", prediction=prediction_text, image_url=image_url)

from flask import Response, stream_with_context

cap = cv2.VideoCapture(0)  # global video capture

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Flip horizontally for mirror effect (optional)
            frame = cv2.flip(frame, 1)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Yield in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

import random
from flask import jsonify

@app.route("/detect_emotion")
def detect_emotion():
    from deepface import DeepFace
    success, frame = cap.read()
    if not success:
        return jsonify({"emotion": None})

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return jsonify({"emotion": None})

    # Use the first face
    x, y, w, h = faces[0]
    face_roi = frame[y:y+h, x:x+w]
    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
    emotion = result[0]['dominant_emotion']
    return jsonify({"emotion": emotion})

@app.route("/mood_image")
def mood_image():
    from flask import request, url_for
    emotion = request.args.get("emotion")
    folder = emotion_folders.get(emotion)
    if not folder:
        return jsonify({"url": ""})

    img_files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not img_files:
        return jsonify({"url": ""})

    chosen = random.choice(img_files)

    # Always build relative path inside static/
    rel_path = f"images/{emotion}/{chosen}"
    url = url_for("static", filename=rel_path)

    print("Serving:", url)  # Debug
    return jsonify({"url": url})

if __name__ == "__main__":
    app.run(debug=True)