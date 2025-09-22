# image_utils.py
import os, io, uuid
from PIL import Image
import numpy as np
import cv2
from werkzeug.utils import secure_filename

ALLOWED_EXT = {"png","jpg","jpeg","bmp","gif"}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def allowed_file(filename):
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXT

def save_bytes_as_png(bytes_data, save_dir, filename):
    """
    Save uploaded bytes as PNG (force PNG). Returns (basename, full_path).
    Files are saved into save_dir (create if missing).
    """
    ensure_dir(save_dir)
    filename = secure_filename(filename)
    root, _ = os.path.splitext(filename)
    uid = uuid.uuid4().hex[:8]
    out_name = f"{root}_{uid}.png"
    out_path = os.path.join(save_dir, out_name)
    try:
        img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        img.save(out_path, format="PNG")
    except Exception:
        # fallback
        with open(out_path, "wb") as f:
            f.write(bytes_data)
    return out_name, out_path

def load_image_bgr_from_path(path):
    """Return OpenCV BGR image from a saved path."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cv2 failed to read {path}")
    return img

def load_image_bgr_from_bytes(bytes_data):
    """Return OpenCV BGR image from raw bytes (does not write to disk)."""
    arr = np.frombuffer(bytes_data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        # fallback via PIL
        pil = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return img

# --------- Preprocessing for TF / Keras -----------
# image_utils.py
def preprocess_for_keras(img_bgr, size=(224,224), normalize=True):
    # Convert BGR -> RGB (3 channels)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, size, interpolation=cv2.INTER_AREA)
    arr = img_resized.astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)  # shape (1, H, W, 3)