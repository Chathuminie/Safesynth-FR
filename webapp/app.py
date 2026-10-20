import os
import secrets

# Reduce TensorFlow log messages in terminal (clean output)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KMP_WARNINGS"] = "0"

import io
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, jsonify, session, redirect, url_for

# =========================================================
# CONFIGURATION
# =========================================================

# Get the root directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to trained face recognition model
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.h5")

# Path to class labels (names of people)
CLASSES_PATH = os.path.join(BASE_DIR, "models", "classes_maskaug.txt")

# Model input size (MobileNetV2 expects 224x224 images)
IMG_SIZE = 224
 
# Supported upload types for image inference
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "BMP", "WEBP"}

# Confidence threshold → below this = "Unknown"
CONF_THRESH = 0.60
UNCERTAIN_MARGIN = 0.08

# Add margin around detected face (improves accuracy)
FACE_MARGIN = 0.30

# =========================================================
# FLASK APP SETUP
# =========================================================

app = Flask(__name__)

# Set SAFESYNTH_SECRET_KEY to keep sessions valid across server restarts.
# Otherwise, generate a fresh key for this local demo process.
app.secret_key = os.environ.get("SAFESYNTH_SECRET_KEY") or secrets.token_hex(32)

# =========================================================
# LOAD MODEL AND CLASS LABELS
# =========================================================

# Load trained deep learning model (MobileNetV2)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Load class names (person identities)
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

# =========================================================
# FACE DETECTOR (HAAR CASCADE)
# =========================================================

# Load OpenCV face detector
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# =========================================================
# SIMPLE USER DATABASE (FOR DEMO ONLY)
# =========================================================

# Register an account through the UI. Users are temporary and cleared on restart.
users_db = {}

# =========================================================
# IMAGE PREPROCESSING FUNCTIONS
# =========================================================

def preprocess(face_bgr):
    """
    Convert face image into format required by model.

    Steps:
    - Resize to 224x224
    - Convert to float
    - Normalize pixels to [-1, 1]
    - Add batch dimension
    """
    face = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
    face = face.astype(np.float32)
    face = (face / 127.5) - 1.0
    return np.expand_dims(face, axis=0)


def center_crop_square(img):
    """
    If face detection fails,
    take center square portion of image.
    """
    h, w = img.shape[:2]
    s = min(h, w)
    y = (h - s) // 2
    x = (w - s) // 2
    return img[y:y + s, x:x + s]


def crop_with_margin(img, x, y, w, h, margin=0.30):
    """
    Expand detected face region.

    This helps include more facial context
    (forehead, chin, etc.)
    """
    H, W = img.shape[:2]

    # Calculate margin size
    mx = int(w * margin)
    my = int(h * margin)

    # Expand bounding box safely
    x1 = max(0, x - mx)  #mx-extra width,move left
    y1 = max(0, y - my)  #my-extra height,move up
    x2 = min(W, x + w + mx)
    y2 = min(H, y + h + my)

    crop = img[y1:y2, x1:x2]

    return crop if crop.size else None


def extract_face_for_inference(bgr):
    """
    Detect and extract face from image.

    Steps:
    - Convert to grayscale
    - Detect face
    - If no face → fallback center crop
    - If face → crop largest face + margin
    - Return square face image
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    # If no face detected
    if len(faces) == 0:
        return None, "no_face"

    detection_status = "multiple_faces" if len(faces) > 1 else "detected"

    # Select largest detected face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

    face = crop_with_margin(bgr, x, y, w, h, margin=FACE_MARGIN)

    if face is None:
        return None, "no_face"

    # Convert to square
    face = center_crop_square(face)

    return face, detection_status


def predict_from_bgr(bgr):
    """
    Run face recognition prediction.

    Returns:
    - predicted label
    - confidence score
    - detection method used
    """
    face, detection_status = extract_face_for_inference(bgr)

    if face is None:
        return {
            "label": "No Face",
            "raw_label": "",
            "confidence": 0.0,
            "detection_status": detection_status
        }

    # Get prediction probabilities
    probs = model.predict(preprocess(face), verbose=0)[0]

    sorted_indices = np.argsort(probs)[::-1]
    pred_idx = int(sorted_indices[0])  # index of best prediction
    conf = float(probs[pred_idx])      # confidence value
    second_conf = float(probs[int(sorted_indices[1])]) if len(sorted_indices) > 1 else 0.0

    raw_label = class_names[pred_idx]
    ambiguity_gap = conf - second_conf

    # Apply confidence threshold and ambiguity rule
    if conf < CONF_THRESH:
        label = "Unknown"
    elif ambiguity_gap < UNCERTAIN_MARGIN:
        label = "Uncertain"
    else:
        label = raw_label

    return {
        "label": label,
        "raw_label": raw_label,
        "confidence": round(conf, 4),
        "detection_status": detection_status,
        "second_confidence": round(second_conf, 4),
        "ambiguity_gap": round(ambiguity_gap, 4)
    }


def load_uploaded_image(file_storage, *, invalid_error, unsupported_error):
    """
    Validate and decode uploaded image content.
    """
    filename = (file_storage.filename or "").strip()
    extension = os.path.splitext(filename)[1].lower()

    if not filename:
        return None, {"ok": False, "error": "No file selected"}, 400

    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        return None, {"ok": False, "error": unsupported_error}, 415

    img_bytes = file_storage.read()
    if not img_bytes:
        return None, {"ok": False, "error": invalid_error}, 400

    try:
        with Image.open(io.BytesIO(img_bytes)) as pil_image:
            image_format = (pil_image.format or "").upper()
            if image_format not in ALLOWED_PIL_FORMATS:
                return None, {"ok": False, "error": unsupported_error}, 415
            rgb = np.array(pil_image.convert("RGB"))
    except Exception:
        return None, {"ok": False, "error": invalid_error}, 400

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), None, None

# =========================================================
# AUTHENTICATION ROUTES
# =========================================================

@app.route("/")
def index():
    """
    Redirect user based on login status.
    """
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    """
    Login system.
    """
    if request.method == "POST":
        data = request.get_json()

        email = data.get("email")
        password = data.get("password")

        # Check credentials
        if email in users_db and users_db[email]["password"] == password:
            session["user"] = email
            session["name"] = users_db[email]["name"]
            return jsonify({"ok": True})

        return jsonify({"ok": False, "error": "Invalid credentials"}), 401

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """
    User registration.
    """
    if request.method == "POST":
        data = request.get_json()

        name = data.get("name")
        email = data.get("email")
        password = data.get("password")

        # Prevent duplicate users
        if email in users_db:
            return jsonify({"ok": False, "error": "Email exists"}), 400

        users_db[email] = {"name": name, "password": password}

        session["user"] = email
        session["name"] = name

        return jsonify({"ok": True})

    return render_template("register.html")


@app.route("/logout")
def logout():
    """
    Logout user.
    """
    session.clear()
    return redirect(url_for("login"))

# =========================================================
# PAGES
# =========================================================

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", name=session["name"])


@app.route("/face-recognition")
def face_recognition():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("face_recognition.html", name=session["name"])


@app.route("/live-recognition")
def live_recognition():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("live_recognition.html", name=session["name"])


@app.route("/about")
def about():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("about.html", name=session["name"])

# =========================================================
# API ROUTES (FACE RECOGNITION)
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle uploaded image prediction.
    """
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image uploaded"}), 400

    file = request.files["image"]
    bgr, error_payload, status_code = load_uploaded_image(
        file,
        invalid_error="Invalid image",
        unsupported_error="Unsupported file type. Please upload JPG, JPEG, PNG, BMP, or WEBP.",
    )
    if error_payload:
        return jsonify(error_payload), status_code

    result = predict_from_bgr(bgr)

    return jsonify({"ok": True, **result})


@app.route("/predict_webcam", methods=["POST"])
def predict_webcam():
    """
    Handle live webcam frame prediction.
    """
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No frame received"}), 400

    file = request.files["image"]
    bgr, error_payload, status_code = load_uploaded_image(
        file,
        invalid_error="Invalid frame",
        unsupported_error="Unsupported frame type. Please send JPG, JPEG, PNG, BMP, or WEBP.",
    )
    if error_payload:
        return jsonify(error_payload), status_code

    result = predict_from_bgr(bgr)

    return jsonify({"ok": True, **result})

# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":
    """
    Start Flask server
    """
    app.run(host="0.0.0.0", port=5000, debug=True)
