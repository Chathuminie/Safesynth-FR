import os                      # Used for file and folder operations
import cv2                     # OpenCV library for image processing and face detection
import numpy as np             # NumPy library for numerical operations

# ---------------- CONFIG ----------------

# Input folder containing raw synthetic images
IN_DIR = r"dataset\synthetic_raw"

# Output folder for processed face images
OUT_DIR = r"dataset\processed_faces"

# Final image size for model training
IMG_SIZE = 224

# Extra margin added around detected face
MARGIN = 0.30

# ----------------------------------------

# Create output folder if it does not exist
os.makedirs(OUT_DIR, exist_ok=True)

# Load Haar Cascade face detection model
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cascade_path)

# Function to crop image into a centered square
def center_crop_square(img):

    # Get image height and width
    h, w = img.shape[:2]

    # Find smallest dimension
    s = min(h, w)

    # Calculate crop starting points
    y = (h - s) // 2
    x = (w - s) // 2

    # Return square cropped image
    return img[y:y+s, x:x+s]

# Function to crop face with extra margin around it
def crop_with_margin(img, x, y, w, h, margin=0.30):

    # Get image dimensions
    H, W = img.shape[:2]

    # Calculate margin size
    mx = int(w * margin)
    my = int(h * margin)

    # Calculate crop coordinates safely within image boundaries
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(W, x + w + mx)
    y2 = min(H, y + h + my)

    # Crop image
    crop = img[y1:y2, x1:x2]

    # Return cropped image if valid
    return crop if crop.size else None

# Function to detect the largest face in an image
def detect_largest_face(bgr):

    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,      # Controls image scaling
        minNeighbors=5,       # Controls detection quality
        minSize=(80, 80)      # Minimum face size
    )

    # If no faces detected, return None
    if len(faces) == 0:
        return None

    # Select the largest detected face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

    # Crop face with margin
    return crop_with_margin(bgr, x, y, w, h, margin=MARGIN)

# Counters for statistics
used_det = 0          # Number of successful face detections
used_fallback = 0     # Number of fallback center crops
saved = 0             # Number of saved images

# Get all image files from input folder
files = [
    f for f in os.listdir(IN_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

# Sort files alphabetically
files.sort()

# Process each image one by one
for i, fname in enumerate(files):

    # Create full image path
    path = os.path.join(IN_DIR, fname)

    # Read image
    bgr = cv2.imread(path)

    # Skip invalid images
    if bgr is None:
        continue

    # Detect largest face
    face = detect_largest_face(bgr)

    # If no face detected
    if face is None:

        # Increase fallback counter
        used_fallback += 1

        # Use center crop as fallback
        face = center_crop_square(bgr)

    else:
        # Increase successful detection counter
        used_det += 1

        # Apply square crop to detected face
        face = center_crop_square(face)

    # Resize image to model input size
    face = cv2.resize(
        face,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # Output filename
    out_name = f"img_{i:04d}.png"

    # Full output path
    out_path = os.path.join(OUT_DIR, out_name)

    # Save processed image
    cv2.imwrite(out_path, face)

    # Increase saved counter
    saved += 1

# Print processing summary
print("✅ Done.")
print("Total input:", len(files))
print("Saved:", saved)
print("Detections used:", used_det)
print("Fallback center-crops:", used_fallback)
print("Output folder:", OUT_DIR)
```
