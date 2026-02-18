import os
import numpy as np
from tensorflow.keras.models import load_model
from src.preprocess import preprocess_image

# ===============================
# PROJECT ROOT
# ===============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "forgery_model.h5")
TEST_DIR = os.path.join(BASE_DIR, "datasets", "casia", "tampered")

print("MODEL PATH:", MODEL_PATH)
print("TEST DIR:", TEST_DIR)

# ===============================
# CHECKS
# ===============================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

# ===============================
# LOAD MODEL
# ===============================
model = load_model(MODEL_PATH)
print("✅ Model loaded")

# ===============================
# TEST IMAGES
# ===============================
image_files = [
    f for f in os.listdir(TEST_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

if not image_files:
    raise RuntimeError("No images found in test directory")

print(f"🔍 Testing {len(image_files)} images...\n")

for img_name in image_files[:10]:
    img_path = os.path.join(TEST_DIR, img_name)
    img = preprocess_image(img_path)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0][0]
    label = "FORGED" if pred > 0.5 else "AUTHENTIC"

    print(f"{img_name} → {label} ({pred:.3f})")
