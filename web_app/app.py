import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Image Forgery Detection",
    layout="centered"
)

MODEL_PATH = "../models/forgery_model.h5"
IMG_SIZE = 224
CLASS_NAMES = ["Authentic", "Tampered"]

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# -------------------------
# IMAGE PREPROCESSING
# -------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -------------------------
# UI
# -------------------------
st.title("🕵️ Image Forgery Detection System")
st.write("Upload an image to check whether it is **Authentic** or **Tampered**.")

uploaded_file = st.file_uploader(
    "📂 Upload Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------
# PREDICTION
# -------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_image = preprocess_image(image)

    predictions = model.predict(input_image)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    st.subheader("🔍 Prediction Result")

    if predicted_class == 0:
        st.success("✅ AUTHENTIC IMAGE")
    else:
        st.error("🚨 TAMPERED IMAGE")

    st.metric("Confidence", f"{confidence * 100:.2f}%")

    # -------------------------
    # DEBUG INFO (VERY IMPORTANT)
    # -------------------------
    with st.expander("🔎 Model Output (Debug)"):
        st.write(f"Authentic probability: {predictions[0][0]:.4f}")
        st.write(f"Tampered probability: {predictions[0][1]:.4f}")
