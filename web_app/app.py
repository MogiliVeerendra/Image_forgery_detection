import os

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Image Forgery Detection",
    page_icon="🕵️",
    layout="centered"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "forgery_model.h5"
)

IMG_SIZE = 224
LAST_CONV_LAYER = "conv5_block3_out"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


model = load_model()


# ============================================================
# GRAD-CAM MODEL
# ============================================================

@st.cache_resource
def load_gradcam_model():

    return tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(LAST_CONV_LAYER).output,
            model.output
        ]
    )


gradcam_model = load_gradcam_model()


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image_array = np.array(image)

    image_array = cv2.resize(
        image_array,
        (IMG_SIZE, IMG_SIZE)
    )

    image_array = image_array.astype(
        np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# PREDICTION
# ============================================================

def predict_image(input_image):

    predictions = model.predict(
        input_image,
        verbose=0
    )

    # Single sigmoid output
    tampered_probability = float(
        predictions[0][0]
    )

    authentic_probability = (
        1.0 - tampered_probability
    )

    if tampered_probability >= 0.50:

        predicted_class = 1
        predicted_label = "Tampered"
        confidence = tampered_probability

    else:

        predicted_class = 0
        predicted_label = "Authentic"
        confidence = authentic_probability

    return (
        predicted_class,
        predicted_label,
        confidence,
        authentic_probability,
        tampered_probability,
        predictions
    )


# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam_heatmap(input_image):

    with tf.GradientTape() as tape:

        conv_outputs, predictions = gradcam_model(
            input_image,
            training=False
        )

        tampered_score = predictions[:, 0]

    grads = tape.gradient(
        tampered_score,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = heatmap / (
        max_value + 1e-8
    )

    return heatmap.numpy()


# ============================================================
# CREATE VISUALIZATION
# ============================================================

def create_visualizations(
    image,
    heatmap,
    threshold
):

    original = np.array(
        image.convert("RGB")
    )

    height, width = original.shape[:2]

    heatmap_resized = cv2.resize(
        heatmap,
        (width, height)
    )

    heatmap_uint8 = np.uint8(
        heatmap_resized * 255
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        0.55,
        heatmap_color,
        0.45,
        0
    )

    # Create high-attention mask
    mask = (
        heatmap_resized >= threshold
    ).astype(
        np.uint8
    ) * 255

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    highlighted = original.copy()

    regions = 0

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < 200:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        cv2.rectangle(
            highlighted,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            3
        )

        cv2.putText(
            highlighted,
            "High Attention",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        regions += 1

    return (
        heatmap_color,
        overlay,
        highlighted,
        regions
    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "🕵️ Image Forgery Detection System"
)

st.write(
    "Upload an image to check whether it is "
    "**Authentic** or **Tampered**."
)

st.info(
    "Grad-CAM shows regions that influenced "
    "the model's prediction."
)


# ============================================================
# UPLOAD IMAGE
# ============================================================

uploaded_file = st.file_uploader(
    "📂 Upload Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PROCESS
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader(
        "🖼️ Uploaded Image"
    )

    st.image(
        image,
        caption="Original Image",
        use_container_width=True
    )

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    input_image = preprocess_image(
        image
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    (
        predicted_class,
        predicted_label,
        confidence,
        authentic_probability,
        tampered_probability,
        predictions
    ) = predict_image(
        input_image
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.subheader(
        "🔍 Prediction Result"
    )

    if predicted_class == 0:

        st.success(
            "✅ AUTHENTIC IMAGE"
        )

    else:

        st.error(
            "🚨 TAMPERED IMAGE"
        )

    st.metric(
        "Confidence",
        f"{confidence * 100:.2f}%"
    )

    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    st.subheader(
        "📊 Prediction Probabilities"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Authentic",
            f"{authentic_probability * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Tampered",
            f"{tampered_probability * 100:.2f}%"
        )

    st.write(
        "Authentic"
    )

    st.progress(
        authentic_probability
    )

    st.write(
        "Tampered"
    )

    st.progress(
        tampered_probability
    )

    # ========================================================
    # LOCALIZATION
    # ========================================================

    st.subheader(
        "📍 Model Attention / Localization"
    )

    st.warning(
        "Grad-CAM highlights regions that influenced "
        "the prediction. These are not guaranteed "
        "to be the exact forged pixels."
    )

    threshold = st.slider(
        "Localization Threshold",
        0.30,
        0.90,
        0.60,
        0.05
    )

    # Generate Grad-CAM
    heatmap = make_gradcam_heatmap(
        input_image
    )

    (
        heatmap_color,
        overlay,
        highlighted,
        regions
    ) = create_visualizations(
        image,
        heatmap,
        threshold
    )

    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            heatmap_color,
            caption="Grad-CAM Heatmap",
            use_container_width=True
        )

    with col2:

        st.image(
            overlay,
            caption="Grad-CAM Overlay",
            use_container_width=True
        )

    # --------------------------------------------------------
    # HIGHLIGHTED REGIONS
    # --------------------------------------------------------

    st.subheader(
        "🔎 Highlighted High-Attention Regions"
    )

    if regions > 0:

        if predicted_class == 1:

            st.write(
                f"Found {regions} high-attention "
                "region(s). These may be suspicious "
                "regions."
            )

        else:

            st.write(
                f"Found {regions} high-attention "
                "region(s) that influenced the "
                "Authentic prediction."
            )

        st.image(
            highlighted,
            caption="Highlighted Regions",
            use_container_width=True
        )

    else:

        st.info(
            "No high-attention regions were found "
            "at the selected threshold."
        )

    # ========================================================
    # DEBUG
    # ========================================================

    with st.expander(
        "🔎 Model Output (Debug)"
    ):

        st.write(
            "Raw Model Output:",
            predictions
        )

        st.write(
            "Output Shape:",
            predictions.shape
        )

        st.write(
            "Output Type:",
            type(predictions)
        )

        st.write(
            "Authentic Probability:",
            authentic_probability
        )

        st.write(
            "Tampered Probability:",
            tampered_probability
        )

        st.write(
            "Classification Threshold:",
            0.50
        )

        st.write(
            "Grad-CAM Layer:",
            LAST_CONV_LAYER
        )

        st.write(
            "Localization Threshold:",
            threshold
        )

        st.write(
            "High-Attention Regions:",
            regions
        )