import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Image Forgery Detection",
    page_icon="🕵️",
    layout="centered"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = 224

CLASS_NAMES = [
    "Authentic",
    "Tampered"
]


# ============================================================
# MODEL PATH
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


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file not found:\n{MODEL_PATH}"
        )
        st.stop()

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


model = load_model()


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
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def find_last_conv_layer(model):

    last_conv_layer = None

    for layer in model.layers:

        if isinstance(
            layer,
            (
                tf.keras.layers.Conv2D,
                tf.keras.layers.DepthwiseConv2D
            )
        ):
            last_conv_layer = layer

    return last_conv_layer


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    image,
    model,
    conv_layer
):

    if conv_layer is None:
        return None

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            conv_layer.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = (
            grad_model(image)
        )

        tampered_score = predictions[:, 0]

    gradients = tape.gradient(
        tampered_score,
        conv_outputs
    )

    if gradients is None:
        return None

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]

    pooled_gradients = pooled_gradients[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    maximum = tf.reduce_max(
        heatmap
    )

    if maximum > 0:

        heatmap = (
            heatmap / maximum
        )

    return heatmap.numpy()


# ============================================================
# CREATE HEATMAP
# ============================================================

def create_heatmap_overlay(
    original_image,
    heatmap
):

    original = np.array(
        original_image.convert("RGB")
    )

    height, width = original.shape[:2]

    heatmap = cv2.resize(
        heatmap,
        (width, height)
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        0.60,
        heatmap_color,
        0.40,
        0
    )

    return overlay


# ============================================================
# HIGHLIGHT SUSPICIOUS REGIONS
# ============================================================

def highlight_regions(
    original_image,
    heatmap,
    threshold=0.55
):

    original = np.array(
        original_image.convert("RGB")
    )

    height, width = original.shape[:2]

    heatmap = cv2.resize(
        heatmap,
        (width, height)
    )

    mask = np.uint8(
        heatmap >= threshold
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

    result = original.copy()

    region_count = 0

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            3
        )

        cv2.putText(
            result,
            "Suspicious Region",
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        region_count += 1

    return result, region_count


# ============================================================
# USER INTERFACE
# ============================================================

st.title(
    "🕵️ Image Forgery Detection System"
)

st.write(
    "Upload an image to check whether it is "
    "**Authentic** or **Tampered**."
)


# ============================================================
# FILE UPLOADER
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
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

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
    # PREDICTION
    # --------------------------------------------------------

    predictions = model.predict(
        input_image,
        verbose=0
    )

    tampered_probability = float(
        predictions[0][0]
    )

    authentic_probability = (
        1.0 - tampered_probability
    )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if tampered_probability >= 0.5:

        predicted_class = 1

        predicted_label = "Tampered"

        confidence = (
            tampered_probability
        )

    else:

        predicted_class = 0

        predicted_label = "Authentic"

        confidence = (
            authentic_probability
        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.subheader(
        "🔍 Prediction Result"
    )

    if predicted_class == 1:

        st.error(
            "🚨 TAMPERED IMAGE"
        )

    else:

        st.success(
            "✅ AUTHENTIC IMAGE"
        )

    st.write(
        f"**Prediction:** {predicted_label}"
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

    # ========================================================
    # LOCALIZATION
    # ========================================================

    if predicted_class == 1:

        st.subheader(
            "📍 Forgery Localization"
        )

        st.info(
            "The heatmap shows regions that strongly "
            "influenced the model's Tampered prediction."
        )

        conv_layer = find_last_conv_layer(
            model
        )

        if conv_layer is None:

            st.warning(
                "No convolutional layer was found. "
                "Grad-CAM localization cannot be generated."
            )

        else:

            try:

                heatmap = generate_gradcam(
                    input_image,
                    model,
                    conv_layer
                )

                if heatmap is None:

                    st.warning(
                        "Unable to generate the localization heatmap."
                    )

                else:

                    # ------------------------------------------------
                    # HEATMAP OVERLAY
                    # ------------------------------------------------

                    heatmap_overlay = (
                        create_heatmap_overlay(
                            image,
                            heatmap
                        )
                    )

                    # ------------------------------------------------
                    # HIGHLIGHTED REGIONS
                    # ------------------------------------------------

                    highlighted_image, region_count = (
                        highlight_regions(
                            image,
                            heatmap
                        )
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.image(
                            heatmap_overlay,
                            caption="🔥 Grad-CAM Heatmap",
                            use_container_width=True
                        )

                    with col2:

                        st.image(
                            highlighted_image,
                            caption="📍 Highlighted Regions",
                            use_container_width=True
                        )

                    if region_count > 0:

                        st.success(
                            f"Detected {region_count} "
                            f"suspicious region(s)."
                        )

                    else:

                        st.warning(
                            "No strong suspicious region "
                            "was detected by the localization threshold."
                        )

                    # ------------------------------------------------
                    # RAW HEATMAP
                    # ------------------------------------------------

                    with st.expander(
                        "🔥 View Raw Grad-CAM"
                    ):

                        st.image(
                            heatmap,
                            caption="Grad-CAM Activation Map",
                            use_container_width=True
                        )

            except Exception as error:

                st.error(
                    "Localization could not be generated."
                )

                st.code(
                    str(error)
                )

    else:

        st.info(
            "No suspicious-region localization is shown "
            "because the image was classified as Authentic."
        )

    # ========================================================
    # DEBUG INFORMATION
    # ========================================================

    with st.expander(
        "🔎 Model Output (Debug)"
    ):

        st.write(
            "Raw Predictions:",
            predictions
        )

        st.write(
            "Prediction Shape:",
            predictions.shape
        )

        st.write(
            "Prediction Type:",
            type(predictions)
        )

        st.write(
            "Tampered Probability:",
            tampered_probability
        )

        st.write(
            "Authentic Probability:",
            authentic_probability
        )

        if conv_layer is not None:

            st.write(
                "Grad-CAM Layer:",
                conv_layer.name
            )