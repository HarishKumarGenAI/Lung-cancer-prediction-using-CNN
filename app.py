import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image

# ============================================
# PROJECT PATH
# ============================================

BASE_DIR = r"E:\Lung cancer_prediction"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "lung_cancer_model.keras"

)

CLASS_PATH = os.path.join(
    BASE_DIR,
    "class_indices.json"
)
print("MODEL_PATH =", MODEL_PATH)
print("FILE EXISTS =", os.path.exists(MODEL_PATH))

# ============================================
# PAGE SETTINGS
# ============================================

st.set_page_config(
    page_title="Lung Cancer Prediction",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 Lung Cancer Prediction Using CNN")

st.write(
    "Upload a lung histopathology image and the "
    "trained CNN model will predict the class."
)

# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH

    )

    return model

# ============================================
# LOAD CLASS NAMES
# ============================================

@st.cache_data
def load_classes():

    with open(CLASS_PATH, "r") as f:
        class_names = json.load(f)

    return class_names

# ============================================
# CHECK FILES
# ============================================

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found")
    st.stop()

if not os.path.exists(CLASS_PATH):
    st.error("class_indices.json not found")
    st.stop()

model = load_model()
class_names = load_classes()

# ============================================
# FILE UPLOAD
# ============================================

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# ============================================
# PREDICTION
# ============================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    image = image.resize((128, 128))

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array
    )

    predictions = predictions[0]

    predicted_index = np.argmax(
        predictions
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = (
        predictions[predicted_index]
        * 100
    )

    st.success(
        f"Prediction : {predicted_class}"
    )

    st.write(
        f"Confidence : {confidence:.2f}%"
    )

    st.subheader("Class Scores")

    for label, score in zip(
        class_names,
        predictions
    ):

        st.write(
            f"{label} : {score*100:.2f}%"
        )

    st.warning(
        "Educational demonstration only. "
        "Not for medical diagnosis."
    )