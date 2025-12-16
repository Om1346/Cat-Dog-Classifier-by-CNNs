import os.path

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
from tensorflow.keras.applications.efficientnet import preprocess_input

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Cat/Dog Classifier 🐱🐶")

IMG_SIZE = 224
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/cat_dog_classifier.keras"

FILE_ID = "1Wg1JQTbnFsFFTB0dKFSZkWDAJOJkgQ8o"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        with st.spinner("Downloading model..."):
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Load Labels
class_names = ["Cat","Dog"]

# Image Preprocess
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# UI
st.title("Cat Dog Classifier")
st.write("Upload an image and let's see is the model mews or barks!")

uploaded_file = st.file_uploader("Choose an image.....",type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image,caption="Uploaded Image",use_column_width=True)

    if st.button("Predict"):
        img = preprocess_image(image)
        preds = model.predict(img)[0]

        confidence = np.max(preds) * 100
        predicted_class = class_names[np.argmax(preds)]

        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")

