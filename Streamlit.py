import streamlit as st
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf  # Import TensorFlow
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import io

# Constants
MODEL_PATH = r"C:\Users\yranu\Downloads\MIA project\Dataset for Fetus Framework\fetal_model.h5"  # Update this path if needed

# Load your trained model
@st.cache_data
def load_trained_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_trained_model()

# Preprocess the uploaded image
def preprocess_image(image, target_size=(224, 224)):
    # Convert image to RGB if it's grayscale
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


# Predict and draw bounding boxes
def predict_and_draw_boxes(image):
    preprocessed_image = preprocess_image(image)
    predicted_boxes = model.predict(preprocessed_image)[0]
    
    # Convert PIL Image to numpy array
    image_np = np.array(image)

    # Draw bounding boxes
    for i in range(0, len(predicted_boxes), 4):
        x_min, y_min, x_max, y_max = predicted_boxes[i:i+4].astype(int)
        cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    return image_np

# Streamlit UI
st.title("Fetal Feature Detection")
st.header("Upload an image for analysis")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        with st.spinner('Processing...'):
            annotated_image = predict_and_draw_boxes(image)
            st.image(annotated_image, caption='Processed Image with Bounding Boxes', use_column_width=True)
