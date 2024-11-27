import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('blood_group_model.h5')

# Define blood group labels
blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# App title
st.title("Blood Group Detector")

# File uploader for fingerprint image
uploaded_file = st.file_uploader("Upload Fingerprint Image", type=["jpg", "png", "bmp"])

if uploaded_file:
    # Preprocess the uploaded image
    img = Image.open(uploaded_file).resize((224, 224)).convert("RGB")
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, 224, 224, 3)  # Add batch dimension

    # Predict the blood group
    prediction = model.predict(img_array)

    # Get the class with the highest probability
    predicted_class = np.argmax(prediction[0])
    predicted_blood_group = blood_groups[predicted_class]

    # Display the result
    st.write("Predicted Blood Group:", predicted_blood_group)
