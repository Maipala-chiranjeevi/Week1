import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image

# Load the saved model
model = load_model('waste_classification_model.h5')

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit UI
st.title('Waste Classification')
st.write("Upload an image of waste for classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an image
    img = Image.open(uploaded_file)
    
    # Preprocess the image for prediction
    preprocessed_image = preprocess_image(img)

    # Make prediction
    prediction = model.predict(preprocessed_image)
    class_names = ['Organic', 'Recyclable']
    predicted_class = class_names[np.argmax(prediction)]

    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write(f"Prediction: {predicted_class} with confidence {np.max(prediction):.2f}")
