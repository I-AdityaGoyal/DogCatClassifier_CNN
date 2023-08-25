import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the trained model
model = tf.keras.models.load_model('trained_model.h5')

# Define labels for binary classification
class_labels = ['Cat', 'Dog']

# Streamlit app
st.title('Dog and Cat Classification Project')
st.markdown("By Aditya Goyal")
st.write('Upload an image and I will predict whether it\'s a dog or a cat!')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    test_img = cv2.imdecode(file_bytes, 1)
    
    # Preprocess the image
    test_img = cv2.resize(test_img, (150, 150))
    test_input = test_img.reshape((1, 150, 150, 3)) / 255.0
    
    # Make prediction using the model
    prediction = model.predict(test_input)
    predicted_class = class_labels[int(prediction[0][0] > 0.5)]
    
    # Display the uploaded image and prediction
    st.image(test_img, caption=f'Uploaded Image: {predicted_class}', use_column_width=True)
    st.write(f'Prediction: {predicted_class}')
    
    # Display the image using Matplotlib
    plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Prediction: {predicted_class}')
    plt.axis('off')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
