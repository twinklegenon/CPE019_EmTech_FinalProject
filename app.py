import streamlit as st
import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

@st.cache_resource

def load_model():
  model=tf.keras.models.load_model('rps_classifier.h5')
  return model

model = load_model()
classes = {0: 'paper', 1: 'rock', 2: 'scissors'}

st.write("# Rock-Paper-Scissors Classifier")
st.write("###### By Genon, Twinkle S. & Murao, Christian Ivan P.")
st.write("###### CPE019 - CPE32S3  Emerging Technologies 2 in CpE")

file = st.file_uploader("Rock, Paper or Scissors: Choose Your Champion!", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    image = image / 255.0
    image_reshape = np.reshape(image, (1, 256, 256, 3))
    prediction = model.predict(image_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    try:
        image = Image.open(file) if file else None
        if image:
            st.image(image, use_column_width=True)
            prediction = import_and_predict(image, model)
            class_names = classes
            string = f"It is a {class_names[np.argmax(prediction)]}!"
            st.success(string)
        else:
            st.text("Invalid file. Please upload a valid image file.")
    except Exception as e:
        st.text("Error occurred while processing the image.")
        st.text(str(e))
        
        st.write("---")

# Create two columns
col1, col2 = st.beta_columns([1, 3])

# Column 1: Display the sample image
with col1:
    st.image("sample_image.png", use_column_width=True)

# Column 2: Add the disclosure text
with col2:
    st.write("### Pose of Rock, Paper, and Scissors")
    st.write("To get accurate predictions, please ensure that the poses of rock, paper, and scissors in your uploaded image match the sample image above. The hand gestures should be clearly visible and follow the standard poses for each category.")
