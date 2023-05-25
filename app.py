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
st.write("### By Genon, Twinkle S. & Murao, Christian Ivan P.")

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
