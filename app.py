import streamlit as st
import tensorflow as tf

@st.cache_resource

def load_model():
  model=tf.keras.models.load_model('rps_classifier.h5')
  return model

model = load_model()
classes = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}

st.write("# Rock-Paper-Scissors Classifier")

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
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = classes
    string = f"It is a {class_names[np.argmax(prediction)]}!"
    st.success(string)
