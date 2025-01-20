import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('classification_model.h5')

labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

st.sidebar.title('Image Classification')
st.sidebar.write("""
    Choose an image from the following categories:
    - Airplane
    - Bird
    - Ship
    - Truck
    - Automobile
    - Dog
    - Horse
    - Cat
    - Deer
    - Frog
    
    Model will predict which category the image belongs to.
""")

st.title('Let\'s Classify an Image')
st.markdown('### Upload an image and see the model prediction below.')

uploaded_file = st.file_uploader("Upload Image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    image = image.convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = image.resize((32, 32)) 
    img_array = np.array(image) / 255.0 
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions) 

    st.write(f"### Prediction: {labels[predicted_class]}")