import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras import preprocessing
import numpy as np

st.title('Green stem Classifier')
file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
class_btn = st.button("Classify")


loaded_model = tf.keras.models.load_model("saved_model/sopa.hdf5")
    
def predictor(image):
    model = loaded_model
    test_image = image.resize(-1,(28,28),1)
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    predictions = model.predict(test_image)
    scores = tf.nn.argmax(predictions[0])
    print(prediction)
    return result


if file_uploaded is not None:    
    image = Image.open(file_uploaded)
    st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                predictions = predictor(image)
    
                

                
