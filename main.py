
import os 
import cv2
import numpy as np
import streamlit as st
st.title("Welcome to My Home")

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input



loaded_model = tf.keras.models.load_model('saved_model/mdl_wts.hdf5')
### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")


map_dict = {0: 'dog',
            1: 'horse',
            2: 'elephant',
            3: 'butterfly',
            4: 'chicken',
            5: 'cat',
            6: 'cow',
            7: 'sheep',
            8: 'spider',
            9: 'squirrel'}


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224, 224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")
    Genrate_pred = st.button("Generate Prediction")
    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]



    if Genrate_pred:
        img_reshape = img_reshape.reshape(-1, 224, 224, 1)
        #prediction = loaded_model.predict(img_reshape).argmax()
        #print(loaded_model.predict(img_reshape).argmax())
        prediction = loaded_model.predict(img_reshape).argmax()
        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))
        #print(prediction)
        #pred = np.exp(prediction[:,3])
        #print(pred)
        
        #prod = np.argmax(prediction, axis=1)
        print("hello")
        #pred = np.argmax(prediction)

        
       


