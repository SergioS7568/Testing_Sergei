
import os 
import cv2
import numpy as np
import streamlit as st
st.title("Welcome to My Home")

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input



loaded_model = tf.keras.models.load_model('saved_model/mdl_wt.hdf5')
### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")

classes = ['aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 'angel', 'ant', 'anvil', 'apple', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball bat', 'basket', 'basketball', 'bat', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'butterfly', 'chandelier', 'church', 'clock', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crocodile', 'dog', 'dragon', 'garden hose', 'giraffe', 'goatee', 'grass', 'guitar', 'hamburger', 'hammer', 'hat', 'headphones', 'hospital', 'hot air balloon', 'hot dog', 'hourglass', 'mailbox', 'map', 'marker', 'microwave', 'monkey', 'owl', 'paintbrush', 'paint can', 'palm tree', 'paper clip', 'parachute', 'parrot', 'passport', 'power outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'remote control', 'rhinoceros', 'rifle', 'river', 'roller coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'stairs', 'star', 'steak', 'string bean', 'The Eiffel Tower', 'toothpaste', 'tornado', 'tractor', 'traffic light', 'train', 'violin', 'washing machine', 'watermelon', 'waterslide', 'whale', 'wine bottl']


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(28, 28))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")
    Genrate_pred = st.button("Generate Prediction")
    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]



    if Genrate_pred:
        print("hello")
        prediction = loaded_model(img_reshape.reshape(-1, 28, 28, 1))
        pred = np.argmax(prediction, axis=1)
        print(pred)
       


