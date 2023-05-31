
import os 

import numpy as np
import streamlit as st
st.title("Welcome to My Home")

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input


from keras.models import load_model
new_model = tf.keras.models.load_model('saved_model/mdl_ws.hdf5')
### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")



