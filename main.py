import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


st.title('Green stem Classifier')
file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
class_btn = st.button("Classify")
if file_uploaded is not None:    
    image = Image.open(file_uploaded)
    st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                # plt.imshow(image)
                # plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                
