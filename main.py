import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

st.title('Green stem Classifier')
file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
class_btn = st.button("Classify")


loaded_model = tf.keras.models.load_model("saved_model/mdl.wts2")
    
def predictor(image):
    classifier_model = "keras.h5"
    model = loaded_model(classifier_model)
    test_image = image.resize((200,200))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    print(prediction)
    print(scores)
    return result


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
                predictions = predictor(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                

                
