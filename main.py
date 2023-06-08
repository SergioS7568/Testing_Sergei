import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from keras.models import load_model

# Load your trained model and define class names
model = load_model('saved_model/mdl_wts2.hdf5')
class_names = ['aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 'angel', 'ant', 'anvil', 'apple', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball bat', 'basket', 'basketball', 'bat', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'butterfly', 'chandelier', 'church', 'clock', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crocodile', 'dog', 'dragon', 'garden hose', 'giraffe', 'goatee', 'grass', 'guitar', 'hamburger', 'hammer', 'hat', 'headphones', 'hospital', 'hot air balloon', 'hot dog', 'hourglass', 'mailbox', 'map', 'marker', 'microwave', 'monkey', 'owl', 'paintbrush', 'paint can', 'palm tree', 'paper clip', 'parachute', 'parrot', 'passport', 'power outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'remote control', 'rhinoceros', 'rifle', 'river', 'roller coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'stairs', 'star', 'steak', 'string bean', 'The Eiffel Tower', 'toothpaste', 'tornado', 'tractor', 'traffic light', 'train', 'violin', 'washing machine', 'watermelon', 'waterslide', 'whale', 'wine bottle']
# Add your class names

# Create Streamlit app
st.title('Image Classification')
print(model.input_shape)
# File uploader widget
uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

# Process the uploaded image
if uploaded_file is not None:
    img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image to 28x28 or any desired size
    img = cv2.resize(img, (28, 28))
  
    # Preprocess the image
    #img = preprocess_image(img)  # Modify this function based on your preprocessing requirements
    img = img.reshape(28, 28)
   

    # Perform the prediction
    pred = model.predict(np.expand_dims(img, axis=0))[0]
    ind = (-pred).argsort()[:5]
    top_classes = [class_names[i] for i in ind]
    top_probabilities = pred[ind]

    # Display the results
    st.write('Top 5 Predicted Classes:')
    for i in range(len(top_classes)):
        st.write(f'{top_classes[i]}: {top_probabilities[i]*100:.2f}%')

    # Display the uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
