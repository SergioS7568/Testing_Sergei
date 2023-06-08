import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from keras.models import load_model

# Load your trained model and define class names
model = load_model('saved_model/kerasv3.h5')
class_names = ['airplane', 'clock', 'ambulance', 'anvil', 'arm', 'backpack', 'bandage', 'barn', 'baseball', 'baseball bat', 'basket', 'basketball', 'bathtub', 'beach', 'belt', 'bicycle', 'binoculars', 'book', 'bowtie', 'bracelet', 'brain', 'bridge', 'broom', 'bucket', 'bulldozer', 'bus', 'cactus', 'calculator', 'calendar', 'camera', 'campfire', 'candle', 'canoe', 'car', 'castle', 'cat', 'cello', 'cell phone', 'chair', 'church', 'clarinet', 'clock', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'crayon', 'crown', 'cruise ship', 'diamond', 'dishwasher', 'dog', 'dresser', 'drill', 'drums', 'dumbbell', 'ear', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'flashlight', 'floor lamp', 'garden', 'gaden hose', 'guitar', 'hammer', 'harp', 'headphones', 'helicopter', 'hockey stick', 'horse', 'hospital', 'hourglass', 'house', 'jacket', 'keyboard', 'lantern', 'laptop', 'light bulb', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'microphone', 'microwave', 'moon', 'mosquito', 'mountain', 'mouse', 'mug', 'parachute', 'passport', 'pencil', 'piano', 'postcard', 'purse', 'radio', 'rake', 'remote control', 'sailboat', 'saw', 'school bus', 'scissors', 'screwdriver', 'snorkel', 'snowman', 'stereo', 'stove', 'stethoscope', 'spryinge', 'telephone', 'tennis racquet', 'tractor', 'trombone', 'trumpet', 'windmill']
# Add your class names

# Create Streamlit app
st.title('Image Classification')
#print(model.input_shape)
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
    def preprocess_img(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        normalized = gray / 255.0
        reshaped = normalized.reshape(28,28,1)
        
        return reshaped
    
    img = preprocess_img(img)
    
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
    
