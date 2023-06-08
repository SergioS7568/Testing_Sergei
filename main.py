import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from keras.models import load_model

# Load your trained model and define class names
model = load_model('saved_model/mdl_wt.hdf5')
special_class_names = ['airplane', 'clock', 'ambulance', 'anvil', 'arm', 'backpack', 'bandage', 'barn', 'baseball', 'baseball bat', 'basket', 'basketball', 'bathtub', 'beach', 'belt', 'bicycle', 'binoculars', 'book', 'bowtie', 'bracelet', 'brain', 'bridge', 'broom', 'bucket', 'bulldozer', 'bus', 'cactus', 'calculator', 'calendar', 'camera', 'campfire', 'candle', 'canoe', 'car', 'castle', 'cat', 'cello', 'cell phone', 'chair', 'church', 'clarinet', 'clock', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'crayon', 'crown', 'cruise ship', 'diamond', 'dishwasher', 'dog', 'dresser', 'drill', 'drums', 'dumbbell', 'ear', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'flashlight', 'floor lamp', 'garden', 'gaden hose', 'guitar', 'hammer', 'harp', 'headphones', 'helicopter', 'hockey stick', 'horse', 'hospital', 'hourglass', 'house', 'jacket', 'keyboard', 'lantern', 'laptop', 'light bulb', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'microphone', 'microwave', 'moon', 'mosquito', 'mountain', 'mouse', 'mug', 'parachute', 'passport', 'pencil', 'piano', 'postcard', 'purse', 'radio', 'rake', 'remote control', 'sailboat', 'saw', 'school bus', 'scissors', 'screwdriver', 'snorkel', 'snowman', 'stereo', 'stove', 'stethoscope', 'spryinge', 'telephone', 'tennis racquet', 'tractor', 'trombone', 'trumpet', 'windmill']
class_names = ['aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 'angel', 'ant', 'anvil', 'apple', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball bat', 'basket', 'basketball', 'bat', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'butterfly', 'chandelier', 'church', 'clock', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crocodile', 'dog', 'dragon', 'garden hose', 'giraffe', 'goatee', 'grass', 'guitar', 'hamburger', 'hammer', 'hat', 'headphones', 'hospital', 'hot air balloon', 'hot dog', 'hourglass', 'mailbox', 'map', 'marker', 'microwave', 'monkey', 'owl', 'paintbrush', 'paint can', 'palm tree', 'paper clip', 'parachute', 'parrot', 'passport', 'power outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'remote control', 'rhinoceros', 'rifle', 'river', 'roller coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'stairs', 'star', 'steak', 'string bean', 'The Eiffel Tower', 'toothpaste', 'tornado', 'tractor', 'traffic light', 'train', 'violin', 'washing machine', 'watermelon', 'waterslide', 'whale', 'wine bottle']

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
        #COLOR_RGB2GRAY
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        normalized = gray /255.0
        reshaped = normalized.reshape(28,28,1).astype('float32')
        
        return reshaped
    
    img = preprocess_img(img)
    
    # Perform the prediction
    pred = model.predict(np.expand_dims(img, axis=0))[0]
    ind = (-pred).argsort()[:10]
    top_classes = [class_names[i] for i in ind]
    top_probabilities = pred[ind]

    # Display the results
    st.write('Top 5 Predicted Classes:')
    for i in range(len(top_classes)):
        st.write(f'{top_classes[i]}: {top_probabilities[i]*100:.2f}%')

    # Display the uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
