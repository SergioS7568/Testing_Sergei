
import os 
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow import keras
st.title("Welcome to My Home")

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras.utils import img_to_array


loaded_model = tf.keras.models.load_model('saved_model/mdl_cripple.hdf5')
### load file
uploaded_file = st.file_uploader("Choose a image file", type=['png', 'jpeg', 'jpg'])

classes = ['aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 'angel', 'ant', 'anvil', 'apple', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball bat', 'basket', 'basketball', 'bat', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'butterfly', 'chandelier', 'church', 'clock', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crocodile', 'dog', 'dragon', 'garden hose', 'giraffe', 'goatee', 'grass', 'guitar', 'hamburger', 'hammer', 'hat', 'headphones', 'hospital', 'hot air balloon', 'hot dog', 'hourglass', 'mailbox', 'map', 'marker', 'microwave', 'monkey', 'owl', 'paintbrush', 'paint can', 'palm tree', 'paper clip', 'parachute', 'parrot', 'passport', 'power outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'remote control', 'rhinoceros', 'rifle', 'river', 'roller coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'stairs', 'star', 'steak', 'string bean', 'The Eiffel Tower', 'toothpaste', 'tornado', 'tractor', 'traffic light', 'train', 'violin', 'washing machine', 'watermelon', 'waterslide', 'whale', 'wine bottle']
selection_chosen = ['airplane', 'clock', 'ambulance', 'anvil', 'arm', 'backpack', 'bandage', 'barn', 'baseball', 'baseball bat', 'basket', 'basketball', 'bathtub', 'beach', 'belt', 'bicycle', 'binoculars', 'book', 'bowtie', 'bracelet', 'brain', 'bridge', 'broom', 'bucket', 'bulldozer', 'bus', 'cactus', 'calculator', 'calendar', 'camera', 'campfire', 'candle', 'canoe', 'car', 'castle', 'cat', 'cello', 'cell phone', 'chair', 'church', 'clarinet', 'clock', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'crayon', 'crown', 'cruise ship', 'diamond', 'dishwasher', 'dog', 'dresser', 'drill', 'drums', 'dumbbell', 'ear', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'flashlight', 'floor lamp', 'garden', 'garden hose', 'guitar', 'hammer', 'harp', 'headphones', 'helicopter', 'hockey stick', 'horse', 'hospital', 'hourglass', 'house', 'jacket', 'keyboard', 'lantern', 'laptop', 'light bulb', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'microphone', 'microwave', 'moon', 'mosquito', 'mountain', 'mouse', 'mug', 'parachute', 'passport', 'pencil', 'piano', 'postcard', 'purse', 'radio', 'rake', 'remote control', 'sailboat', 'saw', 'school bus', 'scissors', 'screwdriver', 'snorkel', 'snowman', 'stereo', 'stove', 'stethoscope', 'syring', 'telephone', 'tennis racquet', 'tractor', 'trombone', 'trumpet', 'windmill']

map_dict2 = {0: 'dog',
            1: 'horse',
            2: 'elephant',
            3: 'butterfly',
            4: 'chicken',
            5: 'cat',
            6: 'cow',
            7: 'sheep',
            8: 'spider',
            9: 'squirrel'}

map_dict = {0: ' aircraft carrier ',
1: 'airplane',
2: ' alarm clock ',
3: ' ambulance ',
4: ' angel ',
5: ' ant ',
6: ' anvil ',
7: ' apple ',
8: ' asparagus ',
9: ' axe ',
10: ' backpack ',
11: ' banana ',
12: ' bandage ',
13: ' barn ',
14: ' baseball ',
15: ' baseball bat ',
16: ' basket ',
17: ' basketball ',
18: ' bat ',
19: ' beard ',
20: ' bed ',
21: ' bee ',
22: ' belt ',
23: ' bench ',
24: ' bicycle ',
25: ' binoculars ',
26: ' bird ',
27: ' birthday cake ',
28: ' book ',
29: ' boomerang ',
30: ' bottlecap ',
31: ' bowtie ',
32: ' bracelet ',
33: ' brain ',
34: ' bread ',
35: ' bridge ',
36: ' broccoli ',
37: ' broom ',
38: ' bucket ',
39: ' bulldozer ',
40: ' bus ',
41: ' butterfly ',
42: ' chandelier ',
43: ' church ',
44: ' clock ',
45: ' coffee cup ',
46: ' compass ',
47: ' computer ',
48: ' cookie ',
49: ' cooler ',
50: ' couch ',
51: ' cow ',
52: ' crab ',
53: ' crocodile ',
54: ' dog ',
55: ' dragon ',
56: ' garden hose ',
57: ' giraffe ',
58: ' goatee ',
59: ' grass ',
60: ' guitar ',
61: ' hamburger ',
62: ' hammer ',
63: ' hat ',
64: ' headphones ',
65: ' hospital ',
66: ' hot air balloon ',
67: ' hot dog ',
68: ' hourglass ',
69: ' mailbox ',
70: ' map ',
71: ' marker ',
72: ' microwave ',
73: ' monkey ',
74: ' owl ',
75: ' paintbrush ',
76: ' paint can ',
77: ' palm tree ',
78: ' paper clip ',
79: ' parachute ',
80: ' parrot ',
81: ' passport ',
82: ' power outlet ',
83: ' purse ',
84: ' rabbit ',
85: ' raccoon ',
86: ' radio ',
87: ' remote control ',
88: ' rhinoceros ',
89: ' rifle ',
90: ' river ',
91: ' roller coaster ',
92: ' rollerskates ',
93: ' sailboat ',
94: ' sandwich ',
95: ' saw ',
96: ' saxophone ',
97: ' stairs ',
98: ' star ',
99: ' steak ',
100: ' string bean ',
101: ' The Eiffel Tower ',
102: ' toothpaste ',
103: ' tornado ',
104: ' tractor ',
105: ' traffic light ',
106: ' train ',
107: ' violin ',
108: ' washing machine ',
109: ' watermelon ',
110: ' waterslide ',
111: ' whale ',
112: ' wine bottle '}

if uploaded_file is not None:
    st.image(uploaded_file)
    # Convert the file to an opencv image.
    
    #file_bytes = file_bytes[:,:,0]
    #uploaded_file = st.file_uploader("Upload Image")
    #image = Image.open(uploaded_file)
    #st.image(uploaded_file, caption='Input', use_column_width=True)
    #img_array = np.array(image)
    #img= cv2.imwrite('out.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    #img = cv2.cvtColor(file_bytes, cv2.COLOR_BGR2RGB)
    Genrate_pred = st.button("Generate Prediction")
    #resized = mobilenet_v2_preprocess_input(resized)       
    #resized = mobilenet_v2_preprocess_input(resized)
    #img_reshape = resized[np.newaxis,...]
    #img = img.reshape(1,28, 28,3)
    #arr = np.array(img, dtype = 'float32')
    #arr = arr.reshape((28, 28))
    #arr = arr/255.0

    if Genrate_pred:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            #img_normalized = cv2.normalize(file_bytes, None, 0, 1.0, cv2.NORM_MINMAX)
            #img = img_normalized
            
            #plt.imshow(tf.squeeze(img[0])) 
            #ind = (-pred).argsort()[:5]
            #latex = [selection_chosen[x] for x in ind]
            #print(latex)
            #score = tf.nn.softmax(pred)
            #print(score)
            #img = cv2.resize(img, (28,28))
            
            
            img = file_bytes
            img = cv2.resize(img, (28,28))
            img_array = image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch
            #pred = loaded_model.predict(img_array)[0]
            #print((-pred).argsort()[:5])
            
            #sigmoid
            pred = loaded_model.predict(img_array)
            predictions = tf.nn.sigmoid(pred)
            print(predictions)
            
            
            #predictions = loaded_model.predict(img_array)  
            #print(predictions) 
            #score = tf.nn.softmax(pred)
            #print(score)
            #ind = (-pred).argsort()[:5]
            #latex = [selection_chosen[x] for x in ind]
            #print(latex)
  
            # Initiating a Tensorflow session
            #with tf.Session() as sess:
                        #print(pred)

            #print(predictions)
            #print(score)
            #print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(classes[np.argmax(score)], 100 * np.max(score)))
            #print('Image after resizing to 28x28')
            #st.title(classes[np.argmax(score)])
            #ax = plt.subplot(2, 4, i + 1)
            #plt.imshow(img)
            #ax.set_title(class_names[np.argmax(score)])
            #plt.axis("off")
            #print(predictions) 
            #print(score)
            
            #prediction = model.predict(arr)
            #print(prediction)
            #img_reshape = img_reshape.astype('float32')        
            #img_reshape = img_reshape.reshape(-1, 28, 28, 3)
            #prediction =loaded_model.predict(img_reshape)
            #prediction = prediction[0]
            #print(prediction.shape)
            #pred = np.exp(prediction)
            #print(pred)
            #pred = prediction.reshape(-1)
            #a = np.array(pred)
            #b = np.dstack([a])
            #print(pred)
            #a = np.array(prediction)
            #b = np.dstack([a])
            
            #print(b.shape)
            #print(pred[5])
            #print(pred.argmax())
            #st.title("Predicted Label for the image is {}".format(map_dict [prediction]))
            #pred = loaded_model.predict(img_reshape).argsort()[:5] 
            #print(prediction)
            #prediction= np.argmax(prediction)
            #print(prediction[0])
