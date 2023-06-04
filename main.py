
import os 
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
st.title("Welcome to My Home")

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input





loaded_model = tf.keras.models.load_model('saved_model/mdl_wts2.hdf5')
### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")

classes = ['aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 'angel', 'ant', 'anvil', 'apple', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball bat', 'basket', 'basketball', 'bat', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'butterfly', 'chandelier', 'church', 'clock', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crocodile', 'dog', 'dragon', 'garden hose', 'giraffe', 'goatee', 'grass', 'guitar', 'hamburger', 'hammer', 'hat', 'headphones', 'hospital', 'hot air balloon', 'hot dog', 'hourglass', 'mailbox', 'map', 'marker', 'microwave', 'monkey', 'owl', 'paintbrush', 'paint can', 'palm tree', 'paper clip', 'parachute', 'parrot', 'passport', 'power outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'remote control', 'rhinoceros', 'rifle', 'river', 'roller coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'stairs', 'star', 'steak', 'string bean', 'The Eiffel Tower', 'toothpaste', 'tornado', 'tractor', 'traffic light', 'train', 'violin', 'washing machine', 'watermelon', 'waterslide', 'whale', 'wine bottle']


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
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = np.asarray(file_bytes)
    img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    img = img / 255
    resized = cv2.resize(img,(28, 28))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")
    Genrate_pred = st.button("Generate Prediction")
    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]




    if Genrate_pred:
                        
            img_reshape = tf.keras.utils.normalize(img_reshape, axis=1)  # x becomes a tensor
            img_reshape = img_reshape.astype('float32')  # dtype of x changed back to numpy float32
            print(img_reshape.dtype)           # prints correctly float32 (notice that its not tf.float32)
            prediction = loaded_model.predict([img_reshape]) 
            #img_reshape = img_reshape.astype('float32')        
            #img_reshape = img_reshape.reshape(-1, 28, 28, 1)
            #prediction = loaded_model.predict(img_reshape).argmax()
            #st.title("Predicted Label for the image is {}".format(map_dict [prediction]))
            #pred = loaded_model.predict(img_reshape).argsort()[:5] 
            #print(prediction)
            #pred_class = classes[pred.argmax()]
            #plt.imshow(img_reshape)
            #plt.title(pred_class)
            #plt.axis(False);
            
            
            
        #img = img_reshape
        #plt.imshow(img.squeeze()) 
        #plt.imshow(img_reshape) 
        #np.expand_dims(img_reshape, axis=0)
        #predictions = loaded_model.predict(img_reshape)
        #string = "this is a prediction:"+classes[np.argmax(predictions)]
        #print(string)

            
            
            
        #pred = loaded_model.predict(img_reshape) 
        #pred = tf.nn.tanh(pred, name ='tanh')
        #prediction = loaded_model.predict(img_reshape[:1])
        #print("prediction shape:", prediction.shape)
        #tensorshow = tf.
        #print(pred)
        #ind = (-pred).argsort()[:5]
        #print(ind)
        #latex = ind
        #prod = loaded_model.predict(pred) 
        #ond = (-prod).argsort()[:5]
        #print(ond)
        #latex = [classes[x] for x in ind]
        #print(latex)
        
        #prod = np.argmax(prediction, axis=1)
        #print("hello")
        #pred = np.argmax(prediction)
        #prediction = loaded_model.predict(img_reshape).argmax()
        #print(loaded_model.predict(img_reshape).argmax())
        #prediction = loaded_model.predict(img_reshape)
        #prediction = np.argmax(prediction, axis=1)
        #st.title("Predicted Label for the image is {}".format(map_dict [prediction]))
        #print(prediction)
        #pred = np.exp(prediction[:,3])
        #print(pred)

        
       


