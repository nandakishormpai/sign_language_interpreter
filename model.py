"""

Nanda Kishor M Pai

model.py : training a model for prediciting sign languages using CNN based neural network

"""

import cv2
import os
import numpy as np
from keras_squeezenet import SqueezeNet
from keras.optimizers import Adam
from tensorflow.python.keras.utils import np_utils
from keras.layers import Dropout, GlobalAveragePooling2D, Conv2D, Dense,MaxPooling2D, Flatten
from keras.models import Sequential
import tensorflow as tf
from tensorflow.keras import layers

#Inputting Train Data source
IMG_SAVE_PATH = "asl_alphabet_train"

CODES = {"nothing": 0}

def make_labels():
    alpha="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range (1,27):
        CODES[alpha[i-1]]=i

    CODES["del"]=27
    CODES["space"]=28
    return CODES




def code_conv(val):
    return CODES[val]

def make_model():
    make_labels()
    print(CODES)
    dataset = []
    for directory in os.listdir(IMG_SAVE_PATH):
        path = os.path.join(IMG_SAVE_PATH, directory)
        #skip if path doesn't exist
        if not os.path.isdir(path):
            continue
        for item in os.listdir(path):
            #resize the image after inputting and append to our dataset
            img = cv2.imread(os.path.join(path, item))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (50, 50))
            dataset.append([img, directory])

    data, labels = zip(*dataset)
    print("len= ",np.array(labels))
    
    #converting to vector for classification model
    labels = list(map(code_conv, labels))
    labels = np_utils.to_categorical(labels)

    #model architecture
    model = Sequential()
    model.add(SqueezeNet(input_shape=(50, 50, 3), include_top=False))
    model.add(Dropout(0.4))
    model.add(Conv2D(32, (1, 1), padding='valid', activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(29, activation='softmax'))

    print(model.summary())
    model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',  metrics=['accuracy'])
    model.fit(np.array(data), np.array(labels), batch_size=16, epochs=10, verbose=1)                  

    model.save("my-model.h5")

if __name__=="__main__":
    make_model()