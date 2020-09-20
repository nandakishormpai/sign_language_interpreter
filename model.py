
import cv2
import os
import numpy as np
from keras_squeezenet import SqueezeNet
from keras.optimizers import Adam
from tensorflow.python.keras.utils import np_utils
from keras.layers import Dropout, GlobalAveragePooling2D, Conv2D, Dense,   MaxPooling2D, Flatten
from keras.models import Sequential
import tensorflow as tf
from tensorflow.keras import layers

IMG_SAVE_PATH = os.path.join("asl_alphabet_train","asl_alphabet_train")

CODES = {
    "nothing": 0
}

alpha="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
for i in range (1,27):
    CODES[alpha[i-1]]=i

CODES["del"]=27
CODES["space"]=28

print(CODES)


NUM_CLASSES = len(CODES)


def code_conv(val):
    return CODES[val]


dataset = []
for directory in os.listdir(IMG_SAVE_PATH):
    path = os.path.join(IMG_SAVE_PATH, directory)
    if not os.path.isdir(path):
    continue
    # if len(dataset)>0:
      # print(dataset[-1][0],"\n")
    for item in os.listdir(path):

    if item.startswith("."):
        continue
    img = cv2.imread(os.path.join(path, item))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (50, 50))
    dataset.append([img, directory])

data, labels = zip(*dataset)
 

labels = list(map(code_conv, labels))
labels = np_utils.to_categorical(labels)


model = Sequential()

model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=(50,50,3)))
model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))
model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu'))
model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("\n done \n")


model.fit(np.array(data), np.array(labels), batch_size=64, epochs=10, verbose=1)                   

model.save("RPS-model.h5")