# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:15:44 2019

@author: caleb
"""

from keras.datasets import mnist
(train_X,train_Y), (test_X,test_Y) = mnist.load_data()

from keras.utils import to_categorical
import numpy as np
classes = np.unique(train_Y)
num_classes = len(classes)

train_X = train_X.reshape(-1, 28,28, 1)
test_X = test_X.reshape(-1, 28,28, 1)
train_X.shape, test_X.shape

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

train_X.shape,valid_X.shape,train_label.shape,valid_label.shape

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU

batch_size = 64
epochs = 10
num_classes = 10
hw_model = None

def create():
    global hw_model
    hw_model = Sequential()
    hw_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
    hw_model.add(LeakyReLU(alpha=0.1))
    hw_model.add(MaxPooling2D((2, 2),padding='same'))
    hw_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    hw_model.add(LeakyReLU(alpha=0.1))
    hw_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    hw_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    hw_model.add(LeakyReLU(alpha=0.1))                  
    hw_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    hw_model.add(Flatten())
    hw_model.add(Dense(128, activation='linear'))
    hw_model.add(LeakyReLU(alpha=0.1))                  
    hw_model.add(Dense(num_classes, activation='softmax'))

def train():    
    hw_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    hw_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

import pickle
import datetime as dt
import time as t

date = dt.datetime.fromtimestamp(t.time()).strftime('%Y-%m-%d-%H-%M-%S')
#pkl_filename = date + "-hw_model.pkl"
import os.path

directory = os.path.abspath(os.path.pardir) + "/resources/"
pkl_filename = directory + "2019-03-27-14-10-46-hw_model.pkl"

def save():
    with open(pkl_filename, 'wb') as file:  
        global hw_model
        pickle.dump(hw_model, file)

def load():
    with open(pkl_filename, 'rb') as file:  
        global hw_model
        hw_model = pickle.load(file)
        
def evaluate():
    global hw_model
    test = hw_model.evaluate(test_X, test_Y_one_hot, verbose=1)
    print('Test loss:', test[0])
    print('Test accuracy:', test[1])
    
from PIL import Image
def predict_image():
    global directory
    global hw_model
    img = Image.open(directory + 'temp.png').convert('L')
    img = np.resize (img, (28,28,1))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(-1,28,28,1)
    prediction = hw_model.predict_classes(im2arr)
    print(prediction)
#create()
#train()
#save()
load()
#train()
#save()
#evaluate()
predict_image()
'''
predicted_classes = hw_model.predict(test_X)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
predicted_classes.shape, test_Y.shape

print(predicted_classes)



correct = np.where(predicted_classes==test_Y)[0]
print ("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
    plt.tight_layout()

incorrect = np.where(predicted_classes!=test_Y)[0]
print ("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
    plt.tight_layout()
'''