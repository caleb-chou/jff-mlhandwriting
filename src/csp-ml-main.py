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
epochs = 3
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
directory = "resources/"
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

#create()
#train()
#save()
load()
evaluate()



