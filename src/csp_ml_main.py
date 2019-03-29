# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:15:44 2019

@author: caleb
"""
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from PIL import Image
import pickle
import datetime as dt
import time as t
import os.path
    
class Recognizer:
    date = dt.datetime.fromtimestamp(t.time()).strftime('%Y-%m-%d-%H-%M-%S')
    directory = os.path.abspath(os.path.pardir) + "/resources/"
    pkl_filename = directory + "2019-03-27-14-10-46-hw_model.pkl"
    
    def __init__(self, model = "defaultmodel.pkl"):
        self.pkl_filename = self.directory + model
        
    (train_X,train_Y), (test_X,test_Y) = mnist.load_data()
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
    
    
    train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
    
    train_X.shape,valid_X.shape,train_label.shape,valid_label.shape
    

    
    batch_size = 64
    epochs = 20
    num_classes = 10
    hw_model = None
    
    def create(self):
        self.hw_model = Sequential()
        self.hw_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
        self.hw_model.add(LeakyReLU(alpha=0.1))
        self.hw_model.add(MaxPooling2D((2, 2),padding='same'))
        self.hw_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
        self.hw_model.add(LeakyReLU(alpha=0.1))
        self.hw_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        self.hw_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
        self.hw_model.add(LeakyReLU(alpha=0.1))                  
        self.hw_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        self.hw_model.add(Flatten())
        self.hw_model.add(Dense(128, activation='linear'))
        self.hw_model.add(LeakyReLU(alpha=0.1))                  
        self.hw_model.add(Dense(self.num_classes, activation='softmax'))
    
    def train(self):    
        self.hw_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
        self.hw_model.fit(self.train_X, self.train_label, batch_size=self.batch_size,epochs=self.epochs,verbose=1,validation_data=(self.valid_X, self.valid_label))
    
    def save(self):
        with open(self.pkl_filename, 'wb') as file:  
            pickle.dump(self.hw_model, file)
    
    def load(self):
        with open(self.pkl_filename, 'rb') as file:  
            self.hw_model = pickle.load(file)
            
    def evaluate(self):
        test = self.hw_model.evaluate(self.test_X, self.test_Y_one_hot, verbose=1)
        print('Test loss:', test[0])
        print('Test accuracy:', test[1])
        

    def predict_image_from_path(self,image = "temp.png"):
        img = Image.open(self.directory + image).convert('L')
        img = np.resize (img, (28,28,1))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(-1,28,28,1)
        prediction = self.hw_model.predict_classes(im2arr)
        return prediction[0]
    
    def predict_image(self,image):
        image = image.convert('L')
        image = np.resize(image, (28,28,1))
        image = np.array(image)
        image = image.reshape(-1,28,28,1)
        prediction = self.hw_model.predict_classes(image)
        return prediction[0]

#m = Recognizer()
#m.load()
#m.train()
#m.save()
#m.evaluate()