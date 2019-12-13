#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 23:23:28 2019

"""
from keras.models import Sequential, load_model 
from keras.layers import Input, Dense, BatchNormalization, Dropout
#from keras import backend as K
from keras.callbacks.callbacks import EarlyStopping
from keras.optimizers import  Adam
from scipy.io import loadmat
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import joblib
tf.compat.v1.enable_eager_execution()

data = loadmat('ex1data.mat')
training_data = joblib.load('training_data.joblib')
testing_data = joblib.load('testing_data.joblib')
train_data = training_data['train_data'];
train_class = training_data['train_class'];
test_data = testing_data['test_data'];
test_class = testing_data['test_class'];
tw1 = data['tw1']; tw2 = data['tw2']; tw3 = data['tw3']; 
phonemes = training_data['phonemes']; 
train_data = train_data.T; test_data = test_data.T; 

dim=26; 
epochs=100
network = Sequential()
output_layers=len(phonemes);
#do_value=0.2; 
train_output = np.zeros([len(train_class), output_layers])
test_output = np.zeros([len(test_class), output_layers])


for i in range(len(train_class)):
    train_output[i, int(train_class[i])] = 1
    
for i in range(len(test_class)):
    test_output[i, int(test_class[i])] = 1
    
    
train_data, train_output = shuffle(train_data, train_output)


network.add(Dense(256, activation = 'relu', input_dim=dim)); 
network.add(Dropout(0.2))
network.add(BatchNormalization())
network.add(Dense(256, activation = 'relu'));
network.add(Dropout(0.2))
network.add(BatchNormalization()) 
network.add(Dense(256, activation = 'relu'));
network.add(Dropout(0.2)) 
network.add(BatchNormalization())


network.add(Dense(output_layers, activation = 'softmax')); 

opt  = Adam(learning_rate=0.001)
network.compile(optimizer=opt, loss='mse',metrics=['accuracy'])



print('Number of parameters is', network.count_params())
#Following line ensures that training stops when overfitting is seen.
early_stopping = EarlyStopping(monitor='val_loss', patience=4);

history = network.fit(train_data,train_output,epochs=epochs, batch_size=500,  validation_data = (test_data, test_output));

op1 = network.predict(tw1)
ind1 = [np.argmax(op1, axis=1)]
guess1 = [phonemes[int(i)] for i in np.array(ind1)[0]]
guess1 = ''.join(guess1)

op2 = network.predict(tw2)
ind2 = [np.argmax(op2, axis=1)]
guess2 = [phonemes[int(i)] for i in np.array(ind2)[0]]
guess2 = ''.join(guess2)


op3 = network.predict(tw3)
ind3 = [np.argmax(op3, axis=1)]
guess3 = [phonemes[int(i)] for i in np.array(ind3)[0]]
guess3 = ''.join(guess3)

test_pred = network.predict(test_data);
test_pred = np.array([np.argmax(test_pred, axis=1)])
y_pred =  [phonemes[int(i)] for i in np.array(test_pred[0,:]) ]
y_true =  [phonemes[int(i)] for i in np.array(test_class) ]

cm_test = confusion_matrix(y_true, y_pred, labels = phonemes);
                   

train_data = training_data['train_data'];
train_data = train_data.T;
train_pred = network.predict(train_data);
train_pred = np.array([np.argmax(train_pred, axis=1)])
y_pred =  [phonemes[int(i)] for i in np.array(train_pred[0,:]) ]
y_true =  [phonemes[int(i)] for i in np.array(train_class) ]

cm_train = confusion_matrix(y_true, y_pred, labels = phonemes)

network.save('phoneme_recognizer.h5')


