#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from keras.models import Sequential, load_model 
import tensorflow as tf
import librosa
import numpy as np 
import joblib
from utils import search_wav

wav_files = search_wav('../fin_data/enhanced/DDAE_0720/Target/')

#network = load_model('phoneme_recognizer.h5');
training_data = joblib.load('training_data.joblib') 
phonemes = training_data['phonemes']; 
err1 = np.array([]); err2 = np.array([]);

for f in wav_files:
    file = f.split('Target/')[-1]
    if file[0]=='1':
        y, sr = librosa.load('../fin_data/enhanced/DDAE_0720/Target/'+file , sr=16000, mono=True)
        mfcc = librosa.feature.mfcc(y, win_length=512, n_fft=512,
            hop_length=256,
            sr = sr, n_mfcc = 13)
        delta = librosa.feature.delta(mfcc);        
        mfcc_new = np.concatenate([mfcc, delta]); 
        test_pred = network.predict(mfcc_new.T);
        test_pred = np.array([np.argmax(test_pred, axis=1)])
        y_pred_target =  [phonemes[int(i)] for i in np.array(test_pred[0,:]) ]
        #
        y, sr = librosa.load('../fin_data/enhanced/DDAE_0720/Source/' + file, sr=16000, mono=True)
        mfcc = librosa.feature.mfcc(y, win_length=512, n_fft=512,
            hop_length=256,
            sr = sr, n_mfcc = 13)
        delta = librosa.feature.delta(mfcc);        
        mfcc_new = np.concatenate([mfcc, delta]); 
        test_pred = network.predict(mfcc_new.T);
        test_pred = np.array([np.argmax(test_pred, axis=1)])
        y_pred_source =  [phonemes[int(i)] for i in np.array(test_pred[0,:]) ]
        #
        y, sr = librosa.load('../fin_data/enhanced/DDAE_0720/REG/'+file, sr=16000, mono=True)
        mfcc = librosa.feature.mfcc(y, win_length=512, n_fft=512,
            hop_length=256,
            sr = sr, n_mfcc = 13)
        delta = librosa.feature.delta(mfcc);        
        mfcc_new = np.concatenate([mfcc, delta]); 
        test_pred = network.predict(mfcc_new.T);
        test_pred = np.array([np.argmax(test_pred, axis=1)])
        y_pred_enhanced =  [phonemes[int(i)] for i in np.array(test_pred[0,:]) ]
        #
        err1 = np.append(err1, np.mean(np.array(y_pred_source)!=np.array(y_pred_target)))
        err2 = np.append(err2, np.mean(np.array(y_pred_enhanced)!=np.array(y_pred_target)) )


#err1 = np.mean(err1); err2 = np.mean(err2);