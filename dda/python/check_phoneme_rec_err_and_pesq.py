#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from keras.models import Sequential, load_model 
import tensorflow as tf
import librosa
import numpy as np 
import joblib
from utils import search_wav
import os
from pypesq import pypesq
import matplotlib.pyplot as plt

wav_files = search_wav('data/test_data/noisy')
filenames_source = {}; filenames_100k_wo_clean = {}; filenames_200k_noisyclean = {}; filenames_basemodel150 = {}; filenames_200k_cleandata_only = {}; 
network = load_model('phoneme_recognizer.h5');
training_data = joblib.load('training_data.joblib') 
phonemes = training_data['phonemes']; 

filenames = {}

for f in wav_files:
    file = f.split('noisy/')[-1]
    y, sr = librosa.load('data/test_data/clean/'+file[-11:] , sr=16000, mono=True)
    ref = y;
    mfcc = librosa.feature.mfcc(y, win_length=512, n_fft=512,
        hop_length=256,
        sr = sr, n_mfcc = 13)
    delta = librosa.feature.delta(mfcc);        
    mfcc_new = np.concatenate([mfcc, delta]); 
    test_pred = network.predict(mfcc_new.T);
    test_pred = np.array([np.argmax(test_pred, axis=1)])
    y_pred_target =  [phonemes[int(i)] for i in np.array(test_pred[0,:]) ]
        #
    y, sr = librosa.load('data/test_data/noisy/'+file , sr=16000, mono=True)
    mfcc = librosa.feature.mfcc(y, win_length=512, n_fft=512,
        hop_length=256,
        sr = sr, n_mfcc = 13)
    delta = librosa.feature.delta(mfcc);        
    mfcc_new = np.concatenate([mfcc, delta]); 
    test_pred = network.predict(mfcc_new.T);
    test_pred = np.array([np.argmax(test_pred, axis=1)])
    y_pred_source =  [phonemes[int(i)] for i in np.array(test_pred[0,:]) ]
    err = np.mean(np.array(y_pred_source)!=np.array(y_pred_target))
    pesq =  pypesq(sr, ref, y, 'wb')
    filenames_source[file] = (err, pesq)
        #
    if(os.path.exists('data/enhanced_100k_without_clean/' + file)):
        y, sr = librosa.load('data/enhanced_100k_without_clean/' + file, sr=16000, mono=True)
        mfcc = librosa.feature.mfcc(y, win_length=512, n_fft=512,
            hop_length=256,
            sr = sr, n_mfcc = 13)
        delta = librosa.feature.delta(mfcc);        
        mfcc_new = np.concatenate([mfcc, delta]); 
        test_pred = network.predict(mfcc_new.T);
        test_pred = np.array([np.argmax(test_pred, axis=1)])
        y_pred_100k_wo_clean =  [phonemes[int(i)] for i in np.array(test_pred[0,:]) ]
        err = np.mean(np.array(y_pred_100k_wo_clean)!=np.array(y_pred_target)) 
        pesq = pypesq(sr, ref, y, 'wb')
        filenames_100k_wo_clean[file] = (err, pesq)
        #
    if(os.path.exists('data/enhanced_200k_with_noisy_and_clean/' + file)):
        y, sr = librosa.load('data/enhanced_200k_with_noisy_and_clean/'+file, sr=16000, mono=True)
        mfcc = librosa.feature.mfcc(y, win_length=512, n_fft=512,
            hop_length=256,
            sr = sr, n_mfcc = 13)
        delta = librosa.feature.delta(mfcc);        
        mfcc_new = np.concatenate([mfcc, delta]); 
        test_pred = network.predict(mfcc_new.T);
        test_pred = np.array([np.argmax(test_pred, axis=1)])
        y_pred_200k_noisyclean =  [phonemes[int(i)] for i in np.array(test_pred[0,:]) ]
        err = np.mean(np.array(y_pred_200k_noisyclean)!=np.array(y_pred_target)) 
        pesq = pypesq(sr, ref, y, 'wb')
        filenames_200k_noisyclean[file] = (err, pesq)
        #
    if(os.path.exists('data/enhanced_trained_on_base_model_150/' + file)):
        y, sr = librosa.load('data/enhanced_trained_on_base_model_150/' + file, sr=16000, mono=True)
        mfcc = librosa.feature.mfcc(y, win_length=512, n_fft=512,
            hop_length=256,
            sr = sr, n_mfcc = 13)
        delta = librosa.feature.delta(mfcc);        
        mfcc_new = np.concatenate([mfcc, delta]); 
        test_pred = network.predict(mfcc_new.T);
        test_pred = np.array([np.argmax(test_pred, axis=1)])
        y_pred_basemodel150 =  [phonemes[int(i)] for i in np.array(test_pred[0,:]) ]
        err = np.mean(np.array(y_pred_basemodel150)!=np.array(y_pred_target)) 
        pesq = pypesq(sr, ref, y, 'wb')
        filenames_basemodel150[file] = (err, pesq)
        #
    if(os.path.exists('data/enhanced_trained_with_clean_data_only/'+file)):
        y, sr = librosa.load('data/enhanced_trained_with_clean_data_only/'+file, sr=16000, mono=True)
        mfcc = librosa.feature.mfcc(y, win_length=512, n_fft=512,
            hop_length=256,
            sr = sr, n_mfcc = 13)
        delta = librosa.feature.delta(mfcc);        
        mfcc_new = np.concatenate([mfcc, delta]); 
        test_pred = network.predict(mfcc_new.T);
        test_pred = np.array([np.argmax(test_pred, axis=1)])
        y_pred_200k_cleandata_only =  [phonemes[int(i)] for i in np.array(test_pred[0,:]) ]
        err = np.mean(np.array(y_pred_200k_cleandata_only)!=np.array(y_pred_target)) 
        pesq = pypesq(sr, ref, y, 'wb')    
        filenames_200k_cleandata_only[file] = (err, pesq)
        #
        
        
results = {'Source files':filenames_source, '100k without clean': filenames_100k_wo_clean, '200k noisy & clean':filenames_200k_noisyclean, '200k clean data only':filenames_200k_cleandata_only, 'Base Model 150':filenames_basemodel150}





pesq0 = []; pesq8 = []; err0 = []; err8 = [];
#results = joblib.load('Spec-results.joblib')
#8dB results 
for key in results:
    values = np.mean(np.array(list(results[key].values())))
    pesq8_tmp = np.array([]); pesq0_tmp = np.array([]);
    err8_tmp = np.array([]); err0_tmp = np.array([]);
    filenames = list(results[key].keys())
    for file in filenames:
        if file[0] == '8':
            pesq8_tmp = np.append(pesq8_tmp, results[key][file][1])
            err8_tmp = np.append(err8_tmp, results[key][file][0])
        else:
            pesq0_tmp = np.append(pesq0_tmp, results[key][file][1])
            err0_tmp = np.append(err0_tmp, results[key][file][0])
    pesq0.append(np.mean(pesq0_tmp)); pesq8.append(np.mean(pesq8_tmp)); 
    err0.append(np.mean(err0_tmp)); err8.append(np.mean(err8_tmp)); 
#0dB results
        
N = 5

ind = np.arange(N) 
width = 0.35       
plt.bar(ind, pesq0, width, label='PESQ 0dB')
plt.bar(ind + width, pesq8, width,
    label='PESQ 8dB')

plt.ylabel('PESQ values')

plt.xticks(ind + width / 2, results.keys())
plt.legend(loc='best')
#plt.axis().set_aspect('equal', 'datalim')
plt.xticks(rotation=45, ha="right")
plt.savefig('figures/PESQ.png', dpi=500, bbox_inches='tight' )
plt.show()


N = 5

ind = np.arange(N) 
width = 0.35       
plt.bar(ind, err0, width, label='Error 0dB')
plt.bar(ind + width, err8, width,
    label='Error 8dB')

plt.ylabel('Phoneme error percentage')

plt.xticks(ind + width / 2, results.keys())
plt.legend(loc='best')
#plt.axis().set_aspect('equal', 'datalim')
plt.xticks(rotation=45, ha="right")
plt.savefig('figures/Phoneme_error.png', dpi=500, bbox_inches='tight'   )
plt.show()
#err1 = np.mean(err1); err2 = np.mean(err2);