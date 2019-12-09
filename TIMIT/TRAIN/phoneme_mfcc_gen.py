#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 22:32:30 2019

@author: meghna
"""

import  glob, librosa, joblib
import pandas as pd
import numpy as np

files = []
for file in glob.glob("**/**/*.PHN"):
    files.append(file[0:-4])

    
phonemes = {}
phoneme_count = {}
    
for file in files: 
    df = pd.read_csv(file + '.PHN', delimiter=' ')
    df = list(df.values)
    y, sr = librosa.load(file+'.WAV', sr=16000, mono=True)
    for p in df: 
        snippet = y[p[0]: p[1]];
        mfcc = librosa.feature.mfcc(snippet, win_length=512, n_fft=512,
            hop_length=256,
            sr = sr, n_mfcc = 13)
        if(np.shape(mfcc)[1]<9):
            delta = librosa.feature.delta(mfcc, mode = 'nearest' );
        else:
            delta = librosa.feature.delta(mfcc);        
        mfcc_new = np.concatenate([mfcc, delta]); 
        if not p[2] in phonemes:
            phoneme_count[p[2]] = 0
            phonemes[p[2]] = mfcc_new;
        else:
            phoneme_count[p[2]] = phoneme_count[p[2]] + 1;
            phonemes[p[2]] = np.concatenate((phonemes[p[2]], mfcc_new), axis=1)
        


#collapsing of phoneme classes. ref: Lopes, Carla, and Fernando Perdigao. "Phone recognition on the TIMIT database." Speech Technologies/Book 1 (2011): 285-302.
phonemes.pop('q');

phonemes['aa'] = np.concatenate((phonemes['aa'], phonemes['ao']), axis=1)
phoneme_count['aa'] = sum((phoneme_count['aa'], phoneme_count['ao']))
phonemes.pop('ao');

phonemes['ah'] = np.concatenate((phonemes['ah'], phonemes['ax'],  phonemes['ax-h']), axis=1)
phoneme_count['ah'] = sum((phoneme_count['ah'], phoneme_count['ax'],  phoneme_count['ax-h']))
phonemes.pop('ax'); phonemes.pop('ax-h');

phonemes['er'] = np.concatenate((phonemes['er'], phonemes['axr']), axis=1)
phoneme_count['er'] = sum((phoneme_count['er'], phoneme_count['axr']))
phonemes.pop('axr');

phonemes['hh'] = np.concatenate((phonemes['hh'], phonemes['hv']), axis=1)
phoneme_count['hh'] = sum((phoneme_count['hh'], phoneme_count['hv']))
phonemes.pop('hv');

phonemes['ih'] = np.concatenate((phonemes['ih'], phonemes['ix']), axis=1)
phoneme_count['ih'] = sum((phoneme_count['ih'], phoneme_count['ix']))
phonemes.pop('ix');

phonemes['l'] = np.concatenate((phonemes['l'], phonemes['el']), axis=1)
phoneme_count['l'] = sum((phoneme_count['l'], phoneme_count['el']))
phonemes.pop('el');

phonemes['m'] = np.concatenate((phonemes['m'], phonemes['em']), axis=1)
phoneme_count['m'] = sum((phoneme_count['m'], phoneme_count['em']))
phonemes.pop('em');

phonemes['n'] = np.concatenate((phonemes['n'], phonemes['en'],  phonemes['nx']), axis=1)
phoneme_count['n'] = sum((phoneme_count['n'], phoneme_count['en'],  phoneme_count['nx']))
phonemes.pop('en'); phonemes.pop('nx');

phonemes['ng'] = np.concatenate((phonemes['ng'], phonemes['eng']), axis=1)
phoneme_count['ng'] = sum((phoneme_count['ng'], phoneme_count['eng']))
phonemes.pop('eng');

phonemes['sh'] = np.concatenate((phonemes['sh'], phonemes['zh']), axis=1)
phoneme_count['sh'] = sum((phoneme_count['sh'], phoneme_count['zh']))
phonemes.pop('zh');

phonemes['uw'] = np.concatenate((phonemes['uw'], phonemes['ux']), axis=1)
phoneme_count['uw'] = sum((phoneme_count['uw'], phoneme_count['ux']))
phonemes.pop('ux');

phonemes['sil'] = np.concatenate((phonemes['pcl'], phonemes['tcl'],  phonemes['kcl'], phonemes['bcl'], phonemes['gcl'],  phonemes['dcl'], phonemes['h#'], phonemes['pau'],  phonemes['epi']), axis=1)
phoneme_count['sil'] = sum((phoneme_count['pcl'], phoneme_count['tcl'],  phoneme_count['kcl'], phoneme_count['bcl'], phoneme_count['gcl'],  phoneme_count['dcl'], phoneme_count['h#'], phoneme_count['pau'],  phoneme_count['epi']))
phonemes.pop('pcl'); phonemes.pop('tcl'); phonemes.pop('kcl'); phonemes.pop('bcl'); phonemes.pop('gcl'); phonemes.pop('dcl'); phonemes.pop('h#'); phonemes.pop('epi'); phonemes.pop('pau');

            
            
phoneme_labels = list(phonemes);   #Comment if 'phoneme_labels.joblib' exists
joblib.dump( phoneme_labels, 'phoneme_labels.joblib');  #Comment if 'phoneme_labels.joblib' exists 

#phoneme_labels = joblib.load('phoneme_labels.joblib')  #Unomment if 'phoneme_labels.joblib' exists 

min_num = 10e10;
for p in phonemes:
    if(min_num>np.shape(phonemes[p])[1]):
        min_num = np.shape(phonemes[p])[1]
        
train_data = np.zeros((26,1)); train_class = []; 

for p in phonemes: 
    np.random.shuffle(phonemes[p].T);
    train_data = np.concatenate((train_data, phonemes[p][:,0:min_num]), axis=1)
    train_class = np.append(train_class, np.repeat(phoneme_labels.index(p), min_num))
    
train_data = np.delete(train_data, 0, axis=1);
training_data = {'train_data':train_data, 'train_class': train_class, 'phonemes':phoneme_labels}
joblib.dump(training_data, 'training_data.joblib')

    
    
    

        
    