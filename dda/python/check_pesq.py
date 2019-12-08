#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import librosa
import numpy as np 
from utils import search_wav
from pypesq import pypesq

wav_files = search_wav('../fin_data/enhanced/DDAE_0720/Target/')


pesq1 = np.array([]); pesq2 = np.array([]);

for f in wav_files:
    file = f.split('Target/')[-1]
    if file[0]=='1': #only for the 15dB files
        ref, sr = librosa.load('../fin_data/enhanced/DDAE_0720/Target/'+file , sr=16000, mono=True)
        #
        y_noisy, sr = librosa.load('../fin_data/enhanced/DDAE_0720/Source/' + file, sr=16000, mono=True)
        pesq1 = np.append(pesq1, pypesq(sr, ref,y_noisy, 'wb'))
        #
        y_enh, sr = librosa.load('../fin_data/enhanced/DDAE_0720/REG/'+file, sr=16000, mono=True)
        pesq2 = np.append(pesq2, pypesq(sr, ref,y_enh, 'wb'))
        


#err1 = np.mean(err1); err2 = np.mean(err2);