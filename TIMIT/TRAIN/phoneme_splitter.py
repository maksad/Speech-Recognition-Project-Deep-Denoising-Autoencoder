#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 22:32:30 2019

@author: meghna
"""

import  os, glob, librosa
import pandas as pd

files = []
for file in glob.glob("**/**/*.PHN"):
    files.append(file[0:-4])
    
if not os.path.exists('phonemes'):
    os.mkdir('phonemes')
    
phoneme_count = {}
    
for file in files: 
    df = pd.read_csv(file + '.PHN', delimiter=' ')
    df = list(df.values)
    y, sr = librosa.load(file+'.WAV', sr=16000, mono=True)
    for p in df: 
        snippet = y[p[0]: p[1]];
        if not os.path.exists('phonemes/'+p[2]):
            os.mkdir('phonemes/'+p[2])
            phoneme_count[p[2]] = 0
        else:
            phoneme_count[p[2]] = phoneme_count[p[2]] + 1;
        librosa.output.write_wav('phonemes/'+p[2]+'/'+str(phoneme_count[p[2]]) +'.wav', snippet, sr)
        

        
    