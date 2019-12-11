#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:13:55 2019

@author: meghna
"""
import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy

NUM_FFT =  512
wavfile = 'data/train_data/clean/S03/S03W000.wav' 
sr = 16000;
y, sr = librosa.load(wavfile, sr, mono=True)
mfcc = librosa.feature.mfcc(y, win_length=512, n_fft=512,
            hop_length=256,
            sr = sr, n_mfcc = 13)
delta = librosa.feature.delta(mfcc);
mfcc_new = np.concatenate([mfcc, delta]);

spec = librosa.stft(y,
                     n_fft=NUM_FFT,
                     hop_length=hop_length,
                     win_length=512,
                     window=scipy.signal.hann)

mel = librosa.feature.melspectrogram(y,
                     n_fft=NUM_FFT,
                     hop_length=hop_length,
                     win_length=512,
                     window=scipy.signal.hann)


mfcc = librosa.feature.mfcc(y, win_length=512, n_fft=512,
         hop_length=256,
         sr = sr, n_mfcc = 13)


w_s =     librosa.istft(spec,
                           hop_length=256,
                           win_length=512,
                           window=scipy.signal.hann)

librosa.output.write_wav('w_s.wav', w_s, sr)

w_m = librosa.feature.inverse.mel_to_audio(mel, sr=16000)
librosa.output.write_wav('w_m.wav', w_m, sr)


w_m1 = librosa.feature.inverse.mel_to_audio(mel, sr = 16000,
                          hop_length=256,
                          win_length=512,
                          window=scipy.signal.hann)
librosa.output.write_wav('w_m1.wav', w_m1, sr)


w_mfcc = librosa.feature.inverse.mfcc_to_audio(mfcc, sr=16000)
librosa.output.write_wav('w_mfcc.wav', w_mfcc, sr)


