#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:33:22 2019

@author: meghna
"""

from utils import *
import numpy as np
import matplotlib.pyplot as plt

spec = wav2spec('data/train_data/clean/S03/S03W000.wav', 16e3);
mel = wav2melspec('data/train_data/clean/S03/S03W000.wav', 16e3);
mfcc = wav2mfcc('data/train_data/clean/S03/S03W000.wav', 16e3);

spec2wav('data/train_data/clean/S03/S03W000.wav', 16000, 'w_s.wav', spec);
melspec2wav('data/train_data/clean/S03/S03W000.wav', 16000, 'w_m.wav', mel);
mfccupload('data/train_data/clean/S03/S03W000.wav', 16000, 'w_mfcc.wav', mfcc);

w_s = np.loadtxt('w_s.mfc')
w_m = np.loadtxt('w_m.mfc')
w_mfcc = np.loadtxt('w_mfcc.mfc')

plt.imshow(w_s); plt.show()
plt.imshow(w_m); plt.show()
plt.imshow(w_mfcc); plt.show()