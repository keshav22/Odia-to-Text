# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 18:43:03 2018

@author: keshav
"""
#import os
import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
#import wave
#import sys
import librosa
#import glob
#from pydub import AudioSegment
#from scipy.io import wavfile
#from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)



#fs, data = wavfile.read('audio'+str(i)+'.wav')
#sc = StandardScaler()


#X_train = sc.fit_transform(data)
Y_train=[]
j=1
X_train=[]
X_test=[]
while j<145:
    Y_train.append('audio'+str(j)+'.wav')
    X_train.append(j)
    X_train[j-1]=[]
    X_test.append(j)
    X_test[j-1]=[]
    j=j+1
Y_train = np.array(Y_train, dtype='str')

i=1

while i<145:
    j=1    
    X, sample_rate = librosa.load('audio'+str(i)+'.wav', res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
    mfccs=mfccs.reshape(-1,1)
    while j<40:    
        X_train[i-1].append(mfccs[j][0])
        j=j+1
    i=i+1
    
classifier.fit(X_train, Y_train)

#mfccs=mfccs.transpose()
X, sample_rate1 = librosa.load('audio25.wav', res_type='kaiser_fast')
X = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate1, n_mfcc=40).T,axis=0) 
X=X.reshape(-1,1)
i=1
X_test=[]
#X_test[0]=X_test[0].reshape(-1,1)  
while i<40:
    X_test.append(X[i-1][0])
    i=i+1
X_test = np.array(X_test ,dtype='float64')
X_test=X_test.reshape(1,-1)
y_pred = classifier.predict(X_test)
    
  
X_test=X_test[0].transpose()     
