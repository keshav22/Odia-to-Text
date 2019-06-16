# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 04:10:18 2018

@author: keshav
"""

import librosa
import math
d,sr=librosa.load('text.wav')

dbfs=[]
cnt=0;
for i in d:
    if i<0:
        i=i*-1;
    if i==0.0:
        continue;
    b=math.log10(i);
    dbfs.append(20*b);
    cnt=cnt+1;
    
mean=numpy.mean(dbfs)