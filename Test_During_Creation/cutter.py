# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 23:11:34 2018

@author: keshav
"""

import pyaudio
import wave
from array import array
from matplotlib import pyplot as plt
FORMAT=pyaudio.paInt16
CHANNELS=2
RATE=44100
CHUNK=1024
RECORD_SECONDS=3
FILE_NAME="RECORDING"

audio=pyaudio.PyAudio() 

stream=audio.open(format=FORMAT,channels=CHANNELS, 
                  rate=RATE,
                  input=True,
                  frames_per_buffer=CHUNK)

frames=[]
i1=0
way=0
for i in range(0,int(RATE/CHUNK*RECORD_SECONDS)):
    data=stream.read(CHUNK)
    data_chunk=array('h',data)
    vol=max(data_chunk)
    if(vol>=6000):
        print("something said")
        frames.append(data)
             
        way=1
    else:
        plt.plot(data_chunk)
        plt.show()

        if way==1:
            wavfile=wave.open(FILE_NAME+str(i1)+".wav",'wb')
            wavfile.setnchannels(CHANNELS)
            wavfile.setsampwidth(audio.get_sample_size(FORMAT))
            wavfile.setframerate(RATE)
            wavfile.writeframes(b''.join(frames))
            wavfile.close()
            i1+=1
            frames = []
            way=0
        print("nothing")
    print("\n")



stream.stop_stream()
stream.close()
audio.terminate()
#Importing pyplot


#Plotting to our canvas
plt.plot(data_chunk)

#Showing what we plotted
plt.show()
