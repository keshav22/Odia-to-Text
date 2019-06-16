# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 02:20:27 2018

@author: keshav
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 23:11:34 2018

@author: keshav
"""

import pyaudio
import wave
from array import array
#from matplotlib import pyplot as plt
FORMAT=pyaudio.paInt16
CHANNELS=2
RATE=16000
CHUNK=1024
RECORD_SECONDS=6
FILE_NAME="sentence"

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
    frames.append(data)             
    
wavfile=wave.open(FILE_NAME+".wav",'wb')
wavfile.setnchannels(CHANNELS)
wavfile.setsampwidth(audio.get_sample_size(FORMAT))
wavfile.setframerate(RATE)
wavfile.writeframes(b''.join(frames))
wavfile.close()
            

stream.stop_stream()
stream.close()
audio.terminate()
#Importing pyplot


#Plotting to our canvas
#plt.plot(data_chunk)รณ

#Showing what we plotted
#plt.show()
