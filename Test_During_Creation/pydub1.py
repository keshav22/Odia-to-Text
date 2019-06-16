# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 20:47:46 2018

@author: keshav
"""

from pydub import AudioSegment
from pydub.silence import split_on_silence

sound_file = AudioSegment.from_wav("text.wav")
audio_chunks = split_on_silence(sound_file, 
    
    min_silence_len=75,

    silence_thresh=-47
)

for i, chunk in enumerate(audio_chunks):

    out_file = "chunk{0}.wav".format(i)
  #  print "exporting", out_file
    chunk.export(out_file, format="wav")


